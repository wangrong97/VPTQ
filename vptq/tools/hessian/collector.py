# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Offline Hessian (and inverse Hessian) collection for VPTQ quantization.

Follows the QuIP# approach (quantize_llama/hessian_offline_llama.py): run
calibration data through the *unquantized* model, capture the input
activation of every linear layer with forward hooks, and accumulate the
second moment H = E[x x^T] per linear input. Linear layers that share the
same input tensor (e.g. q/k/v proj, gate/up proj in LLaMA-family models)
share a single Hessian to save memory and compute.

The collected Hessians are consumed by the VPTQ quantizer for:
  1. Hessian-weighted (weighted K-means) centroid initialization,
     using the diagonal h_ii as per-channel weights;
  2. GPTQ-style error propagation, which requires H^{-1};
  3. outlier channel detection (large Hessian diagonal values).
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

logger = logging.getLogger("vptq.hessian")

# Linear layers (matched by last name segment) that receive the *same*
# input tensor and therefore share one Hessian. Grouping is scoped by
# parent module: only members inside the SAME parent (e.g. one attention
# block, one MoE expert) are merged, so per-expert Hessians of different
# experts are never mixed. The first existing member in each group is
# hooked as the representative.
_SHARED_INPUT_GROUPS = (
    ("attn_in", ("q_proj", "k_proj", "v_proj")),
    ("mlp_in", ("gate_proj", "up_proj", "w1", "w3")),
)

# Common locations of the decoder layer list in HF causal LMs.
_DECODER_LAYER_PATHS = (
    "model.layers",  # llama / qwen / mistral / yi / deepseek
    "layers",  # native deepseek inference model (Transformer.layers)
    "transformer.h",  # gpt2
    "gpt_neox.layers",  # gpt-neox / pythia
    "model.decoder.layers",  # opt
    "transformer.layers",  # chatglm-style
)


def find_decoder_layers(
    model: nn.Module,
) -> Tuple[List[nn.Module], str]:
    """Locate the transformer decoder layer list of a causal LM."""
    for path in _DECODER_LAYER_PATHS:
        mod = model
        found = True
        for name in path.split("."):
            if hasattr(mod, name):
                mod = getattr(mod, name)
            else:
                found = False
                break
        if found and isinstance(mod, (nn.ModuleList, list, tuple)):
            if len(mod) > 0:
                logger.info("found %d decoder layers at `%s`", len(mod), path)
                return list(mod), path
    raise ValueError(
        "Cannot locate decoder layers. Tried: "
        f"{_DECODER_LAYER_PATHS}. This model architecture is not "
        "supported yet; please extend `_DECODER_LAYER_PATHS`."
    )


def build_hook_plan(
    layer_module: nn.Module,
    linear_types: Tuple[type, ...] = (nn.Linear,),
) -> Dict[str, str]:
    """Plan which linears to hook inside one decoder layer.

    Returns a dict mapping the full name of each *representative* linear
    to its Hessian group key. Linears sharing one input tensor within the
    same parent module (q/k/v proj of one attention block, gate/up or
    w1/w3 of one MoE expert) collapse into one group; every other linear
    gets its own group keyed by its full name. Group keys carry the
    parent prefix when the same pattern matches multiple parents (e.g.
    `ffn.experts.7.mlp_in`), so per-expert Hessians stay separate.

    Args:
        layer_module: One decoder layer.
        linear_types: Module classes treated as linear layers. Pass a
            model's custom Linear class for non-HF implementations.
    """
    linears = {
        name: mod
        for name, mod in layer_module.named_modules()
        if isinstance(mod, linear_types)
        or getattr(mod, "_vptq_hessian_hook", False)
    }
    by_parent: Dict[str, Dict[str, str]] = {}
    for name in linears:
        parent, _, leaf = name.rpartition(".")
        by_parent.setdefault(parent, {})[leaf] = name

    # parent -> group_key -> representative full name
    matches: Dict[str, Dict[str, str]] = {}
    for group_key, members in _SHARED_INPUT_GROUPS:
        for parent, leaves in by_parent.items():
            present = [m for m in members if m in leaves]
            if len(present) < 2:
                continue  # a single member shares nothing
            rep = leaves[present[0]]
            matches.setdefault(rep, {})
            matches[rep][group_key] = parent

    counts: Dict[str, int] = {}
    for rep_groups in matches.values():
        for group_key in rep_groups:
            counts[group_key] = counts.get(group_key, 0) + 1

    plan = {}
    for rep, rep_groups in matches.items():
        for group_key, parent in rep_groups.items():
            key = (
                f"{parent}.{group_key}"
                if counts[group_key] > 1
                else group_key
            )
            plan[rep] = key
    grouped_reps = set(plan)
    grouped_members = set()
    for group_key, members in _SHARED_INPUT_GROUPS:
        for parent, leaves in by_parent.items():
            present = [m for m in members if m in leaves]
            if len(present) >= 2 and leaves[present[0]] in grouped_reps:
                grouped_members.update(leaves[m] for m in present)
    for name in linears:
        if name not in grouped_members:
            plan[name] = name
    return plan


class HessianAccumulator:
    """Running accumulation of sum(x x^T), sum(x) and token count."""

    def __init__(
        self,
        n_features: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        self.n_features = n_features
        self.dtype = dtype
        self.device = device or torch.device("cpu")
        self.sum_xxt = torch.zeros(
            n_features, n_features, dtype=dtype, device=self.device
        )
        self.sum_x = torch.zeros(n_features, dtype=dtype, device=self.device)
        self.n_tokens = 0
        # True once update() has accumulated rows since construction or
        # restore; save() uses this to tell top-up groups (recompute) from
        # frozen ones (carry the previous entry, incl. its inv_hessian)
        self.updated = False

    @classmethod
    def from_saved(
        cls,
        hessian: torch.Tensor,
        mean: torch.Tensor,
        n_tokens: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> "HessianAccumulator":
        """Restore an accumulator from a saved (E[x x^T], E[x], n) entry.

        Exact inverse of finalize(): sum_xxt = hessian * n_tokens. Works
        with mmap-backed tensors from torch.load(..., mmap=True); the
        multiplication materializes a private copy, so the read-only
        storage is never written.
        """
        acc = cls(hessian.shape[0], dtype=dtype, device=device)
        acc.sum_xxt = hessian.to(device=acc.device, dtype=dtype) * n_tokens
        acc.sum_x = mean.to(device=acc.device, dtype=dtype) * n_tokens
        acc.n_tokens = int(n_tokens)
        return acc

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().reshape(-1, x.shape[-1])
        # guard: drop non-finite activation rows (e.g. 0/0 in MoE routing
        # weights can poison a few tokens); a Hessian accumulated over
        # finite rows stays finite
        finite = torch.isfinite(x).all(dim=-1)
        if not bool(finite.all()):
            x = x[finite]
            if x.shape[0] == 0:
                return
        x = x.to(device=self.device, dtype=self.dtype)
        self.sum_xxt.addmm_(x.t(), x)
        self.sum_x.add_(x.sum(dim=0))
        self.n_tokens += x.shape[0]
        self.updated = True

    def finalize(
        self, center: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return (E[x x^T], E[x], n_tokens).

        With `center=True`, returns the covariance form
        E[x x^T] - mu mu^T instead (QuIP# convention).
        """
        n = max(self.n_tokens, 1)
        hessian = self.sum_xxt / n
        mean = self.sum_x / n
        if center:
            hessian = hessian - torch.outer(mean, mean)
        return hessian, mean, self.n_tokens


def damped_inverse(
    hessian: torch.Tensor, damp: float = 0.01
) -> torch.Tensor:
    """Invert a Hessian with GPTQ-style damping, in float64.

    Dead (zero-diagonal) channels are set to 1, then
    `damp * mean(diag(H))` is added to the diagonal before a Cholesky
    factorization. The result is symmetrized and returned in float32.
    """
    H = hessian.to(torch.float64)
    dead = torch.diagonal(H) == 0
    if dead.any():
        H[dead, dead] = 1.0
    damp_value = damp * torch.mean(torch.diagonal(H))
    H.diagonal().add_(damp_value)
    L = torch.linalg.cholesky(H)
    inv = torch.cholesky_inverse(L)
    inv = (inv + inv.t()) / 2
    return inv.to(torch.float32)


class HessianCollector:
    """Register forward hooks and accumulate per-linear-input Hessians.

    Args:
        model: The unquantized causal LM (already on target devices).
        layer_range: Optional (start, end) slice of decoder layer indices
            to collect. Use ranges to shard collection for very large
            models whose Hessians do not fit in memory at once.
        accum_dtype: Accumulation dtype (fp32 default; fp64 is safer but
            2x memory and slower).
        accum_device: Device holding the accumulators. Defaults to the
            best available accelerator (CUDA > NPU > CPU). Accumulator
            memory is roughly (3 * hidden^2 + inter^2) * dtype_size per
            decoder layer for dense models; MoE models need per-expert
            Hessians on top.
        linear_types: Module classes treated as linear layers. Defaults
            to nn.Linear; pass custom classes (e.g. the Linear of
            DeepSeek's native inference code) for non-HF models.
        token_caps: Optional callable(group_key) -> Optional[int]. A
            group stops accumulating once its n_tokens reaches the cap
            (top-up mode); None return means uncapped. Combine with
            restore_state()/save(prev_payload=...) for incremental
            re-collection.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_range: Optional[Tuple[int, int]] = None,
        accum_dtype: torch.dtype = torch.float32,
        accum_device: Optional[torch.device] = None,
        linear_types: Tuple[type, ...] = (nn.Linear,),
        token_caps: Optional[callable] = None,
    ):
        self.model = model
        self.accum_dtype = accum_dtype
        if accum_device is None:
            from vptq.utils.device import default_device

            accum_device = default_device()
        self.accum_device = accum_device
        self.linear_types = linear_types
        # Optional per-group token cap: callable(group_key) -> int or
        # None. Groups at/above their cap stop accumulating (top-up mode:
        # cold experts collect until the floor, everything else is frozen
        # by cap 0). None disables capping entirely (original behavior).
        self.token_caps = token_caps
        # groups already at/above their cap *on disk* when restore_state
        # ran: they stay out of memory entirely, and hooks must skip
        # them even though no live accumulator exists to compare against
        self._frozen: Dict[int, set] = {}
        self.layers, self.layer_path = find_decoder_layers(model)
        start, end = layer_range or (0, len(self.layers))
        end = min(end, len(self.layers))
        self.layer_indices = list(range(start, end))
        self.accumulators: Dict[int, Dict[str, HessianAccumulator]] = {
            i: {} for i in self.layer_indices
        }
        self.hook_plans: Dict[int, Dict[str, str]] = {}
        self._handles = []
        self._register_hooks()
        logger.info(
            "collecting Hessians for layers [%d, %d) on %s in %s",
            start,
            end,
            self.accum_device,
            self.accum_dtype,
        )

    def _register_hooks(self) -> None:
        for idx in self.layer_indices:
            plan = build_hook_plan(
                self.layers[idx], linear_types=self.linear_types
            )
            self.hook_plans[idx] = plan
            submodules = dict(self.layers[idx].named_modules())
            for rep_name, group_key in plan.items():
                module = submodules[rep_name]
                handle = module.register_forward_hook(
                    self._make_hook(idx, group_key)
                )
                self._handles.append(handle)

    def _cap_reached(self, layer_idx: int, key: str) -> bool:
        """True if `key` should not accumulate any further.

        Either it was already satisfied on disk at restore time
        (`_frozen`), or its live accumulator has reached the cap.
        """
        if key in self._frozen.get(layer_idx, ()):
            return True
        if self.token_caps is None:
            return False
        cap = self.token_caps(key)
        if cap is None:
            return False
        acc = self.accumulators[layer_idx].get(key)
        n_have = acc.n_tokens if acc is not None else 0
        return n_have >= cap

    def _make_hook(self, layer_idx: int, group_key: str):
        def hook(module, inputs, output):
            x = inputs[0]
            if not isinstance(x, torch.Tensor) or x.dim() < 2:
                return
            group = self.accumulators[layer_idx]
            group_dim = getattr(module, "_vptq_group_dim", None)
            if group_dim is not None:
                # per-group Hessians (e.g. grouped projections consumed
                # via einsum): x -> (-1, G, d), one accumulator per group
                d = x.shape[-1]
                xg = x.detach().reshape(-1, x.shape[group_dim], d)
                # single bulk copy to the accumulator device + dtype:
                # per-group slices are small and latency-bound, so ~110
                # groups x 32 batches of individual .to() calls spent
                # their time in transfers, not addmm (the CPU-accumulator
                # path this replaces). Satisfied groups' rows ride along
                # — the extra bandwidth is far cheaper than the latency.
                if (
                    self.accum_device != xg.device
                    or self.accum_dtype != xg.dtype
                ):
                    xg = xg.to(
                        device=self.accum_device, dtype=self.accum_dtype
                    )
                for gi in range(xg.shape[1]):
                    sub_key = f"{group_key}.g{gi}"
                    if self._cap_reached(layer_idx, sub_key):
                        continue
                    acc = group.get(sub_key)
                    if acc is None:
                        acc = HessianAccumulator(
                            d,
                            dtype=self.accum_dtype,
                            device=self.accum_device,
                        )
                        group[sub_key] = acc
                    acc.update(xg[:, gi])
                return
            if self._cap_reached(layer_idx, group_key):
                return
            acc = group.get(group_key)
            if acc is None:
                acc = HessianAccumulator(
                    x.shape[-1],
                    dtype=self.accum_dtype,
                    device=self.accum_device,
                )
                group[group_key] = acc
            acc.update(x)

        return hook

    def restore_state(
        self, layer_idx: int, payload: dict
    ) -> List[str]:
        """Restore accumulators from a saved layer payload (mmap-friendly).

        Only groups still below their token cap *on disk* are restored;
        satisfied groups are recorded in `_frozen` and stay out of
        memory entirely — save(prev_payload=...) carries them through
        verbatim, which keeps the resident set small in top-up mode (a
        handful of cold experts instead of the full ~22GB per layer).

        Returns the list of restored group keys.
        """
        group = self.accumulators[layer_idx]
        restored = []
        for key, entry in payload.items():
            if self.token_caps is not None:
                cap = self.token_caps(key)
                if cap is not None and entry["n_tokens"] >= cap:
                    self._frozen.setdefault(layer_idx, set()).add(key)
                    continue
            group[key] = HessianAccumulator.from_saved(
                entry["hessian"],
                entry["mean"],
                entry["n_tokens"],
                dtype=self.accum_dtype,
                device=self.accum_device,
            )
            restored.append(key)
        return restored

    def pending_under_caps(self) -> List[str]:
        """Group keys still below their token cap (top-up progress)."""
        pending = []
        for group in self.accumulators.values():
            for key, acc in group.items():
                if self.token_caps is None:
                    continue
                cap = self.token_caps(key)
                if cap is not None and acc.n_tokens < cap:
                    pending.append(key)
        return pending

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @torch.no_grad()
    def collect(
        self, samples: torch.Tensor, batch_size: int = 1
    ) -> None:
        """Run forward passes over token samples and accumulate.

        Args:
            samples: LongTensor of shape (n_samples, seq_len).
            batch_size: Sequences per forward pass.
        """
        was_training = self.model.training
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        try:
            input_device = self.model.get_input_embeddings().weight.device
        except AttributeError:
            input_device = next(self.model.parameters()).device
        n_samples = samples.shape[0]
        for start in range(0, n_samples, batch_size):
            ids = samples[start : start + batch_size].to(input_device)
            self.model(input_ids=ids)
            done = min(start + batch_size, n_samples)
            if done % (batch_size * 10) == 0 or done == n_samples:
                logger.info("processed %d/%d samples", done, n_samples)
        if was_training:
            self.model.train()

    def save(
        self,
        output_dir: str,
        model_name: str = "",
        save_inv: bool = True,
        damp: float = 0.01,
        center: bool = False,
        extra_meta: Optional[dict] = None,
        prev_payload: Optional[dict] = None,
    ) -> None:
        """Write per-layer Hessian files and a meta.json.

        Each `layer_XXXX.pt` holds a dict: group key ->
        {"hessian", "mean", "n_tokens", ["inv_hessian"]}.

        With prev_payload (a previously saved layer payload, mmap-backed
        is fine), groups that were not updated since restore are carried
        over verbatim — including their inv_hessian — so a top-up run
        recomputes only what changed. Updated groups are finalized here;
        pass save_inv=False and run compute_inverse afterwards to fill
        just the missing inverses (it skips entries that already have
        one).
        """
        os.makedirs(output_dir, exist_ok=True)
        for idx in self.layer_indices:
            live = self.accumulators.get(idx, {})
            keys = set(live)
            if prev_payload is not None:
                keys |= set(prev_payload)
            payload = {}
            n_carried = 0
            for group_key in sorted(keys):
                acc = live.get(group_key)
                if acc is None or not acc.updated:
                    if prev_payload is not None and (
                        group_key in prev_payload
                    ):
                        # untouched group: keep the previous entry (and
                        # its inv_hessian) exactly as saved
                        payload[group_key] = prev_payload[group_key]
                        n_carried += 1
                        continue
                    if acc is None:
                        continue
                hessian, mean, n_tokens = acc.finalize(center=center)
                hessian = hessian.float().cpu()
                entry = {
                    "hessian": hessian,
                    "mean": mean.float().cpu(),
                    "n_tokens": n_tokens,
                }
                # release the accumulator (NPU-resident in top-up mode)
                # as soon as its finalized CPU copy is in the payload:
                # with the save running on a worker thread overlapping
                # the next layer's collector, holding all accumulators
                # until the end would double NPU Hessian residency
                live.pop(group_key, None)
                if save_inv:
                    entry["inv_hessian"] = damped_inverse(
                        hessian, damp=damp
                    )
                payload[group_key] = entry
            path = os.path.join(output_dir, f"layer_{idx:04d}.pt")
            n_recomputed = len(payload) - n_carried
            if n_recomputed == 0 and prev_payload is not None:
                # nothing changed on disk: every group is frozen/carried
                # — skip the 45GB rewrite entirely (matters in later
                # top-up rounds where most layers are already satisfied)
                logger.info(
                    "skipped %s (all %d groups carried, file unchanged)",
                    path,
                    len(payload),
                )
                continue
            # atomic write: the previous payload may still be mmap-mapped
            # onto this exact path (top-up mode) — truncating it in place
            # would SIGBUS the carried tensors mid-save; writing .tmp +
            # replace also leaves no torn file behind if the process is
            # killed mid-save
            tmp_path = path + ".tmp"
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
            logger.info(
                "saved %s (%d groups: %d recomputed, %d carried)",
                path,
                len(payload),
                n_recomputed,
                n_carried,
            )
        meta = {
            "model_name": model_name,
            "layer_path": self.layer_path,
            "layer_indices": self.layer_indices,
            "hook_plans": {
                str(i): plan for i, plan in self.hook_plans.items()
            },
            "accum_dtype": str(self.accum_dtype),
            "center": center,
            "damp": damp,
            "save_inv": save_inv,
            "convention": (
                "hessian = E[x x^T]"
                + (" - mean mean^T" if center else "")
            ),
        }
        if extra_meta:
            meta.update(extra_meta)
        with open(
            os.path.join(output_dir, "meta.json"), "w"
        ) as meta_file:
            json.dump(meta, meta_file, indent=2)
