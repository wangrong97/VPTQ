# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch

from vptq.tools.deepseek_v4 import ModelArgs, Transformer
from vptq.tools.deepseek_v4.kernels_torch import (
    act_quant,
    fp4_act_quant,
    hadamard_transform,
    hc_split_sinkhorn,
    sparse_attn,
)
from vptq.tools.deepseek_v4.model import Linear
from vptq.tools.hessian import HessianCollector


def _tiny_args():
    return ModelArgs(
        max_batch_size=2,
        max_seq_len=128,
        dtype="bf16",
        scale_fmt=None,
        scale_dtype="fp32",
        expert_dtype=None,
        vocab_size=512,
        dim=128,
        moe_inter_dim=64,
        n_layers=3,
        n_hash_layers=1,
        n_mtp_layers=1,
        n_heads=4,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=10.0,
        q_lora_rank=32,
        head_dim=128,
        rope_head_dim=64,
        o_groups=2,
        o_lora_rank=32,
        window_size=16,
        compress_ratios=(0, 4, 128, 4),  # n_layers + n_mtp_layers entries
        compress_rope_theta=160000,
        original_seq_len=0,
        rope_theta=10000.0,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=8,
        hc_mult=4,
        hc_sinkhorn_iters=20,
    )


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Transformer(_tiny_args())
    finally:
        torch.set_default_dtype(prev)
    # torch.empty() leaves garbage bits (possibly inf/nan in bf16);
    # the native model relies on checkpoint loading, so init sanely here.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.is_floating_point():
                continue
            if "norm" in name and param.dim() == 1:
                param.fill_(1.0)
            else:
                param.normal_(0, 0.02)
        for layer in model.layers:
            if layer.ffn.gate.hash:
                layer.ffn.gate.tid2eid.copy_(
                    torch.randint(
                        0,
                        8,
                        layer.ffn.gate.tid2eid.shape,
                        dtype=torch.int32,
                    )
                )
    return model.eval()


def test_sparse_attn_matches_naive():
    torch.manual_seed(0)
    b, s, h, d, n, t = 1, 24, 3, 32, 20, 6
    q = torch.randn(b, s, h, d, dtype=torch.bfloat16)
    kv = torch.randn(b, n, d, dtype=torch.bfloat16)
    sink = torch.randn(h)
    idx = torch.randint(0, n, (b, s, t), dtype=torch.int32)
    idx[:, :5] = -1  # mask some early rows entirely
    idx[0, 10, :2] = -1  # partial mask
    scale = d**-0.5

    out = sparse_attn(q, kv, sink, idx, scale)

    # naive reference
    ref = torch.zeros(b, s, h, d)
    for si in range(s):
        valid = idx[0, si][idx[0, si] != -1]
        if len(valid) == 0:
            continue
        scores = (
            q[0, si].float() @ kv[0, valid].float().T * scale
        )  # (h, k)
        m = torch.maximum(scores.max(-1).values, sink)
        p = torch.exp(scores - m.unsqueeze(-1))
        denom = p.sum(-1) + torch.exp(sink - m)
        ref[0, si] = (p / denom.unsqueeze(-1)) @ kv[0, valid].float()
    assert torch.allclose(out.float(), ref, atol=2e-2, rtol=1e-2)


def test_hc_split_sinkhorn_matches_naive():
    torch.manual_seed(0)
    hc, iters, eps = 4, 20, 1e-6
    mixes = torch.randn(2, 5, (2 + hc) * hc)
    scale = torch.randn(3)
    base = torch.randn((2 + hc) * hc)
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc, iters, eps)

    pre_ref = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post_ref = (
        2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1]
                          + base[hc : 2 * hc])
    )
    c = mixes[..., 2 * hc :].view(2, 5, hc, hc) * scale[2] + base[
        2 * hc :
    ].view(hc, hc)
    c = c.softmax(-1) + eps
    c = c / (c.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(-1, keepdim=True) + eps)
        c = c / (c.sum(-2, keepdim=True) + eps)
    assert torch.allclose(pre, pre_ref, atol=1e-6)
    assert torch.allclose(post, post_ref, atol=1e-6)
    assert torch.allclose(comb, c, atol=1e-5)


def test_act_quant_roundtrip_grid_values():
    # values exactly on the fp8 grid must survive quant+dequant
    x = torch.tensor(
        [[0.0, 1.0, -1.5, 2.0, 0.25, -0.75, 3.0, -2.5] * 8],
        dtype=torch.bfloat16,
    )
    y = act_quant(x.clone(), block_size=64, inplace=True)
    # same block => same scale; grid values preserved approximately
    assert torch.allclose(x.float(), y.float(), atol=2e-1)


def test_fp4_quant_grid():
    x = torch.tensor(
        [[0.0, 0.5, -1.0, 1.5, 2.0, -3.0, 4.0, -6.0] * 4],
        dtype=torch.bfloat16,
    )
    y = fp4_act_quant(x.clone(), block_size=32, inplace=True)
    assert torch.allclose(x.float(), y.float(), atol=1e-1)


def test_hadamard_matches_matrix_form():
    torch.manual_seed(0)
    d = 16
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat(
            [torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0
        )
    x = torch.randn(3, d, dtype=torch.bfloat16)
    y = hadamard_transform(x, scale=d**-0.5)
    ref = x.float() @ H * d**-0.5
    assert torch.allclose(y.float(), ref, atol=1e-1)


def test_tiny_model_forward_and_hessian():
    model = _tiny_model()
    samples = torch.randint(0, 512, (2, 128))
    with torch.inference_mode():
        logits = model(samples, 0)
    assert logits.shape == (2, 512)
    assert torch.isfinite(logits.float()).all()

    # collector hooks fire on the custom Linear class
    collector = HessianCollector(
        model,
        layer_range=(1, 2),
        accum_device=torch.device("cpu"),
        linear_types=(Linear,),
    )
    with torch.inference_mode():
        h = model.embed(samples)
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        for layer in model.layers:
            h = layer(h, 0, samples)
    collector.remove_hooks()
    groups = collector.accumulators[1]
    assert len(groups) > 0
    # per-expert groups must exist and stay separate
    expert_keys = [k for k in groups if "experts" in k]
    assert any("mlp_in" in k for k in expert_keys)
    for key, acc in groups.items():
        assert acc.n_tokens > 0, key
        hessian, mean, n = acc.finalize()
        assert hessian.shape == (acc.n_features, acc.n_features)
        assert torch.isfinite(hessian).all(), key


def test_hessian_matches_manual_for_shared_expert():
    model = _tiny_model(seed=1)
    samples = torch.randint(0, 512, (2, 128))

    captured = []

    def hook(module, inputs, output):
        captured.append(
            inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
        )

    handle = model.layers[1].ffn.shared_experts.w1.register_forward_hook(
        hook
    )
    collector = HessianCollector(
        model,
        layer_range=(1, 2),
        accum_device=torch.device("cpu"),
        linear_types=(Linear,),
    )
    with torch.inference_mode():
        h = model.embed(samples)
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        for layer in model.layers:
            h = layer(h, 0, samples)
    collector.remove_hooks()
    handle.remove()

    x = torch.cat(captured)
    ref = x.t() @ x / x.shape[0]
    acc = collector.accumulators[1]["ffn.shared_experts.mlp_in"]
    hessian, _, n = acc.finalize()
    assert n == x.shape[0]
    assert torch.allclose(hessian, ref, atol=1e-3)


def test_build_samples_rounds_are_disjoint(tmp_path):
    # regression for the 2026-09-18 incident: top-up "rounds" that only
    # changed the seed produced 100% identical samples (5 of 6 slices
    # are single-file, so seed-shuffling has no effect on their document
    # order) — repeated batches inflate n_tokens while H stays put.
    # build_samples(round_idx=k) must consume the k-th FRESH segment.
    import hashlib
    import json
    from types import SimpleNamespace

    from vptq.tools.deepseek_v4.collect_hessian import build_samples

    for marker in ("common_crawl", "c4"):  # two single-file slices
        with open(tmp_path / f"{marker}_test.jsonl", "w") as f:
            for i in range(800):
                f.write(
                    json.dumps({"text": f"{marker} doc {i}: " + "x" * (i % 7)})
                    + "\n"
                )

    class FakeTok:
        def __call__(self, text):
            h = hashlib.md5(text.encode()).digest()
            n = 8 + len(text) % 5  # 8..12 pseudo-tokens per doc
            return SimpleNamespace(input_ids=list(h[:n]))

    rounds = [
        build_samples(FakeTok(), str(tmp_path), 64, 16, seed=1, round_idx=k)
        for k in (0, 1, 2)
    ]
    sigs = [
        {hashlib.md5(s[i].numpy().tobytes()).hexdigest() for i in range(s.shape[0])}
        for s in rounds
    ]
    assert all(s.shape[0] == 64 for s in rounds)
    assert not (sigs[0] & sigs[1]), "round 1 must not repeat round 0 text"
    assert not (sigs[1] & sigs[2]), "round 2 must not repeat round 1 text"
    assert not (sigs[0] & sigs[2])
