# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from vptq.tools.hessian import (
    HessianAccumulator,
    HessianCollector,
    build_hook_plan,
    damped_inverse,
    find_decoder_layers,
)


def _tiny_model():
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config)


def test_find_decoder_layers():
    model = _tiny_model()
    layers, path = find_decoder_layers(model)
    assert path == "model.layers"
    assert len(layers) == 2


def test_hook_plan_shares_inputs():
    model = _tiny_model()
    plan = build_hook_plan(model.model.layers[0])
    # q/k/v collapse into attn_in represented by q_proj;
    # gate/up collapse into mlp_in represented by gate_proj.
    assert plan["self_attn.q_proj"] == "attn_in"
    assert plan["mlp.gate_proj"] == "mlp_in"
    assert plan["self_attn.o_proj"] == "self_attn.o_proj"
    assert plan["mlp.down_proj"] == "mlp.down_proj"
    assert "self_attn.k_proj" not in plan
    assert "self_attn.v_proj" not in plan
    assert "mlp.up_proj" not in plan
    assert len(plan) == 4


def test_hook_plan_scopes_groups_by_parent():
    # two "experts" each with w1/w3 must get separate Hessians
    class Expert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.Linear(8, 8)
            self.w3 = torch.nn.Linear(8, 8)
            self.w2 = torch.nn.Linear(8, 8)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.ModuleList([Expert(), Expert()])

    plan = build_hook_plan(Layer())
    assert plan["experts.0.w1"] == "experts.0.mlp_in"
    assert plan["experts.1.w1"] == "experts.1.mlp_in"
    assert plan["experts.0.w2"] == "experts.0.w2"
    assert plan["experts.1.w2"] == "experts.1.w2"
    assert "experts.0.w3" not in plan
    assert "experts.1.w3" not in plan
    assert len(plan) == 4


def test_hessian_matches_manual_accumulation():
    torch.manual_seed(0)
    model = _tiny_model().eval()
    samples = torch.randint(0, 128, (4, 32))

    collector = HessianCollector(model, accum_device=torch.device("cpu"))

    # independent reference: capture the same inputs with plain hooks
    captured = {}
    layer0 = model.model.layers[0]
    ref_names = ["self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj"]
    handles = []

    def make_ref_hook(name):
        def hook(module, inputs, output):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            captured.setdefault(name, []).append(x.float())

        return hook

    submodules = dict(layer0.named_modules())
    for name in ref_names:
        handles.append(
            submodules[name].register_forward_hook(make_ref_hook(name))
        )

    collector.collect(samples)
    collector.remove_hooks()
    for handle in handles:
        handle.remove()

    for name, group_key in [
        ("self_attn.q_proj", "attn_in"),
        ("self_attn.o_proj", "self_attn.o_proj"),
        ("mlp.down_proj", "mlp.down_proj"),
    ]:
        x = torch.cat(captured[name], dim=0)
        ref_h = x.t() @ x / x.shape[0]
        ref_mu = x.mean(dim=0)
        acc = collector.accumulators[0][group_key]
        hessian, mean, n_tokens = acc.finalize()
        assert n_tokens == x.shape[0]
        assert hessian.shape == (x.shape[-1], x.shape[-1])
        assert torch.allclose(hessian, ref_h, atol=1e-4)
        assert torch.allclose(mean, ref_mu, atol=1e-5)


def test_k_proj_shares_attn_in_hessian():
    # k_proj sees the same input tensor as q_proj; its Hessian must be
    # identical to the attn_in one (no separate accumulation).
    torch.manual_seed(0)
    model = _tiny_model().eval()
    samples = torch.randint(0, 128, (2, 32))
    collector = HessianCollector(model, accum_device=torch.device("cpu"))
    assert "k_proj" not in collector.accumulators[0]
    collector.collect(samples)
    collector.remove_hooks()
    assert set(collector.accumulators[0].keys()) == {
        "attn_in",
        "self_attn.o_proj",
        "mlp_in",
        "mlp.down_proj",
    }


def test_damped_inverse():
    torch.manual_seed(0)
    a = torch.randn(16, 16)
    hessian = a @ a.t() / 16
    inv = damped_inverse(hessian, damp=0.01)
    assert inv.shape == hessian.shape
    assert inv.dtype == torch.float32
    assert torch.allclose(inv, inv.t(), atol=1e-5)
    # compare against a direct float64 inverse of the damped matrix
    h64 = hessian.to(torch.float64)
    damp_value = 0.01 * torch.mean(torch.diagonal(h64))
    expected = torch.inverse(
        h64 + damp_value * torch.eye(16, dtype=torch.float64)
    )
    expected = (expected + expected.t()) / 2
    assert torch.allclose(inv, expected.float(), atol=1e-4)


def test_save_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = _tiny_model().eval()
    samples = torch.randint(0, 128, (2, 32))
    collector = HessianCollector(model, accum_device=torch.device("cpu"))
    collector.collect(samples)
    collector.remove_hooks()
    collector.save(str(tmp_path), model_name="tiny", damp=0.01)

    import json
    import os

    meta = json.load(open(os.path.join(tmp_path, "meta.json")))
    assert meta["layer_indices"] == [0, 1]
    payload = torch.load(
        os.path.join(tmp_path, "layer_0000.pt"), weights_only=False
    )
    entry = payload["attn_in"]
    assert entry["hessian"].shape == (32, 32)
    assert entry["inv_hessian"].shape == (32, 32)
    assert entry["mean"].shape == (32,)
    assert entry["n_tokens"] == 2 * 32


def test_accumulator_restore_equivalence():
    # from_saved must be the exact inverse of finalize():
    # sum_xxt = hessian * n_tokens round-trips within fp32 rounding.
    torch.manual_seed(0)
    acc = HessianAccumulator(5)
    acc.update(torch.randn(7, 5))
    hessian, mean, n_tokens = acc.finalize()

    restored = HessianAccumulator.from_saved(hessian, mean, n_tokens)
    assert not restored.updated
    h2, m2, n2 = restored.finalize()
    assert n2 == n_tokens
    assert torch.allclose(h2, hessian, rtol=0, atol=1e-6)
    assert torch.allclose(m2, mean, rtol=0, atol=1e-7)

    restored.update(torch.randn(3, 5))
    assert restored.updated


def test_topup_freezes_satisfied_groups(tmp_path):
    import os

    torch.manual_seed(0)
    model = _tiny_model().eval()
    c1 = HessianCollector(model, accum_device=torch.device("cpu"))
    c1.collect(torch.randint(0, 128, (2, 32)))
    c1.remove_hooks()
    c1.save(str(tmp_path), model_name="tiny")
    payload = torch.load(
        os.path.join(str(tmp_path), "layer_0000.pt"),
        mmap=True,
        map_location="cpu",
    )

    def cap_fn(key):
        if key == "mlp_in":
            return 10 ** 9  # never satisfied: must keep accumulating
        if key == "attn_in":
            return payload["attn_in"]["n_tokens"]  # already satisfied
        return 0  # frozen (o_proj / down_proj)

    c2 = HessianCollector(
        model,
        layer_range=(0, 1),
        accum_device=torch.device("cpu"),
        token_caps=cap_fn,
    )
    restored = c2.restore_state(0, payload)
    assert set(restored) == {"mlp_in"}  # only the under-cap group is loaded
    c2.collect(torch.randint(0, 128, (2, 32)))
    c2.remove_hooks()
    assert c2.accumulators[0]["mlp_in"].n_tokens == 64 + 2 * 32
    assert "attn_in" not in c2.accumulators[0]  # frozen: never created
    assert "self_attn.o_proj" not in c2.accumulators[0]
    assert c2.pending_under_caps() == ["mlp_in"]


def test_topup_end_to_end(tmp_path):
    import os

    torch.manual_seed(0)
    model = _tiny_model().eval()
    samples1 = torch.randint(0, 128, (2, 32))
    samples2 = torch.randint(0, 128, (2, 32))

    # baseline collection with inverses
    c1 = HessianCollector(model, accum_device=torch.device("cpu"))
    c1.collect(samples1)
    c1.remove_hooks()
    c1.save(str(tmp_path), model_name="tiny", save_inv=True, damp=0.01)
    base = torch.load(
        os.path.join(str(tmp_path), "layer_0000.pt"), weights_only=False
    )

    # reference: what mlp_in would look like if both sets were seen
    c_ref = HessianCollector(model, accum_device=torch.device("cpu"))
    c_ref.collect(samples2)
    c_ref.remove_hooks()
    h2, _, n2 = c_ref.accumulators[0]["mlp_in"].finalize()

    # top-up pass: mlp_in below an unreachable floor, everything frozen
    payload = torch.load(
        os.path.join(str(tmp_path), "layer_0000.pt"),
        mmap=True,
        map_location="cpu",
    )

    def cap_fn(key):
        return 10 ** 9 if key == "mlp_in" else 0

    c2 = HessianCollector(
        model,
        layer_range=(0, 1),
        accum_device=torch.device("cpu"),
        token_caps=cap_fn,
    )
    c2.restore_state(0, payload)
    c2.collect(samples2)
    c2.remove_hooks()
    c2.save(
        str(tmp_path), model_name="tiny", save_inv=False, prev_payload=payload
    )
    out = torch.load(
        os.path.join(str(tmp_path), "layer_0000.pt"), weights_only=False
    )

    # frozen groups carried verbatim, including their inverses
    for key in ["attn_in", "self_attn.o_proj", "mlp.down_proj"]:
        assert torch.equal(out[key]["hessian"], base[key]["hessian"])
        assert torch.equal(out[key]["inv_hessian"], base[key]["inv_hessian"])
        assert out[key]["n_tokens"] == base[key]["n_tokens"]

    # topped-up group: merged statistics over both sample sets, and the
    # stale inverse is dropped for compute_inverse to refill
    n1 = base["mlp_in"]["n_tokens"]
    expected = (base["mlp_in"]["hessian"] * n1 + h2 * n2) / (n1 + n2)
    assert out["mlp_in"]["n_tokens"] == n1 + n2
    assert torch.allclose(out["mlp_in"]["hessian"], expected, atol=1e-5)
    assert "inv_hessian" not in out["mlp_in"]


def test_staging_upload_and_flush(tmp_path):
    # staging mode: layers are written locally and uploaded atomically;
    # residuals from a killed run must flush before the resume scan
    from vptq.tools.deepseek_v4.collect_hessian import (
        _flush_staging,
        _upload_layer,
    )

    out = tmp_path / "out"
    staging = tmp_path / "staging"
    out.mkdir()
    staging.mkdir()
    (out / "layer_0003.pt").write_bytes(b"stale")
    (staging / "layer_0003.pt").write_bytes(b"fresh")
    (staging / "meta.json").write_bytes(b"{}")

    _upload_layer(str(staging / "meta.json"), str(out))
    assert (out / "meta.json").read_bytes() == b"{}"
    assert not (staging / "meta.json").exists()

    _flush_staging(str(staging), str(out))
    assert (out / "layer_0003.pt").read_bytes() == b"fresh"
    assert not (staging / "layer_0003.pt").exists()
    assert not (out / "layer_0003.pt.tmp").exists()  # atomic: no torn tmp
