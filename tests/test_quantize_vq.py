# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import torch

from vptq.tools.quantize import (
    QuantConfig,
    cholesky_upper_of_inv,
    vptq_quantize,
    weighted_kmeans,
)


def _proxy_err(What, W, H):
    dW = What - W
    return ((dW @ H * dW).sum() / (W @ H * W).sum()).item()


def test_weighted_kmeans_beats_plain_on_skewed_data():
    torch.manual_seed(0)
    # two clusters: big one (low weight) + small one (high weight)
    a = torch.randn(900, 8) * 0.5
    b = torch.randn(100, 8) * 0.5 + 5.0
    vecs = torch.cat([a, b])
    w = torch.cat([torch.ones(900), torch.full((100,), 50.0)])

    def weighted_err(c):
        d = torch.cdist(vecs, c).min(-1).values
        return (w * d.square()).sum().item()

    c_w = weighted_kmeans(vecs, w, k=4, iters=8, seed=0)
    c_p = weighted_kmeans(vecs, torch.ones_like(w), k=4, iters=8, seed=0)
    assert weighted_err(c_w) < weighted_err(c_p)


def test_cholesky_upper_reconstructs_inverse():
    torch.manual_seed(0)
    a = torch.randn(32, 32)
    H = a @ a.t() / 32
    U = cholesky_upper_of_inv(H, damp=0.01)
    from vptq.tools.hessian.collector import damped_inverse

    inv = damped_inverse(H, 0.01)
    assert torch.allclose((U.double().t() @ U.double()).float(), inv,
                          atol=1e-3)


def _naive_vq(W, config, centroids):
    """Independent nearest-centroid assignment, no error propagation."""
    v = config.vector_len
    out_pad = (-W.shape[0]) % v + W.shape[0]
    Wp = torch.nn.functional.pad(W, (0, 0, 0, out_pad - W.shape[0]))
    Q = torch.zeros_like(Wp)
    for col in range(W.shape[1]):
        vecs = Wp[:, col].view(-1, v)
        idx = torch.cdist(vecs, centroids).argmin(-1)
        Q[:, col] = centroids[idx].view(-1)
    return Q[: W.shape[0]]


def test_vptq_beats_naive_vq_on_proxy_error():
    torch.manual_seed(0)
    in_f, out_f, v = 64, 32, 8
    # correlated activations -> nontrivial Hessian geometry
    mixing = torch.randn(in_f, in_f) * 0.5 + torch.eye(in_f)
    X = torch.randn(4096, in_f) @ mixing
    H = X.t() @ X / X.shape[0]
    W = torch.randn(out_f, in_f) * 0.05

    config = QuantConfig(
        vector_len=v,
        num_centroids=64,
        num_res_centroids=16,
        group_num=2,
        kmeans_iters=6,
    )
    result = vptq_quantize(W, H, config, device=torch.device("cpu"))

    # reconstruct from QuantizedWeight
    out_pad = (-out_f) % v + out_f
    What = torch.zeros(out_pad, in_f)
    gs = in_f // config.group_num
    for g in range(config.group_num):
        cb = result.centroids[g].float()
        rcb = result.res_centroids[g].float()
        idx = result.indices[g].long()
        ridx = result.res_indices[g].long()
        cols = (cb[idx] + rcb[ridx]).permute(1, 2, 0).reshape(-1, gs)
        What[:, g * gs : (g + 1) * gs] = cols
    What = What[:out_f]

    # naive baseline with the same main codebooks
    naive = torch.zeros_like(What)
    for g in range(config.group_num):
        cb = result.centroids[g].float()
        for col in range(gs):
            vecs = W[:, g * gs + col].view(-1, v)
            idx = torch.cdist(vecs, cb).argmin(-1)
            naive[:, g * gs + col] = cb[idx].view(-1)

    err_vptq = _proxy_err(What, W, H)
    err_naive = _proxy_err(naive, W, H)
    print(f"proxy err: vptq {err_vptq:.5f} vs naive {err_naive:.5f}")
    assert err_vptq < err_naive
    assert result.proxy_error < 1.0
    assert result.centroids.shape == (2, 64, v)
    assert result.indices.shape == (2, 32, out_pad // v)
    assert result.res_centroids.shape == (2, 16, v)


def test_residual_reduces_error():
    torch.manual_seed(0)
    in_f, out_f, v = 64, 32, 8
    X = torch.randn(4096, in_f)
    H = X.t() @ X / X.shape[0]
    W = torch.randn(out_f, in_f) * 0.05
    base = QuantConfig(vector_len=v, num_centroids=64,
                       num_res_centroids=-1, kmeans_iters=6)
    with_res = QuantConfig(vector_len=v, num_centroids=64,
                           num_res_centroids=16, kmeans_iters=6)
    r0 = vptq_quantize(W, H, base, device=torch.device("cpu"))
    r1 = vptq_quantize(W, H, with_res, device=torch.device("cpu"))
    print(f"mse: no-res {r0.mse:.5f} vs res {r1.mse:.5f}")
    assert r1.mse < r0.mse
