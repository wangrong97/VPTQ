# Hessian Collection for VPTQ

Offline, hook-based Hessian (and inverse Hessian) collector, following
[QuIP#'s `hessian_offline_llama.py`](https://github.com/Cornell-RelaxML/quip-sharp/blob/main/quantize_llama/hessian_offline_llama.py).
This fills the previously empty `vptq/tools/hessian/` package.

## What it does

For every decoder layer, forward hooks capture the input activation `x` of
each linear operator and accumulate the second moment `H = E[x x^T]`
(with running mean and token count). Linears sharing the same input tensor
(`q/k/v_proj` -> `attn_in`, `gate/up_proj` -> `mlp_in`) share one Hessian;
`o_proj` / `down_proj` get their own. Unrecognized architectures fall back
to one Hessian per linear, which is always correct (just heavier).

The collected matrices are what the VPTQ quantizer consumes:

| Use in VPTQ | Which part of H |
|---|---|
| Weighted K-means centroid init | diagonal `h_ii` as channel weights |
| GPTQ-style error propagation | `H^{-1}` (Cholesky, damped) |
| Outlier channel detection | large diagonal values |

## Usage

```bash
python -m vptq.tools.hessian \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --output-dir ./hessians/llama-3.1-8b \
    --nsamples 128 --seqlen 4096
```

- Calibration data defaults to `togethercomputer/RedPajama-Data-1T-Sample`
  (the dataset used for the official VPTQ Hessian checkpoints). Use
  `--data-path local.jsonl` (one `{"text": ...}` per line) for custom data.
- For very large models, shard by decoder layers and merge the outputs in
  one directory: `--layer-range 0:20`, then `20:40`, ... Each run only
  holds the Hessians of its slice in memory.
- Accumulator memory per layer is roughly
  `(3 * hidden^2 + intermediate^2) * 4 bytes` (fp32). Use
  `--accum-device cpu` if GPU memory is tight, or `--accum-dtype fp64`
  for extra numerical safety.

## Output

```
<output-dir>/
├── meta.json          # model, dataset, hook plans, conventions
├── layer_0000.pt      # per-layer dict: group -> entry
├── layer_0001.pt
└── ...
```

Each entry:

| key | content |
|---|---|
| `hessian` | `E[x x^T]` (fp32), or covariance `H - mu mu^T` with `--center` |
| `inv_hessian` | damped Cholesky inverse (fp32), unless `--no-inv` |
| `mean` | `E[x]` (fp32) |
| `n_tokens` | number of accumulated activation rows |

Damping follows GPTQ: dead diagonal entries are set to 1, then
`--damp * mean(diag(H))` (default 0.01) is added before inversion,
computed in float64.
