"""Per-expert gate score accumulation for fallback ranking.

专家分数 = sum(专家被激活时的 gate 分数)：score 路由层（>= n_hash_layers）
每个 (token, topk) 命中按 routing weight 累加；hash 层（查表路由，无分数）
不参与排序（记 null）。输出 gate_scores.json 供 fallback_list.py 生成回退名单。

与 collect_hessian 同源（同 data_file/seed/round → 样本逐位一致）。

用法：
    python -m vptq.tools.deepseek_v4.gate_scores \
        --ckpt /path/to/ckpt --data-dir ... --data-file ... \
        --output-dir $HESS --device npu
"""

import argparse
import json
import logging
import os

import torch
import transformers

from vptq.tools.deepseek_v4.collect_hessian import build_samples
from vptq.tools.deepseek_v4.loader import build_model, streamed_forward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vptq.deepseek_v4.gate_scores")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--data-file", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="npu")
    p.add_argument("--n-experts", type=int, default=256)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # tokenizer：与 collect_hessian 相同的 fallback 链
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.ckpt)
    except Exception as err:
        logger.warning("AutoTokenizer failed (%s); using tokenizer.json", err)
        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(args.ckpt, "tokenizer.json")
        )
    samples = build_samples(
        tokenizer, args.data_dir, args.nsamples, args.seqlen, args.seed,
        round_idx=0, data_file=args.data_file,
    )
    # data_file 分支返回 list（collect_hessian 主流程同样自行转 tensor）
    if not torch.is_tensor(samples):
        samples = torch.tensor(samples, dtype=torch.long)

    model = build_model(
        args.ckpt,
        max_batch_size=args.batch_size,
        max_seq_len=args.seqlen,
        device="cpu",
    )
    layers = model.layers
    n_layers = len(layers)
    n_hash = sum(
        1 for l in layers
        if getattr(l.ffn, "gate", None) is not None and l.ffn.gate.hash
    )
    logger.info("layers=%d, hash-routing layers=%d", n_layers, n_hash)

    sums = {
        li: torch.zeros(args.n_experts, dtype=torch.float64)
        for li in range(n_hash, n_layers)
    }

    def make_hook(li):
        def hook(_mod, _inp, output):
            weights, indices = output  # (b*s, topk) each
            w = weights.detach().float().cpu().view(-1)
            idx = indices.detach().view(-1).cpu().long()
            sums[li].index_add_(0, idx, w.double())
        return hook

    handles = []
    for li in range(n_hash, n_layers):
        gate = layers[li].ffn.gate
        handles.append(gate.register_forward_hook(make_hook(li)))

    streamed_forward(
        model, samples,
        device=torch.device(args.device),
        residency="stream",
        batch_size=args.batch_size,
    )
    for h in handles:
        h.remove()

    out = {}
    for li in range(n_layers):
        out[str(li)] = (
            None if li < n_hash
            else [round(float(v), 6) for v in sums[li].tolist()]
        )
    path = os.path.join(args.output_dir, "gate_scores.json")
    with open(path, "w") as f:
        json.dump(out, f)
    top_layer = max(
        (li for li in range(n_layers) if out[str(li)] is not None),
        key=lambda li: max(out[str(li)]),
    )
    logger.info(
        "gate scores saved: %s (score layers %d..%d); "
        "example top layer %d max expert score %.2f",
        path, n_hash, n_layers - 1, top_layer,
        max(out[str(top_layer)]),
    )


if __name__ == "__main__":
    main()
