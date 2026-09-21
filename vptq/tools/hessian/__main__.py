# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CLI entry: offline Hessian collection for VPTQ quantization.

Example:
    python -m vptq.tools.hessian \
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \
        --output-dir ./hessians/llama-3.1-8b \
        --nsamples 128 --seqlen 4096

For very large models, shard the collection by decoder layer ranges:
    python -m vptq.tools.hessian --model <70B> --output-dir ./h \
        --layer-range 0:20
    python -m vptq.tools.hessian --model <70B> --output-dir ./h \
        --layer-range 20:40
    ...
"""

import argparse
import json
import logging
import os
import random

import torch
import transformers

from vptq.tools.hessian.collector import HessianCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vptq.hessian")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect Hessian / inverse Hessian matrices "
        "(QuIP#-style offline collection) for VPTQ quantization."
    )
    parser.add_argument(
        "--model", required=True, help="HF model name or local path."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Where to write layer_XXXX.pt."
    )
    # calibration data
    parser.add_argument(
        "--data-path",
        default=None,
        help="Local jsonl file with one text per line. "
        "Overrides --dataset.",
    )
    parser.add_argument(
        "--dataset",
        default="togethercomputer/RedPajama-Data-1T-Sample",
        help="HF dataset used for calibration (VPTQ default).",
    )
    parser.add_argument(
        "--dataset-config", default=None, help="HF dataset config name."
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of sequences."
    )
    parser.add_argument(
        "--seqlen", type=int, default=4096, help="Sequence length."
    )
    parser.add_argument("--seed", type=int, default=0)
    # collection control
    parser.add_argument(
        "--layer-range",
        default=None,
        help="Decoder layer slice `start:end` (for sharding big models).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Sequences per forward."
    )
    parser.add_argument(
        "--accum-device",
        default=None,
        choices=["cuda", "npu", "cpu"],
        help="Device holding Hessian accumulators "
        "(default: best available accelerator).",
    )
    parser.add_argument(
        "--accum-dtype",
        default="fp32",
        choices=["fp32", "fp64"],
        help="Accumulator dtype.",
    )
    parser.add_argument(
        "--model-dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Dtype to load the model with.",
    )
    parser.add_argument("--device-map", default="auto")
    # inverse Hessian
    parser.add_argument(
        "--no-inv",
        action="store_true",
        help="Skip computing inverse Hessians.",
    )
    parser.add_argument(
        "--damp",
        type=float,
        default=0.01,
        help="Damping ratio for the inverse (fraction of mean(diag(H))).",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Use covariance form H - mu mu^T (QuIP# convention) "
        "instead of the raw second moment.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_texts(args):
    if args.data_path is not None:
        logger.info("loading calibration texts from %s", args.data_path)
        texts = []
        with open(args.data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(json.loads(line)[args.text_field])
        return texts
    import datasets

    logger.info("loading calibration dataset %s", args.dataset)
    ds = datasets.load_dataset(
        args.dataset, args.dataset_config, split=args.split
    )
    return [row[args.text_field] for row in ds]


def build_samples(tokenizer, texts, nsamples, seqlen, seed):
    rng = random.Random(seed)
    rng.shuffle(texts)
    need = nsamples * seqlen
    ids = []
    for text in texts:
        ids.extend(tokenizer(text).input_ids)
        if len(ids) >= need:
            break
    if len(ids) < need:
        logger.warning(
            "only %d tokens collected (< %d); the number of samples "
            "will be reduced",
            len(ids),
            need,
        )
    n_full = len(ids) // seqlen
    samples = torch.tensor(ids[: n_full * seqlen]).view(n_full, seqlen)
    logger.info("built %d samples of length %d", n_full, seqlen)
    return samples


def main():
    args = parse_args()
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    layer_range = None
    if args.layer_range is not None:
        start, end = args.layer_range.split(":")
        layer_range = (int(start), int(end))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.model_dtype],
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()

    texts = load_texts(args)
    samples = build_samples(
        tokenizer, texts, args.nsamples, args.seqlen, args.seed
    )

    accum_device = (
        torch.device(args.accum_device) if args.accum_device else None
    )
    collector = HessianCollector(
        model,
        layer_range=layer_range,
        accum_dtype=dtype_map[args.accum_dtype],
        accum_device=accum_device,
    )
    try:
        collector.collect(samples, batch_size=args.batch_size)
    finally:
        collector.remove_hooks()

    collector.save(
        args.output_dir,
        model_name=args.model,
        save_inv=not args.no_inv,
        damp=args.damp,
        center=args.center,
        extra_meta={
            "dataset": args.data_path or args.dataset,
            "nsamples": int(samples.shape[0]),
            "seqlen": args.seqlen,
            "seed": args.seed,
        },
    )
    logger.info("done. Hessians written to %s", args.output_dir)


if __name__ == "__main__":
    main()
