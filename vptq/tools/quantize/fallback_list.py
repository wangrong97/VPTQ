"""Fallback list generation from gate scores.

预算（用户 2026-09-24 定义，以矩阵计）：
  总回退矩阵 = 43*256*3*12.5% = 4128
  ├─ 整专家回退：43*256*3*12.5%*0.5/3 = 688 个专家（w1+w2+w3 全回退）
  └─ down 层回退：43*256*3*12.5%*0.5  = 2064 个 w2（仅 down，w1/w3 保持量化）

排序依据：专家分数 = sum(专家被激活时的 gate 分数)，降序——分数高 =
每次被选中时路由权重大，量化损伤的输出代价高，优先回退。
hash 路由层（gate_scores.json 中为 null）不参与排序。

用法：
    python -m vptq.tools.quantize.fallback_list \
        --gate-scores $HESS/gate_scores.json \
        --output $QUANT/fallback_list.json
"""

import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gate-scores", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--budget-frac", type=float, default=0.125,
        help="总矩阵回退预算占比（默认 12.5%%）",
    )
    p.add_argument(
        "--full-share", type=float, default=0.5,
        help="预算中整专家份额（默认对半：0.5 整专家 + 0.5 down 层）",
    )
    args = p.parse_args()

    with open(args.gate_scores) as f:
        scores = json.load(f)

    # (layer, expert) 对：仅 score 路由层参与
    pairs = []
    for li_s, arr in scores.items():
        if arr is None:
            continue
        li = int(li_s)
        for e, v in enumerate(arr):
            pairs.append((v, li, e))
    n_pairs = len(pairs)
    n_layers = len(scores)
    n_experts = max(
        len(a) for a in scores.values() if a is not None
    )
    total_mats = n_layers * n_experts * 3
    total_budget = int(round(total_mats * args.budget_frac))

    n_full = total_budget * args.full_share // 3 // 1
    n_full = int(total_budget * args.full_share) // 3
    n_down = int(total_budget * (1 - args.full_share))

    pairs.sort(reverse=True)  # 分数降序
    chosen_full = [(li, e) for _, li, e in pairs[:n_full]]
    next_pairs = pairs[n_full:]
    # down-only：跳过已是 full 的专家（其 w2 已随整专家回退）
    full_set = set(chosen_full)
    chosen_down = [
        (li, e) for _, li, e in next_pairs if (li, e) not in full_set
    ][:n_down]

    out = {
        "meta": {
            "budget_frac": args.budget_frac,
            "total_matrices": total_mats,
            "full_experts": len(chosen_full),
            "down_only": len(chosen_down),
            "fallback_matrices": len(chosen_full) * 3 + len(chosen_down),
            "gate_scores": args.gate_scores,
            "ranking": "sum(gate weight at activation), desc",
        },
        "full": [[li, e] for li, e in chosen_full],
        "down_only": [[li, e] for li, e in chosen_down],
    }
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(
        f"fallback list: {args.output}\n"
        f"  full-expert fallback: {len(chosen_full)} experts "
        f"(x3 mats) + down-only: {len(chosen_down)} w2 mats "
        f"= {len(chosen_full)*3 + len(chosen_down)}/{total_budget} "
        f"budget matrices ({pairs and f'{n_pairs} score-routed experts ranked'})"
    )


if __name__ == "__main__":
    main()
