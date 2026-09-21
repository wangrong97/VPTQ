# 会话上下文交接（2026-09-18 16:05，暂停采集时保存）

> 供后续会话/协作方接手。配套详情见 `documents/vptq_algorithm_and_code.md` §8A.7（补采论证）。

## 一、任务目标

DeepSeek-V4-Flash Hessian 补采（floor=1024 token/专家）→ 重新量化 → 部署评测，验证 §8A.7 的收益预测（冷门专家 Hessian 供给不足是量化 MSE 偏高的主因之一）。

## 二、当前状态（暂停时刻）

- **补采已暂停**：`STOP` 标记在 `$OUT/STOP`（`$OUT=/mnt/share/rr08002/weights/hessians/DeepSeek-V4-Flash-BF16-rpmix`）
- **进度**：回滚后重跑的 round 1（data-round 1，全新文本）推进到 L4；**L3 已用真数据补完并上传**（合法增量，保留），L4 被杀未保存
- **产物状态**：主目录 = 原始采集 + 仅 L3 的 round-1 增量；基线快照 `DeepSeek-V4-Flash-BF16-rpmix-pre-topup2`（硬链接，零空间）；污染版存档 `DeepSeek-V4-Flash-BF16-rpmix-polluted-0918`（勿用）
- **待补**：约 668 组（40 个 score 层，<1024 token 的专家 mlp_in/w2 组）
- **恢复命令**：`rm $OUT/STOP && setsid nohup bash /mnt/share/rr08002/work/VPTQ/topup_rpmix.sh &`——看门狗自动从 `topup.round` 计数续跑（当前=1，下轮 data-round 2）

## 三、事故史（三条根因教训，均已修复并有回归测试）

1. **act_ckpt 跨轮空转**：断点只匹配 (nsamples,seqlen,batch) 不匹配样本 → round N 结束后 next_layer=43 存活，下轮"续"过末层空转退出（假 DONE）。修复：看门狗每轮 rm act_ckpt（attempt 级重试保留）。判别：轮次扫描数字不递减 = 空转。
2. **seed 重复**：看门狗重启后 round 计数重置 → 同 seed 跑 3 次。修复：`topup.seed` 持久化（现已由 topup.round 取代）。
3. **单文件 slice 换 seed 无效（根本性）**：6 个 slice 中 5 个单文件，seed shuffle 文件对文档顺序无效——实测 seed 1/2/3 样本 100% 相同、seed 1 与原始重叠 26%。重复样本使 H=sum/n 不变、仅 n_tokens 虚高（假达标锁死）。修复：`build_samples(round_idx=k)` 取文档流第 k 段全新文本（round 0=原始），`--topup-round` 参数 + 回归测试 `test_build_samples_rounds_are_disjoint`。

## 四、补采运行要点（topup_rpmix.sh v3.2）

- 4 轮 × 128 条全新数据（data-round 1..4），floor=1024，每轮 ~1.7h
- 冷却保护：被 kill 后 300s + 确认 NPU 空闲（无他人进程 + HBM 回落）才重启，不抢占
- 优化：`--act-ckpt` 本地盘、`--save-staging /root/topup_staging`（层文件本地写+后台上传）、`--prefetch-next`（restore 预取）
- 停止：`touch $OUT/STOP && pkill -f collect_hessian`
- 环境坑：killer 周期 10-25min；NPU 驱动栈曾整挂（dcmi -8005，升级 25.1.rc2 后恢复）；维护窗口会清进程

## 五、补采完成后的管线（用户已确认的完整路径）

1. `compute_inverse` 补 inv：`for r in 0:11 11:22 22:33 33:43; do nohup python3 -m vptq.tools.hessian.compute_inverse --input-dir $OUT --layer-range $r --device cpu >> $OUT/inv_$r.log 2>&1 & done`（幂等，只补无 inv 的组）
2. **VPTQ 32×16 无残差量化**（8A.2 最优组 2.50bit）：`quantize_dsv4_experts_tile16.sh` 参数 `--vector-len 2 --num-centroids 16 --num-res-centroids -1 --row-tile 16 --group-num 128 --centroid-fp8`；**HESSIAN_DIR=$OUT，OUTPUT_DIR 必须换新**：`weights/quant/dsv4-tile32x16-nores-topup1024/`
3. BF16 重建：`python -m vptq.tools.quantize.rebuild_bf16 --quant-dir <上一步> --ckpt /root/dsv4-weights --output-dir weights/DeepSeek-V4-Flash-vptq-topup1024-tile16-bf16/ --group-num 128 --row-tile 16 --device npu`
4. msmodelslim（attnW8A8ceil + moeW4A8）：配置 `/mnt/share/rr08002/msmodelslim/deepseek-v4-flash-attenw8a8moew4a8.yaml`，脚本 `/mnt/share/rr08002/msmodelslim/modelslim.sh`（改 MODEL_PATH=上一步 BF16、SAVE_PATH=`weights/DeepSeek-V4-Flash-vptq-topup1024-tile16-attnw8a8-moew4a8/`；rot2 线已叫停，原路径不再用但不覆盖）
5. **验证部署权重完整后删除中间 BF16（543GB）**（用户要求）；量化产物目录是否删届时确认（quant_report.json 是 8A.7 复验数据源）
6. vLLM：`cd /mnt/share/rr08002/work/vllm-ascend && ./vllm_start_dp.sh`（serve 路径在脚本内，改成新部署权重；8 卡 DP+MTP，port 9001）
7. 评测：`cd /mnt/share/rr08002/benchmark && ./aisbench.sh`（GPQA/aime 各 3 轮），结果整理到 `/mnt/share/rr08002/work/vllm-ascend/ais_bench_logs/vptq-topup1024-tile16/`
8. 对照基准（8A.2）：tile16+旧Hessian = GPQA 65.53 / aime 71.11；无 VPTQ 基线 = 73.99 / 71.67

## 六、验证方法（补采收益复核）

新 `quant_report.json` 重跑 §8A.7 分桶对照：原 <1024 人群 MSE（~0.19）应显著回落；注意本次 n_tokens 口径干净（回滚后重建）。**已知残余风险（用户知情）**：floor=1024 使 0.5–2k 桶（实测最差区 0.218）不再补，若收益不及预期可提 floor 到 2000–4000 续采（增量机制支持）。

## 七、关键文件

```
topup_rpmix.sh                     补采看门狗 v3.2（round 持久化版）
vptq/tools/deepseek_v4/collect_hessian.py   --topup-round/--save-staging/--prefetch-next
vptq/tools/hessian/collector.py    from_saved/token_caps/restore_state/save(prev_payload)
tests/test_hessian_collector.py    补采 10 项；test_deepseek_v4_port.py 含 round 不重叠回归
weights/hessians/DeepSeek-V4-Flash-BF16-rpmix*   主目录/快照2/污染存档 三份
/root/topup_staging, /root/act_ckpt_topup.pt     本地缓存（轮间清理）
```
