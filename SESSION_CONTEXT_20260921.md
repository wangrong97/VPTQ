# 会话上下文交接（2026-09-21 17:40，20:50 / 09-22 10:45 更新）

> 接续 `SESSION_CONTEXT_20260918.md`。当天已完成：路径迁移修复、pip 包重装、HCCL 故障定位（未修复，用户决定暂缓）。

## 〇b、09-22 10:45 全部任务已停止（环境他人使用）

**用户指示"环境有其他人在用，先停止任务"，已全部停止并释放资源**：
- 已停：评测看门狗 v2 / 管线看门狗 / ais_bench / vllm / 每小时监控 cron —— 全部进程清零，8 卡 HBM 释放（停止时他人任务已在 NPU 上，util 14-35%）
- STOP 标记保留：/tmp/eval_watchdog.STOP、/tmp/pipeline_calibR9.STOP（防误启动，恢复前先删）

**停止时刻的进度快照**：
- dualnorm 评测：有效完成 **GPQA r1=68.69**（1/6），aime r1 曾跑大半被 killer 杀（无 summary）
- 有效轮基线（新口径，只数真实成绩）：gpqa=75 / aime=36，目标各 +3
- calibR9 管线：S0 未过（状态目录 /tmp/pipeline_calibR9/ 已清空，从头开始）
- 环境 bug 已修：`vptq/utils/pack.py` sentence_transformers 改延迟 import；评测看门狗 v2（vllm 看护+预热+有效计数）；管线 9 处计数判据修复

**恢复方法（环境空闲后）**：
1. 删两个 STOP 文件
2. 起评测看门狗：`cd /mnt/share/w00608002/benchmark && EVAL_SKIP_FIRST_COOLDOWN=1 setsid nohup bash watchdog_eval_dualnorm.sh &`（v2 会自动拉起 vllm→预热→续跑剩余 5 轮）
3. 起管线：`cd /mnt/share/w00608002/work/VPTQ && setsid nohup bash pipeline_calibR9.sh &`（S0 独立计数，自动衔接）
4. 恢复前确认他人任务已结束（npu-smi util 全 0 + 无特征进程）

**09-22 上午事件记录**：killer 于 10:03 杀 vllm EngineCore（10:05 评测看门狗不知情启动 GPQA r2 → 整轮空转失败，summary 占位符）→ 暴露看门狗不看护 vllm + 空轮假计数两个缺陷 → 已修（v2 + 计数判据）。教训：**评测编排必须把推理服务纳入看护范围，失败产物要能被完成判据区分**。

## 〇、晚间重大进展（20:50 补记）：环境恢复 + 双线自动化

**HCCL/设备栈已恢复**（无需人工干预，09-18 的 dcmi -8005 与 EI0014 均消失）：
- npu-smi 恢复正常（8 卡 Ascend950DT Health OK）；`/tmp/hccl_sanity_check.py` 已重建（原文件丢失；torchrun 8 卡 allreduce/allgather/broadcast 全 PASS）
- msmodelslim 26.1.0 与 vllm-ascend 0.23.1.dev15 重装完成。注意：**npu-smi 故障期间 setup.py 取芯片型号会失败**，绕过方式 `SOC_VERSION=ascend950dt_9582 pip install -e . --no-build-isolation --no-deps`（值来自 Dockerfile.a5 + 上次构建缓存 ASCEND_COMPUTE_UNIT=ascend950）

**双线自动化运行中（killer 对抗设计，均为 setsid nohup）**：
1. **dualnorm 评测**（20:10 启动）：aisbench.sh 6 轮 + `benchmark/watchdog_eval_dualnorm.sh` 看门狗（基线 g=90/a=70 只认新增）。GPQA r1=**68.69**（20:31 完成），aime r1 进行中。vllm serve dualnorm 部署权重（pid 7755，日志 /tmp/vllm_hccl_test.log）
2. **calibR9 全管线** `pipeline_calibR9.sh`（20:48 启动，pid 40884，日志 /tmp/pipeline_calibR9.log，STOP=/tmp/pipeline_calibR9.STOP）：
   - S0 等 dualnorm 评测 6 轮（独立计数判据，含评测看门狗自愈）→ 冷却 → 停 vllm → 等 HBM 释放
   - S1 calib_collect.sh 续跑（calibR9 Hessian 基线采集，已有 L3 进度）
   - S2 compute_inverse 4 分片（判据：4 个 inv_*.log 均 "all done"）
   - S3 生成 quantize_dsv4_experts_tile16_calibR9.sh（tile32×16 无旋转基线，**用户确认非 wrap/dualnorm**），OUTPUT_DIR=weights/quant/dsv4-tile32x16-nores-calibR9
   - S4 rebuild_bf16 → weights/DeepSeek-V4-Flash-vptq-calibR9-tile16-bf16
   - S5 msmodelslim（生成 modelslim_calibR9.sh）→ ...-calibR9-tile16-attnw8a8-moew4a8
   - S6 生成 vllm_start_calibR9.sh 起服务 + 真实请求预热×2（§8A.2 教训）
   - S7 aisbench 6 轮（锁基线）→ S8 REPORT → ais_bench_logs/vptq-calibR9-tile16-attnw8a8-moew4a8-20260922/
   - **保守冷却（用户指示）**：被杀后静默 20min → 等环境任务（特征进程/NPU util>0）结束 → 连续空闲 20min 才启动
- 教训：S0 原判据 grep 看门狗日志"全部 6 轮完成"被旧 bug 实例的错误日志行污染 → 误判后差点 20min 后杀 vllm；已改独立计数。**判据勿依赖可被历史污染的日志文本**
- pgrep 教训：`bash -c` 包装命令行里引号内的 pattern 会自匹配（用字符类 `[e]` 规避）；pgrep BRE 的 `\|` 无效须用 ERE `|`

**topup4k 线状态不变**（L36/43 中断，续跑 `bash quantize_dsv4_experts_tile16.sh`）——注意它与新管线互抢 NPU，calibR9 线跑完前勿启动。

## 一、环境重大变更：共享路径迁移

- `/mnt/share/rr08002` → `/mnt/share/w00608002`（发生在 09-21 10:11~11:20 之间）
- ✅ 已修：VPTQ 全部 `*.sh`（9 个）、`vptq/tools` 的 `collect_hessian.py`/`eval_fp8_codebook.py`、`msmodelslim/modelslim.sh`、vllm-ascend 的 `vllm_start_dp.sh`/`attn_weight_cmp.py`、ais_bench 的 `vllm_api_*.py`
- ✅ pip 三个 editable 包已重装并验证（msmodelslim 26.1.0 / vllm 0.23.0+empty / vllm_ascend，site-packages 零残留）。vllm-ascend 需先 `rm -rf csrc/build`（CMakeCache 缓存旧路径）；重装命令见各目录（vllm 需 `VLLM_TARGET_DEVICE=empty TMPDIR=/mnt/share`）
- 历史文档/日志/.pt 内部路径保留旧值（运行时产物不受影响）

## 二、实验线状态

| 线 | 状态 | 结果/进度 |
|---|---|---|
| topup1024 全管线 | ✅ 完成含评测 | GPQA 7 轮均值 ≈65.87（67.17/63.64/69.70/62.63/66.16/66.16/65.66），aime 3 轮 74.44 均值（73.33/70/80）。vs 旧Hessian 65.53/71.11：GPQA 持平，aime +3.3 → 开 topup4k 线 |
| topup4k | ⏸ 量化中断 | floor=4000 补采 round=9 于 09-20 17:26 主动 STOP（余 430 组<4000，用户知情）；inv4k 全部完成；8 shard 量化到 **L36/43**（09-21 10:11 被 killer 杀）。脚本幂等，`bash quantize_dsv4_experts_tile16.sh` 直接续跑 |
| dualnorm | ⛔ 评测阻塞 | 量化+BF16+msmodelslim 部署权重就绪（09-20 19:17），aisbench 被 code -9 杀；vllm 再起不来（见三） |
| calibr9 全新采集 | ⏸ 中断 | `calib_collect.sh`（R9tau 语料全新目录 calibR9），09-21 14:33 启动 14:39 被杀，采完 layer 3。续跑命令同脚本 |

## 三、HCCL 多卡通信故障（已定位，未修复——用户 09-21 决定暂缓）

**现象**：多卡（≥2）`hcclCommInitRootInfoConfig` 报 `Config_Error_Ranktable(EI0014): addr ::7:300:10:0:df32:2801 invalid`；换 `HCCL_NPU_SOCKET_PORT_RANGE` 后变 EI0015（RootInfoDetect failed）。**单 rank 通信域完全正常**。vllm 8 卡 DP 从 09-21 11:20 起同样挂（`/tmp/vllm_manual2.log:394`），是 dualnorm 评测阻塞的直接原因。

**根因**（证据链完整）：
1. EI0014 的 addr 值 = `/proc/net/if_inet6` 中 ipourma\*（NPU 网络接口）上 **df32 段 IPv6** —— torch_npu 把陈旧设备网络地址填进 rootInfo，host 侧校验只认 IPv4
2. 该类地址是 hccp 设备网络正常工作地址、**每班任务一个段**；外部任务 11:42 用 df36 段正常 bind 16666（plog device-1447 日志），接口上残留 df32/df38/df3a 多个历史段（被强杀任务残留）
3. 时间线：hccp 本容器最后正常 09-21 10:01（量化）→ 10:11 量化被 killer 强杀 → 之后多卡初始化全挂。昨天（09-20）同环境 vllm 正常
4. 硬件层全净：8 卡 fault-event 空、health OK、UB 全互联、无内核错误、/dev/shm 与 IPC 无残留
5. torch_npu/libhccl 无 IFNAME 类变量可绕过（已 strings 验证）

**修复选项**（卡须空闲）：① `npu-smi set -t reset` 全 8 卡（标准处置，推荐）② 删 ipourma 陈旧 IPv6 段（python ioctl，有破坏 RoCE 面风险）③ 重启容器/驱动栈。诊断脚本留存 `/tmp/hccl_sanity_check.py`（torchrun 8 卡一键复测；注意脚本 expected 公式 world=1 时有笔误）。

## 四、下一步（HCCL 恢复后）

1. `torchrun --nproc_per_node=8 --master_port=29517 /tmp/hccl_sanity_check.py` 验证 PASS
2. 续跑 topup4k 量化（L36-42）→ rebuild_bf16 → msmodelslim（modelslim.sh 的 MODEL_PATH 需改为 topup4k BF16、SAVE_PATH 换新）→ vllm `vllm_start_dp.sh`（serve 路径改 topup4k 部署权重）→ `aisbench.sh`（GPQA/aime 各 3 轮，结果归 `ais_bench_logs/vptq-topup4k-tile16/`）
3. dualnorm 评测补跑（部署权重已就绪）
4. 对照基准：tile16+旧Hessian GPQA 65.53/aime 71.11；topup1024 65.87/74.44；无 VPTQ 基线 73.99/71.67
