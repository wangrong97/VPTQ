#!/bin/bash
# =============================================================================
# Hessian 补采看门狗 v3：v2 多轮小批量 + 抢占保护 + IO/前向优化
#
# v2 死锁分析（2026-09-17）：单层处理时间（前向+累积+保存 45GB NFS）在大冷组层
# 可达 ~15-20 分钟，超过 killer 周期（10-25 分钟波动）→ 层永远无法保存。对策：
# 多轮小批量独立进程——层文件即跨轮 checkpoint（未达标组带统计恢复、已达标组
# frozen、纯 carried 层跳写），任何时刻被杀只损失 ≤1 层。轮间 seed 递增不重叠。
#
# v3 增量（2026-09-17 晚，用户要求）：
#   - 抢占保护：被 kill 后冷却 300s，且循环确认 NPU 无他人计算进程
#     （HBM 回落到驱动常驻水平）才重启——不抢协作方任务
#   - --save-staging 本地 NVMe（层文件本地写 + 后台上传 NFS，移出关键路径）
#   - --prefetch-next（下一层 Hessian 文件内核预读，restore 命中页缓存）
#   - 4 轮 × 128 条（总量 512 不变，省一半进程重启）
#
# v3.1（18:05）：--batch-size 8 回退为 4 —— bs=8 在首个 indexer 层（L2）
# OOM（sparse_attn 的 einsum 中间张量 5GB > 卡上剩余 2.75GB）；bs=4 实测
# 吞吐持平（0.32s/条），回退无损失。教训：改 batch 前先用 --n-layer-limit
# 4 验证 indexer 层。
#
# 停止方法:  touch $OUT/STOP && pkill -f collect_hessian; 删除 STOP 后重启
# 保活:      协作方会话内每分钟检查本脚本与 python 都不在时自动拉起（尊重 STOP）
# =============================================================================
set -u
cd /mnt/share/w00608002/work/VPTQ

OUT=/mnt/share/w00608002/weights/hessians/DeepSeek-V4-Flash-BF16-rpmix
# 09-18: 新容器无本地副本（/root/dsv4-weights 已不存在），改 NFS 直跑。
# 若 NFS 掉线导致 attempt 连续烧完，可再同步本地副本后改回。
CKPT=/mnt/share/weight/DeepSeek-V4-Flash-BF16
DATA=/mnt/share/w00608002/weights/RedPajama-Data-1T-Sample
STAGING=/root/topup_staging
LOG=$OUT/topup.log
# persistent round counter: fresh data comes from --topup-round (the
# k-th consecutive segment of each slice's document stream). Seeds are
# NOT enough — 5 of 6 slices are single-file so seed-shuffling never
# changes their text (verified: seeds 1/2/3 gave 100% identical samples,
# and the seed-1 batch overlapped the original seed-0 batch by 26%).
# Rollback 09-18: the polluted layers were restored from the pre-topup
# hardlink snapshot before this counter existed.
ROUND_FILE=$OUT/topup.round
bump_round() {
  local r=$(( $(cat $ROUND_FILE 2>/dev/null || echo 0) + 1 ))
  echo $r > $ROUND_FILE
  echo $r
}

npus_idle() {
  # idle = no compute process AND per-card HBM back near the driver
  # floor (~4.7GB); a dying task can leave the process list empty while
  # HBM is still draining
  local out busy
  out=$(npu-smi info 2>/dev/null) || return 0
  if echo "$out" | grep -qE "python|vllm|torchrun"; then return 1; fi
  busy=$(echo "$out" | grep -oE "[0-9]+ */ *98304" | \
         awk '{gsub(/ /,"",$1); if ($1+0 > 10000) c++} END{print c+0}')
  [ "$busy" -eq 0 ]
}

wait_cooldown() {
  # after a kill: cool down 5 min, then only relaunch once the cards are
  # actually free (never preempt someone else's job)
  echo "=== cooling down 300s before idle check $(date) ===" >> $LOG
  sleep 300
  while ! npus_idle; do
    echo "=== NPU busy (other task?), deferring 60s $(date) ===" >> $LOG
    sleep 60
  done
  echo "=== NPU idle again, safe to relaunch $(date) ===" >> $LOG
}

# first start: also wait for idle cards (don't preempt)
while ! npus_idle; do
  echo "=== NPU busy at startup, waiting 60s $(date) ===" >> $LOG
  sleep 60
done

# v3.3 (09-20): floor 1024→4000(8A.7 实证:floor=1024 把全盲组推进倒钟
# 坑底,MSE +38%;floor≥4000 才能推过坑)。rounds 4→8;每轮后扫 under-floor,
# 清零即提前结束,不跑空轮。
for round in $(seq 1 8); do
  [ -f $OUT/STOP ] && { echo "=== STOP requested, exit at round $round $(date) ===" >> $LOG; exit 0; }
  # BUGFIX 09-18: act_ckpt matches (nsamples,seqlen,batch) but NOT the
  # sample content — after round N finishes, next_layer=43 survives and
  # round N+1 "resumes" past the last layer without running anything
  # (rounds 2-4 all no-op'd this way). Each round must start fresh;
  # attempt-level retries inside a round KEEP the ckpt (that's the
  # killer-resume mechanism).
  rm -f /root/act_ckpt_topup.pt /root/act_ckpt_topup.pt.tmp
  RIDX=$(bump_round)
  for attempt in $(seq 1 30); do
    [ -f $OUT/STOP ] && { echo "=== STOP requested $(date) ===" >> $LOG; exit 0; }
    echo "=== topup round $round (data-round $RIDX) attempt $attempt $(date) ===" >> $LOG
    python3 -m vptq.tools.deepseek_v4.collect_hessian \
      --ckpt $CKPT \
      --data-dir $DATA \
      --output-dir $OUT \
      --nsamples 128 --seqlen 4096 --batch-size 4 \
      --token-floor 4000 --max-extra-samples 128 \
      --topup-seed 1 --topup-round $RIDX \
      --act-ckpt /root/act_ckpt_topup.pt \
      --save-staging $STAGING --prefetch-next \
      --residency map --devices npu:0,npu:1,npu:2,npu:3,npu:4,npu:5,npu:6,npu:7 \
      >> $LOG 2>&1 && { echo "=== round $round DONE $(date) ===" >> $LOG; break; }
    echo "=== round $round attempt $attempt died ($?) $(date) ===" >> $LOG
    wait_cooldown
  done
  # round finished: rescan; exit early once no group is under the floor
  # (scan failure is fail-open — empty UNDER just keeps the loop going)
  UNDER=$(OUT=$OUT python3 -c "
import os
from vptq.tools.deepseek_v4.collect_hessian import _scan_layers
_, u = _scan_layers(os.environ['OUT'], 4000)
print(sum(len(v) for v in u.values()))" 2>/dev/null | tail -1)
  echo "=== round $round done, groups still <floor(4000): ${UNDER:-scan-failed} $(date) ===" >> $LOG
  [ "${UNDER:-1}" = "0" ] && { echo "=== FLOOR 4000 reached on all groups, exit early $(date) ===" >> $LOG; exit 0; }
done
echo "=== ALL ROUNDS DONE $(date) ===" >> $LOG
