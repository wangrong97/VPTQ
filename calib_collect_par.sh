#!/bin/bash
# =============================================================================
# calib_data 采集 —— 4 分片并行版 v3（2026-09-22，stream + 独立子目录）
#
# 演进：
#   v1 map 2卡/分片 → HBM 爆（map 按全 43 层 divmod 卡数）
#   v2 stream 共享 output-dir → 单实例锁（.collect.lock）互斥，3 分片秒退
#   v3 每分片独立子目录绕锁；层文件天然不重叠，全部完成后合并到主目录
# 资源：stream 每进程全模型驻 RAM（543GB×4=2.2TB<3TB），当前层上独占物理
#   卡（单卡 HBM 峰值 ~25GB）；CPU addmm 累积 ~67核/进程 ×4 < 384核。
# 口径：同 data_file/seed（rng 确定性 → 各分片样本一致）、无 floor、无
#   save-inv、128×4096 batch4、CPU 累积。已采层自动跳过。
# 完成判据：collect.log 出现 CALIB COLLECTION DONE（管线 S1 依赖）。
# =============================================================================
set -u
cd /mnt/share/w00608002/work/VPTQ

OUT=/mnt/share/w00608002/weights/hessians/DeepSeek-V4-Flash-BF16-calibR9
CKPT=/mnt/share/weight/DeepSeek-V4-Flash-BF16
DATA_DIR=/mnt/share/w00608002/weights/calib_data
DATA_FILE=$DATA_DIR/calib_corpus_R9tau.jsonl
STAGING=/root/calib_staging
LOG=$OUT/collect.log
mkdir -p $OUT

npus_idle() {
  local out busy
  out=$(npu-smi info 2>/dev/null) || return 0
  if echo "$out" | grep -qE "python|vllm|torchrun"; then return 1; fi
  busy=$(echo "$out" | grep -oE "[0-9]+ */ *98304" | \
         awk '{gsub(/ /,"",$1); if ($1+0 > 10000) c++} END{print c+0}')
  [ "$busy" -eq 0 ]
}

# shard <start> <end> <phys_card> <shard_id>
shard() {
  local r0=$1 r1=$2 card=$3 sid=$4 i n attempt
  local outs=$OUT/shard$sid
  mkdir -p "$outs"
  for attempt in $(seq 1 60); do
    n=0
    for i in $(seq $r0 $((r1-1))); do
      [ -f "$outs/layer_$(printf %04d $i).pt" ] && n=$((n+1))
    done
    [ $n -ge $((r1-r0)) ] && { echo "=== shard$sid complete $n/$((r1-r0)) layers $(date) ===" >> "$LOG"; return 0; }
    echo "=== shard$sid attempt $attempt ($n/$((r1-r0)) layers, card $card) $(date) ===" >> "$LOG"
    ASCEND_RT_VISIBLE_DEVICES=$card python3 -m vptq.tools.deepseek_v4.collect_hessian \
      --ckpt $CKPT \
      --data-dir $DATA_DIR \
      --data-file $DATA_FILE \
      --output-dir $outs \
      --nsamples 128 --seqlen 4096 --batch-size 4 \
      --act-ckpt /root/act_ckpt_calib_s$sid.pt \
      --save-staging $STAGING --prefetch-next \
      --device npu --residency stream \
      --layer-range $r0:$r1 \
      >> $LOG 2>&1
    echo "=== shard$sid attempt $attempt died ($?), cooling 300s $(date) ===" >> "$LOG"
    sleep 300
    while ! npus_idle; do
      echo "=== NPU busy, deferring 60s $(date) ===" >> "$LOG"
      sleep 60
    done
  done
  echo "=== shard$sid EXHAUSTED ATTEMPTS $(date) ===" >> "$LOG"
  return 1
}

# 已采 L0-10（11 层，主目录），剩 L11-42 = 32 层 ÷ 4 分片
shard 11 19 0 0 &
shard 19 27 1 1 &
shard 27 35 2 2 &
shard 35 43 3 3 &

wait
ok=1
for sid in 0 1 2 3; do
  for f in "$OUT/shard$sid"/layer_*.pt; do
    [ -f "$f" ] || { ok=0; break; }
  done
done
if [ $ok -eq 1 ]; then
  mv "$OUT"/shard*/layer_*.pt "$OUT/" && rm -rf "$OUT"/shard0 "$OUT"/shard1 "$OUT"/shard2 "$OUT"/shard3
  echo "=== CALIB COLLECTION DONE $(date) ===" >> "$LOG"
else
  echo "=== PAR SHARDS INCOMPLETE $(date) ===" >> "$LOG"
fi
