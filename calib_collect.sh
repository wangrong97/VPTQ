#!/bin/bash
# =============================================================================
# calib_data 基线采集(2026-09-20,R9tau 评测对齐语料)
#
# 基线机制 = 原始 RedPajama 采集:无 --token-floor(全组 hook、无 cap)、
# 无 --save-inv(inv 后补)、128×4096。差异仅数据源(--data-file 单文件
# 行序流式拼接切块,见 build_samples data_file 分支)。语料 203k token
# → 49 条整块后耗尽(无 pad,log 有 warning 属预期)。
# tau2 12 条长 policy 占 43% 块:严格基线拼接,配比未修正(用户确认)。
# 抗 killer:attempt 循环 + act_ckpt(attempt 级保留)+ staging 上传。
# =============================================================================
set -u
cd /mnt/share/w00608002/work/VPTQ

OUT=/mnt/share/w00608002/weights/hessians/DeepSeek-V4-Flash-BF16-calibR9
CKPT=/mnt/share/weight/DeepSeek-V4-Flash-BF16
DATA_FILE=/mnt/share/w00608002/weights/calib_data/calib_corpus_R9tau.jsonl
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

for attempt in $(seq 1 30); do
  echo "=== calib collect attempt $attempt $(date) ===" >> $LOG
  python3 -m vptq.tools.deepseek_v4.collect_hessian \
    --ckpt $CKPT \
    --data-dir /mnt/share/w00608002/weights/calib_data \
    --data-file $DATA_FILE \
    --output-dir $OUT \
    --nsamples 128 --seqlen 4096 --batch-size 4 \
    --act-ckpt /root/act_ckpt_calib.pt \
    --save-staging $STAGING --prefetch-next \
    --residency map --devices npu:0,npu:1,npu:2,npu:3,npu:4,npu:5,npu:6,npu:7 \
    >> $LOG 2>&1 && { echo "=== CALIB COLLECTION DONE $(date) ===" >> $LOG; exit 0; }
  echo "=== attempt $attempt died ($?), cooling 300s $(date) ===" >> $LOG
  sleep 300
  while ! npus_idle; do
    echo "=== NPU busy, deferring 60s $(date) ===" >> $LOG
    sleep 60
  done
done
echo "=== CALIB COLLECT EXHAUSTED ATTEMPTS $(date) ===" >> $LOG
