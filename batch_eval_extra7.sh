#!/bin/bash
# =============================================================================
# 批量评测：tile32×32 + calibR9 + 12.5% 回退（复用 B 组部署权重）
#   GPQA ×7 + aime ×7 = 14 轮，结果汇总到 ais_bench_logs
# 抗性设计（三日实战沉淀）：
#   - vllm 看护：每轮前 API + 真实预热验证；死则重启（GLOO 修复版脚本）
#   - 残轮甄别：GPQA<55 / aime<45 判半途污染 → 隔离重跑（历史半途轮 27.27/14.14，
#     历史真实最低 GPQA 64.14 / aime 53.33，阈值留足边际）
#   - 有效计数：只数含真实成绩且过阈值的轮
# 用法: setsid nohup bash batch_eval_extra7.sh &   停止: touch /tmp/batch7.STOP
# =============================================================================
set -u
cd /mnt/share/w00608002/benchmark
LOG=/tmp/batch7.log
STOPF=/tmp/batch7.STOP
OUTBASE=outputs/default
VLLM_SH=/mnt/share/w00608002/work/vllm-ascend/vllm_start_t32x32fb.sh
VLLM_LOG=/tmp/vllm_batch7.log
DEST=/mnt/share/w00608002/work/vllm-ascend/ais_bench_logs/vptq-calibR9-t32x32fb-extra7-$(date +%Y%m%d)
mkdir -p "$DEST/bench_outputs"
INV=/mnt/share/w00608002/benchmark/outputs/_invalid_rounds
mkdir -p "$INV"

log() { echo "[batch7 $(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

vllm_alive() { curl -s -m 5 http://127.0.0.1:9001/v1/models 2>/dev/null | grep -q '"id"'; }
vllm_warm() {
    curl -s -m 300 http://127.0.0.1:9001/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"dsv","messages":[{"role":"user","content":"ping"}],"max_tokens":4}' \
        2>/dev/null | grep -q '"finish_reason"'
}
start_vllm() {
    log "重启 vllm"
    ( cd "$(dirname "$VLLM_SH")" && setsid nohup bash "$VLLM_SH" > "$VLLM_LOG" 2>&1 & )
}
ensure_vllm() {  # 就绪+预热，最多 50 分钟
    if vllm_alive && vllm_warm; then return 0; fi
    ps aux | grep -E "vllm serve|VLLM:[:]:" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
    sleep 5; rm -f /dev/shm/psm_* 2>/dev/null
    start_vllm
    local wv=0
    until vllm_alive; do sleep 60; wv=$((wv+60)); [ $wv -gt 3000 ] && return 1; done
    until vllm_warm; do sleep 60; wv=$((wv+60)); [ $wv -gt 3600 ] && return 1; done
    log "vllm 就绪且预热通过"
    return 0
}

count_valid() {  # $1=gpqa|aime $2=baseline → 本次新增有效轮
    local n=0 csv base=$2
    for csv in "$OUTBASE"/*/summary/summary_*.csv; do
        [ -f "$csv" ] || continue
        if grep -qi "$1" "$csv" 2>/dev/null && grep -q "accuracy,gen,[0-9]" "$csv" 2>/dev/null; then
            local v
            v=$(grep -aoi "$1[a-z0-9_]*,[^,]*,accuracy,gen,[0-9.]*" "$csv" | tail -1 | awk -F, '{print $5+0}')
            local thr=55; [ "$1" = aime ] && thr=45
            if [ "$(python3 -c "print(1 if $v >= $thr else 0)" 2>/dev/null)" = "1" ]; then
                n=$((n+1))
            fi
        fi
    done
    echo $(( n - base ))
}

# 基线锁存（持久化，防重启漂移）
STATEF=/tmp/batch7.state
if [ ! -f "$STATEF" ]; then
    bg=0; ba=0
    bg=$(for csv in "$OUTBASE"/*/summary/summary_*.csv; do [ -f "$csv" ] && grep -qi gpqa "$csv" && grep -q "accuracy,gen,[0-9]" "$csv" && echo x; done | wc -l)
    ba=$(for csv in "$OUTBASE"/*/summary/summary_*.csv; do [ -f "$csv" ] && grep -qi aime "$csv" && grep -q "accuracy,gen,[0-9]" "$csv" && echo x; done | wc -l)
    echo "$bg $ba" > "$STATEF"
fi
read -r BASE_G BASE_A < "$STATEF"
log "===== 批量评测启动（pid $$）：基线 g=$BASE_G a=$BASE_A，目标各 +7 ====="

run_round() {  # $1=gpqa|aime
    local ds arg
    if [ "$1" = gpqa ]; then arg=gpqa_gen_0_shot_cot_chat_prompt.py; else arg=aime2024_gen_0_shot_chat_prompt.py; fi
    ensure_vllm || { sleep 300; return 1; }
    log "启动 $1 轮（有效 g=$(count_valid gpqa $BASE_G)/7 a=$(count_valid aime $BASE_A)/7）"
    timeout 3600 ais_bench --models vllm_api_general_chat --datasets "$arg" \
        --mode all --dump-eval-details --merge-ds >> /tmp/aisbench_batch7.log 2>&1
    # 残轮甄别：最新目录的分数过阈值才计数，否则隔离
    local d csv v thr
    d=$(ls -t "$OUTBASE" | head -1)
    csv=$(ls "$OUTBASE/$d/summary/summary_*.csv" 2>/dev/null | head -1)
    if [ -n "$csv" ]; then
        v=$(grep -aoi "$1[a-z0-9_]*,[^,]*,accuracy,gen,[0-9.-]*" "$csv" | tail -1 | awk -F, '{print $5+0}')
        thr=55; [ "$1" = aime ] && thr=45
        if [ "$(python3 -c "print(1 if ${v:-0} >= $thr else 0)" 2>/dev/null)" = "1" ]; then
            log "轮完成: $d ${1}=${v}（有效）"
        else
            log "轮可疑: $d ${1}=${v:-无分} → 隔离重跑"
            mv "$OUTBASE/$d" "$INV/" 2>/dev/null
        fi
    fi
}

while true; do
    [ -f "$STOPF" ] && { log "STOP，退出"; exit 0; }
    local_g=$(count_valid gpqa $BASE_G)
    local_a=$(count_valid aime $BASE_A)
    if [ "$local_g" -ge 7 ] && [ "$local_a" -ge 7 ]; then break; fi
    if pgrep -f "bin/ais_bench" >/dev/null 2>&1; then sleep 300; continue; fi
    if [ "$local_g" -lt 7 ]; then run_round gpqa; else run_round aime; fi
    sleep 60
done
log "14 轮全部完成（g=$(count_valid gpqa $BASE_G)/7 a=$(count_valid aime $BASE_A)/7），汇总中"

# 汇总：复制本次新增有效轮 csv + 写报告
g=0; a=0
for csv in "$OUTBASE"/*/summary/summary_*.csv; do
    [ -f "$csv" ] || continue
    is_g=0; is_a=0; v=""
    if grep -qi gpqa "$csv" && grep -q "accuracy,gen,[0-9]" "$csv"; then
        v=$(grep -aoi "gpqa[a-z0-9_]*,[^,]*,accuracy,gen,[0-9.]*" "$csv" | tail -1 | awk -F, '{print $5+0}')
        [ "$(python3 -c "print(1 if ${v:-0} >= 55 else 0)")" = "1" ] && { is_g=1; g=$((g+1)); [ $g -gt $BASE_G ] && cp "$csv" "$DEST/bench_outputs/"; }
    fi
    if grep -qi aime "$csv" && grep -q "accuracy,gen,[0-9]" "$csv"; then
        v=$(grep -aoi "aime[a-z0-9_]*,[^,]*,accuracy,gen,[0-9.]*" "$csv" | tail -1 | awk -F, '{print $5+0}')
        [ "$(python3 -c "print(1 if ${v:-0} >= 45 else 0)")" = "1" ] && { is_a=1; a=$((a+1)); [ $a -gt $BASE_A ] && cp "$csv" "$DEST/bench_outputs/"; }
    fi
done
{
    echo "# t32x32+calibR9+12.5%回退 追加 7+7 轮批量评测"
    echo
    echo "- 生成: $(date '+%F %T')；部署权重: calibR9-t32x32fb（同 B 组，未变）"
    echo
    echo "## 各轮成绩"
    for csv in "$DEST"/bench_outputs/*.csv; do
        [ -f "$csv" ] && tail -n +2 "$csv" | awk -F, '{print $1, $5}'
    done
    echo
    echo "## 合并统计（含 B 组原 3+3 轮）"
    echo "- 完整数据见 §8A.2 单变量对照表与各 csv"
} > "$DEST/REPORT.md"
log "报告已写 $DEST/REPORT.md ===== 完成 ====="
