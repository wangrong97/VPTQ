#!/bin/bash
# 元看门狗：检查补采看门狗是否存活，不在则重启（cron 每 5 分钟调用）。
# 背景：2026-09-17 14:41 killer 一次杀掉了看门狗 bash + python 进程组
# （setsid 也未能幸免），补采停摆 38 分钟无人重启。cron 是最外层，
# 不依赖任何用户进程。
LOG=/mnt/share/w00608002/weights/hessians/DeepSeek-V4-Flash-BF16-rpmix/topup.log
STOP=/mnt/share/w00608002/weights/hessians/DeepSeek-V4-Flash-BF16-rpmix/STOP
if [ -f "$STOP" ]; then
    exit 0  # 人工停止信号（协作方 2026-09-17 15:50 touch），不自动拉起
fi
if pgrep -f "topup_rpmix.sh" > /dev/null; then
    exit 0
fi
# 看门狗不在：确认主任务是否已完成（完成则不重启）
if grep -aq "TOPUP DONE\|ALL ROUNDS DONE" "$LOG" 2>/dev/null; then
    exit 0
fi
echo "=== meta-watchdog: topup_rpmix.sh dead, restarting $(date) ===" >> "$LOG"
cd /mnt/share/w00608002/work/VPTQ
setsid nohup bash topup_rpmix.sh > /dev/null 2>&1 &
