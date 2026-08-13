#!/usr/bin/env bash
# wait_ready.sh <server_log> <timeout_s>
# 轮询 server 日志（宿主机可见，因仓库已挂进容器）：
#   见 "The server is fired up and ready to roll" → exit 0（就绪）
#   见致命错误签名（OOM / prefill runner 失败 / Traceback）→ exit 1（崩了，快速失败）
#   超时                                              → exit 2
# profile.sh wrap 会把本脚本当"被测命令"：脚本一返回就停 startup 窗口采集，
# 返回码进 run_meta.json 的 workload_rc；run.sh 据此决定跳过 bench 与否。
set -uo pipefail
LOG="${1:?server_log path}"
TIMEOUT="${2:-600}"

READY_RE='The server is fired up and ready to roll'
FATAL_RE='CUDA error: out of memory|OutOfMemoryError|Fail when using backend|Traceback \(most recent call last\)'

start="$(date +%s)"
echo "[wait_ready] 等 server 就绪（timeout ${TIMEOUT}s）：$LOG"
while :; do
  if [[ -f "$LOG" ]]; then
    if grep -qE "$READY_RE" "$LOG"; then
      echo "[wait_ready] READY ✓"
      exit 0
    fi
    if grep -qE "$FATAL_RE" "$LOG"; then
      echo "[wait_ready] 致命错误签名 ✗："
      grep -nE "$FATAL_RE" "$LOG" | head -5
      exit 1
    fi
  fi
  now="$(date +%s)"
  if (( now - start >= TIMEOUT )); then
    echo "[wait_ready] 超时 ${TIMEOUT}s ✗"
    exit 2
  fi
  sleep 2
done
