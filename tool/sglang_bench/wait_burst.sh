#!/usr/bin/env bash
# wait_burst.sh <server_log> <from_line> <timeout_s> [min_new_seq]
# 轮询 server 日志，等"真正的压测流量到了"，把那一刻的 epoch 打到 stdout（其它信息走 stderr）。
#
# 为什么要它：nsys 的采集窗口按秒排在 bench 开始之后，但 bench_serving 先发一条 warmup 请求、
# 再等 tokenizer/连接就绪才猛发，这段前戏时长会漂（实测 ~8s，跟模型/镜像有关）。
# 以"第一条真正的压测 Prefill batch"为锚点，窗口位置就不受这段漂移影响。
#
# 判据：Prefill batch 且 #new-seq >= min_new_seq（默认 4）。
#       warmup 那几条恒为 #new-seq: 1，压测第一批一次就 admit 十几条，区分度足够。
# from_line：只看这一行之后的新内容（= 发 bench 前的 wc -l），免得匹配到启动期的旧行。
# 超时 → exit 1（调用方自己决定退回"以 bench 开始时刻为锚点"还是放弃）。
set -uo pipefail
LOG="${1:?server_log path}"
FROM="${2:-0}"
TIMEOUT="${3:-180}"
MIN_NEW_SEQ="${4:-4}"

start="$(date +%s)"
echo "[wait_burst] 等压测流量（从第 ${FROM} 行起，#new-seq>=${MIN_NEW_SEQ}，timeout ${TIMEOUT}s）" >&2
while :; do
  if [[ -f "$LOG" ]] && tail -n "+$((FROM + 1))" "$LOG" 2>/dev/null \
      | awk -v m="$MIN_NEW_SEQ" '/Prefill batch/ && match($0, /#new-seq: [0-9]+/) {
             if (substr($0, RSTART + 10, RLENGTH - 10) + 0 >= m) { found = 1; exit }
           } END { exit !found }'; then
    date +%s.%3N
    echo "[wait_burst] 压测流量已到 ✓" >&2
    exit 0
  fi
  if (( $(date +%s) - start >= TIMEOUT )); then
    echo "[wait_burst] 超时 ${TIMEOUT}s ✗（没等到 #new-seq>=${MIN_NEW_SEQ} 的 Prefill batch）" >&2
    exit 1
  fi
  sleep 0.2
done
