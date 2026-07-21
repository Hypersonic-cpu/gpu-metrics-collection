#!/usr/bin/env bash
# profile.sh —— dcgmi dmon 旁路采集入口（MVP，尽量简洁）
#
# 做三件事：
#   1) 起 `dcgmi dmon`，每行用 stamp.py 打上墙钟 epoch -> runs/<id>/dcgm_raw.log
#   2) 在 runs/<id>/marks.txt 记录关键时刻（collector/workload 的 start/stop，epoch）
#   3) 两种模式：
#        wrap  : profile.sh [opts] -- <命令>   起采集→(lead)→跑命令→(lag)→停采集
#        attach: profile.sh [opts] --duration N  被测程序在别处已在跑，只采一个 N 秒窗口
#
#   wrap 边距（保证踩全 + 看到流量回落空载）：
#        --lead N  命令启动【前】先空采 N 秒基线（默认 2）
#        --lag  N  命令结束【后】再续采 N 秒等流量回落（默认 3）；设 0 关闭
#
# dcgmi 在宿主机跑（本机 DCGM 装在 host）；-i 用【物理卡号】。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 默认参数 ----
GPUS=""                 # 物理卡号，逗号分隔，如 "4,5"（必填）
FIELDS="pass1_core"     # tool/dcgmi/<name>.txt
INTERVAL=1000           # ms，默认低频 smoke；正式采集用 100（下限 100）
OUTDIR="$HERE/../runs"
DURATION=""             # attach 模式的秒数
LEAD=2                  # wrap: 命令前空采基线秒数
LAG=3                   # wrap: 命令后续采（等流量回落空载）秒数
CMD=()                  # wrap 模式的命令（-- 之后）

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

# ---- 解析参数 ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)        GPUS="$2"; shift 2;;
    --fields)      FIELDS="$2"; shift 2;;
    --interval-ms) INTERVAL="$2"; shift 2;;
    --out)         OUTDIR="$2"; shift 2;;
    --duration)    DURATION="$2"; shift 2;;
    --lead)        LEAD="$2"; shift 2;;
    --lag)         LAG="$2"; shift 2;;
    --) shift; CMD=("$@"); break;;
    -h|--help) usage;;
    *) echo "unknown arg: $1" >&2; usage;;
  esac
done

# ---- 校验 ----
command -v dcgmi >/dev/null || { echo "ERROR: 找不到 dcgmi（本机 DCGM 应装在宿主机）" >&2; exit 1; }
[[ -n "$GPUS" ]] || { echo "ERROR: 必须用 --gpus 指定物理卡号，如 --gpus 4,5" >&2; exit 1; }
[[ "$INTERVAL" -ge 100 ]] || { echo "ERROR: --interval-ms 不得 <100（NVML GPM 有效地板 ~100ms）" >&2; exit 1; }
FIELDS_FILE="$HERE/dcgmi/${FIELDS}.txt"
[[ -f "$FIELDS_FILE" ]] || { echo "ERROR: 字段文件不存在: $FIELDS_FILE" >&2; exit 1; }
FIELD_IDS="$(grep -oE '^[0-9]+' "$FIELDS_FILE" | paste -sd, -)"
[[ -n "$FIELD_IDS" ]] || { echo "ERROR: 字段文件里没有有效 id: $FIELDS_FILE" >&2; exit 1; }
if [[ ${#CMD[@]} -eq 0 && -z "$DURATION" ]]; then
  echo "ERROR: 要么 wrap 模式（-- <命令>），要么 attach 模式（--duration N）" >&2; exit 1
fi

# ---- 建 run 目录 ----
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUNDIR="$OUTDIR/$RUN_ID"
mkdir -p "$RUNDIR"
RAW="$RUNDIR/dcgm_raw.log"
MARKS="$RUNDIR/marks.txt"
FIFO="$RUNDIR/.dmon.fifo"
: > "$MARKS"

mark() { printf '%s\t%s\n' "$(date +%s.%3N)" "$1" >> "$MARKS"; }

echo "[profile] run=$RUN_ID gpus=$GPUS interval=${INTERVAL}ms fields=$FIELDS ($FIELD_IDS)"
echo "[profile] outdir=$RUNDIR"

# ---- 起采集：dcgmi dmon -> stamp.py（写 raw.log + 实时打屏）----
# stamp.py <raw> 会把 `<epoch>\t行` 写进 raw.log，同时把 `HH:MM:SS 行` 打到终端 -> 实时看进度。
mkfifo "$FIFO"
python3 "$HERE/stamp.py" "$RAW" < "$FIFO" &
STAMP_PID=$!
stdbuf -oL dcgmi dmon -e "$FIELD_IDS" -i "$GPUS" -d "$INTERVAL" > "$FIFO" 2>&1 &
DMON_PID=$!
mark "collector_start"
echo "[profile] 实时采样中（终端显示 HH:MM:SS + dmon 行；完整 epoch 存 $RAW）..."

cleanup() {
  mark "collector_stop"
  kill "$DMON_PID" 2>/dev/null || true
  wait "$DMON_PID" 2>/dev/null || true   # 关闭 FIFO 写端 -> stamp 收到 EOF 退出
  wait "$STAMP_PID" 2>/dev/null || true
  rm -f "$FIFO"
}
trap cleanup EXIT

RC=0
if [[ ${#CMD[@]} -gt 0 ]]; then
  # ---- wrap 模式 ----
  if [[ "$LEAD" != "0" ]]; then echo "[profile] lead: 先空采 ${LEAD}s 基线..."; sleep "$LEAD"; fi
  mark "workload_start"
  echo "[profile] wrap: ${CMD[*]}"
  ( "${CMD[@]}" ) > "$RUNDIR/workload.log" 2>&1 || RC=$?
  mark "workload_end"
  echo "[profile] workload 结束 (rc=$RC)，见 $RUNDIR/workload.log"
  if [[ "$LAG" != "0" ]]; then echo "[profile] lag: 续采 ${LAG}s 等流量回落空载..."; sleep "$LAG"; fi
else
  # ---- attach 模式 ----
  mark "window_start"
  echo "[profile] attach: 采 ${DURATION}s 窗口（被测程序请在别处运行）"
  sleep "$DURATION"
  mark "window_end"
fi

cleanup; trap - EXIT

# ---- run 元数据 ----
cat > "$RUNDIR/run_meta.json" <<EOF
{
  "run_id": "$RUN_ID",
  "host": "$(uname -n)",
  "gpus": "$GPUS",
  "fields": "$FIELDS",
  "field_ids": "$FIELD_IDS",
  "interval_ms": $INTERVAL,
  "mode": "$([[ ${#CMD[@]} -gt 0 ]] && echo wrap || echo attach)",
  "workload_rc": $RC
}
EOF

echo "[profile] 完成。下一步: python3 $HERE/metrics.py parse $RUNDIR"
exit "$RC"
