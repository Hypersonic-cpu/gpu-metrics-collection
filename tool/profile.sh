#!/usr/bin/env bash
# profile.sh —— GPU 接口 metric 旁路采集入口（MVP，尽量简洁）
#
# 两个后端（--backend，各自独立采集、互不合并；一个 run 只跑一个后端）：
#   dcgm     (默认) 起 `dcgmi dmon` 采 GPU 侧 metric（HBM/SM/PCIe/GPU 侧 NVLink/显存）
#            字段组 = tool/dcgmi/<name>.txt（--fields 选）。
#   nvswitch 起 tool/nvswitch_traffic 采过交换机的 NVLink 流量（每台 switch / 每端口 rx/tx）
#            配置组 = tool/nvswitch_traffic/<name>.conf（--sw-config 选）。
#
# 做三件事（两后端一致）：
#   1) 起采集子进程 -> runs/<id>/（dcgm_raw.log / nvswitch.csv，均带墙钟 epoch）
#   2) 在 runs/<id>/marks.txt 记关键时刻（collector/workload 的 start/stop，epoch）
#   3) 两种模式：
#        wrap  : profile.sh [opts] -- <命令>       起采集→(lead)→跑命令→(lag)→停采集
#        attach: profile.sh [opts] --duration N    被测程序在别处已在跑，只采一个 N 秒窗口
#
#   wrap 边距（保证踩全 + 看到流量回落空载）：
#        --lead N  命令启动【前】先空采 N 秒基线（默认 2）
#        --lag  N  命令结束【后】再续采 N 秒等流量回落（默认 3）；设 0 关闭
#
#   --note "一段话"  给这次采集写说明（方便之后分类）：写进 runs/<id>/README.md（单 run 详情）
#                    + 追加到 runs/README.md（总索引一行）+ 存进 run_meta.json 的 note 字段。
#
# dcgm 用 --gpus 选【物理卡号】；nvswitch 用配置组里的 switches 选 switch。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 默认参数 ----
BACKEND="dcgm"          # dcgm | nvswitch
GPUS=""                 # 物理卡号，逗号分隔，如 "4,5"（dcgm 必填；nvswitch 忽略）
FIELDS="pass1_core"     # dcgm 字段组: tool/dcgmi/<name>.txt
SW_CONFIG="sw_switch"   # nvswitch 配置组: tool/nvswitch_traffic/<name>.conf
INTERVAL=1000           # ms，默认低频 smoke；正式采集用 100（下限 100）
OUTDIR="$HERE/../runs"
DURATION=""             # attach 模式的秒数
LEAD=2                  # wrap: 命令前空采基线秒数
LAG=3                   # wrap: 命令后续采（等流量回落空载）秒数
NOTE=""                 # 这次采集的说明（--note），写进 README + run_meta.json
CMD=()                  # wrap 模式的命令（-- 之后）

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

# ---- 解析参数 ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)     BACKEND="$2"; shift 2;;
    --gpus)        GPUS="$2"; shift 2;;
    --fields)      FIELDS="$2"; shift 2;;
    --sw-config)   SW_CONFIG="$2"; shift 2;;
    --interval-ms) INTERVAL="$2"; shift 2;;
    --out)         OUTDIR="$2"; shift 2;;
    --duration)    DURATION="$2"; shift 2;;
    --lead)        LEAD="$2"; shift 2;;
    --lag)         LAG="$2"; shift 2;;
    --note)        NOTE="$2"; shift 2;;
    --) shift; CMD=("$@"); break;;
    -h|--help) usage;;
    *) echo "unknown arg: $1" >&2; usage;;
  esac
done

# ---- 通用校验 ----
[[ "$BACKEND" == "dcgm" || "$BACKEND" == "nvswitch" ]] || { echo "ERROR: --backend 只能是 dcgm 或 nvswitch" >&2; exit 1; }
[[ "$INTERVAL" -ge 100 ]] || { echo "ERROR: --interval-ms 不得 <100（采样有效地板 ~100ms）" >&2; exit 1; }
if [[ ${#CMD[@]} -eq 0 && -z "$DURATION" ]]; then
  echo "ERROR: 要么 wrap 模式（-- <命令>），要么 attach 模式（--duration N）" >&2; exit 1
fi

# ---- 后端专属校验 + 采集参数准备 ----
VIEW_PID=""; FIFO=""; SW_RAW_BOOL=false; CLEANED=0; RC=0   # RC 须在装 EXIT trap 前就有值
if [[ "$BACKEND" == "dcgm" ]]; then
  command -v dcgmi >/dev/null || { echo "ERROR: 找不到 dcgmi（本机 DCGM 应装在宿主机）" >&2; exit 1; }
  [[ -n "$GPUS" ]] || { echo "ERROR: dcgm 后端必须用 --gpus 指定物理卡号，如 --gpus 4,5" >&2; exit 1; }
  FIELDS_FILE="$HERE/dcgmi/${FIELDS}.txt"
  [[ -f "$FIELDS_FILE" ]] || { echo "ERROR: 字段文件不存在: $FIELDS_FILE" >&2; exit 1; }
  FIELD_IDS="$(grep -oE '^[0-9]+' "$FIELDS_FILE" | paste -sd, -)"
  [[ -n "$FIELD_IDS" ]] || { echo "ERROR: 字段文件里没有有效 id: $FIELDS_FILE" >&2; exit 1; }
else
  SW_BIN="$HERE/nvswitch_traffic/nvswitch_traffic"
  [[ -x "$SW_BIN" ]] || SW_BIN="$(command -v nvswitch_traffic || true)"
  [[ -n "$SW_BIN" && -x "$SW_BIN" ]] || { echo "ERROR: 找不到 nvswitch_traffic 二进制，请先构建: cd tool/nvswitch_traffic && make" >&2; exit 1; }
  SW_CONF_FILE="$HERE/nvswitch_traffic/${SW_CONFIG}.conf"
  [[ -f "$SW_CONF_FILE" ]] || { echo "ERROR: nvswitch 配置文件不存在: $SW_CONF_FILE" >&2; exit 1; }
  # sw_get <key> <default>：取 conf 里 key=value（忽略 # 注释与空白）
  sw_get() {
    local v
    v="$(sed -n -E "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*(.*)$/\1/p" "$SW_CONF_FILE" | head -1)"
    v="${v%%#*}"; v="$(printf '%s' "$v" | tr -d '[:space:]')"
    [[ -n "$v" ]] && printf '%s' "$v" || printf '%s' "$2"
  }
  SW_LEVEL="$(sw_get level switch)"
  SW_FIELDS="$(sw_get fields rx,tx,total)"
  SW_SWITCHES="$(sw_get switches all)"
  SW_RAW_FLAG=""; [[ "$(sw_get raw 0)" == "1" ]] && { SW_RAW_FLAG="-r"; SW_RAW_BOOL=true; }
fi

# ---- 建 run 目录 ----
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUNDIR="$OUTDIR/$RUN_ID"
mkdir -p "$RUNDIR"
MARKS="$RUNDIR/marks.txt"
: > "$MARKS"

mark() { printf '%s\t%s\n' "$(date +%s.%3N)" "$1" >> "$MARKS"; }

if [[ "$BACKEND" == "dcgm" ]]; then
  echo "[profile] backend=dcgm run=$RUN_ID gpus=$GPUS interval=${INTERVAL}ms fields=$FIELDS ($FIELD_IDS)"
else
  echo "[profile] backend=nvswitch run=$RUN_ID interval=${INTERVAL}ms config=$SW_CONFIG (level=$SW_LEVEL fields=$SW_FIELDS switches=$SW_SWITCHES${SW_RAW_FLAG:+ raw})"
fi
echo "[profile] outdir=$RUNDIR"

# ---- 起采集子进程（两后端都过 FIFO -> 副进程：写文件 + 实时打屏）----
# dcgm 副进程 = stamp.py（打墙钟 HH:MM:SS + 写 dcgm_raw.log）；nvswitch 副进程 = tee（CSV 自带 epoch，直接同步落盘 + 打屏）。
if [[ "$BACKEND" == "dcgm" ]]; then
  RAW="$RUNDIR/dcgm_raw.log"
  FIFO="$RUNDIR/.dmon.fifo"
  mkfifo "$FIFO"
  python3 "$HERE/stamp.py" "$RAW" < "$FIFO" &
  VIEW_PID=$!
  stdbuf -oL dcgmi dmon -e "$FIELD_IDS" -i "$GPUS" -d "$INTERVAL" > "$FIFO" 2>&1 &
  COLLECTOR_PID=$!
  echo "[profile] 实时采样中（终端显示 HH:MM:SS + dmon 行；完整 epoch 存 $RAW）..."
else
  CSV="$RUNDIR/nvswitch.csv"
  FIFO="$RUNDIR/.nvsw.fifo"
  mkfifo "$FIFO"
  tee "$CSV" < "$FIFO" &
  VIEW_PID=$!
  "$SW_BIN" -l "$SW_LEVEL" -e "$SW_FIELDS" -i "$SW_SWITCHES" $SW_RAW_FLAG -d "$INTERVAL" --csv \
      > "$FIFO" 2> "$RUNDIR/nvswitch.err" &
  COLLECTOR_PID=$!
  echo "[profile] 实时采样中 -> $CSV（终端同步打印 CSV 行；启动报错见 nvswitch.err）..."
fi
mark "collector_start"

json_escape() {   # 转义字符串使其能安全放进 JSON 双引号里（\ " 换行 制表）
  local s="$1"
  s="${s//\\/\\\\}"; s="${s//\"/\\\"}"; s="${s//$'\n'/\\n}"; s="${s//$'\t'/\\t}"
  printf '%s' "$s"
}

write_run_readme() {   # runs/<id>/README.md —— 单 run 详情（note 放最前，方便分类阅读）
  local mode; mode="$([[ ${#CMD[@]} -gt 0 ]] && echo wrap || echo attach)"
  {
    echo "# run $RUN_ID"
    echo
    [[ -n "$NOTE" ]] && { echo "$NOTE"; echo; }
    echo "## meta"
    echo "- **host**: $(uname -n)"
    echo "- **backend**: $BACKEND"
    if [[ "$BACKEND" == "dcgm" ]]; then
      echo "- **gpus**: $GPUS"
      echo "- **fields**: $FIELDS ($FIELD_IDS)"
    else
      echo "- **sw_config**: $SW_CONFIG (level=$SW_LEVEL fields=$SW_FIELDS switches=$SW_SWITCHES$([[ "$SW_RAW_BOOL" == true ]] && echo ' raw'))"
    fi
    echo "- **interval_ms**: $INTERVAL"
    echo "- **mode**: $mode"
    [[ ${#CMD[@]} -gt 0 ]] && echo "- **cmd**: \`${CMD[*]}\`"
  } > "$RUNDIR/README.md"
  return 0   # 上面 [[ ]] && echo 在 attach 模式返回非零，别让 set -e 中断 finalize
}

append_index() {   # runs/README.md —— 总索引，每个 run 追加一行（最新在下）
  local idx="$OUTDIR/README.md" n
  if [[ ! -f "$idx" ]]; then
    {
      echo "# runs 索引"
      echo
      echo "每次采集一行（最新在下）。详情见各 run 目录的 README.md。"
      echo
      echo "| run | backend | note |"
      echo "|---|---|---|"
    } > "$idx"
  fi
  n="${NOTE//$'\n'/ }"; n="${n//|/\\|}"; [[ -z "$n" ]] && n="—"   # 换行→空格、转义表格分隔符
  printf '| [%s](%s/README.md) | %s | %s |\n' "$RUN_ID" "$RUN_ID" "$BACKEND" "$n" >> "$idx"
}

finalize() {   # 写 run_meta.json + 两个 README（幂等：已存在就不重写）—— 信号中断(Ctrl-C)也能收尾
  [[ -f "$RUNDIR/run_meta.json" ]] && return 0
  local backend_meta
  if [[ "$BACKEND" == "dcgm" ]]; then
    backend_meta="\"gpus\": \"$GPUS\", \"fields\": \"$FIELDS\", \"field_ids\": \"$FIELD_IDS\""
  else
    backend_meta="\"sw_config\": \"$SW_CONFIG\", \"sw_level\": \"$SW_LEVEL\", \"sw_fields\": \"$SW_FIELDS\", \"sw_switches\": \"$SW_SWITCHES\", \"sw_raw\": $SW_RAW_BOOL"
  fi
  cat > "$RUNDIR/run_meta.json" <<EOF
{
  "run_id": "$RUN_ID",
  "host": "$(uname -n)",
  "backend": "$BACKEND",
  $backend_meta,
  "interval_ms": $INTERVAL,
  "mode": "$([[ ${#CMD[@]} -gt 0 ]] && echo wrap || echo attach)",
  "note": "$(json_escape "$NOTE")",
  "workload_rc": $RC
}
EOF
  write_run_readme
  append_index
}

cleanup() {    # 幂等：停采集 + 收尾；正常结束与信号退出(EXIT trap)都走这
  [[ "$CLEANED" == "1" ]] && return 0
  CLEANED=1
  mark "collector_stop"
  kill "$COLLECTOR_PID" 2>/dev/null || true
  wait "$COLLECTOR_PID" 2>/dev/null || true   # 关闭 FIFO 写端 -> tee/stamp 收到 EOF；nvswitch 收 SIGTERM 干净退出
  [[ -n "$VIEW_PID" ]] && { wait "$VIEW_PID" 2>/dev/null || true; }
  [[ -n "$FIFO" ]] && rm -f "$FIFO"
  finalize
  return 0
}
trap cleanup EXIT

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

cleanup; trap - EXIT   # 停采集 + 写 run_meta.json（finalize）

echo "[profile] 完成。下一步: python3 $HERE/metrics.py parse $RUNDIR"
exit "$RC"
