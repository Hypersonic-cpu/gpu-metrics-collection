#!/usr/bin/env bash
# NVSwitch 实验跑测器：宿主机同时起【两条】dcgmi dmon 流，中间跑 cross-card P2P workload 容器打满 NVLink。
#   - 流A（switch 侧）: nvswitch 实体的 tx/rx 吞吐字段（全 12 台交换机）。
#   - 流B（GPU 侧基准）: 源/目的 GPU 的 nvlink_tx/rx_bytes(1011/1012) + 449，做口径基准（已知可信，见 §5.2）。
# 两条流用同一 interval/count，夹住同一时间窗（idle 头 -> workload -> idle 尾）。日志落 ./logs/。
#
# 用法：run_probe.sh <name> <src_gpu> <dst_gpu> [size_mb] [seconds]
#   name    : 实验基名（日志文件用）
#   src_gpu : 源物理 GPU id（容器内 ord0）
#   dst_gpu : 目的物理 GPU id（容器内 ord1）
#   size_mb : 每次 memcpy 大小 MB（默认 2048）
#   seconds : workload 持续秒数（默认 30）
# 环境变量：
#   DMON_MS : 采样间隔 ms（默认 1000=1Hz smoke；正式用 100=10Hz）
#   SWITCHES: 采哪些 switch（默认全 12 台）。dcgmi dmon 不支持 nvswitch:* wildcard，必须显式列。
# 例：
#   experiments/nvswitch/run_probe.sh smoke        1 2            # 1Hz smoke
#   DMON_MS=100 experiments/nvswitch/run_probe.sh p2p_10hz 1 2    # 10Hz 正式
set -euo pipefail

NAME="$1"; SRC="$2"; DST="$3"; SIZE_MB="${4:-2048}"; SECONDS_RUN="${5:-30}"
DMON_MS="${DMON_MS:-1000}"
SWITCHES="${SWITCHES:-nvswitch:0,nvswitch:1,nvswitch:2,nvswitch:3,nvswitch:4,nvswitch:5,nvswitch:6,nvswitch:7,nvswitch:8,nvswitch:9,nvswitch:10,nvswitch:11}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# 字段集：
#  switch 侧: 861/862(聚合 tx/rx, SWTX/SWRX) 780/781(per-link tx/rx, SWLNKTX/RX) 843/897(nvlink_*_bandwidth_total, MB/s)
#  GPU  侧: 1011/1012(nvlink_tx/rx_bytes, 口径基准) 449(nvlink_bandwidth_total) 843/897(per-GPU nvlink 总带宽 MB/s)
SW_FIELDS="${SW_FIELDS:-861,862,780,781,843,897}"
GPU_FIELDS="${GPU_FIELDS:-1011,1012,449,843,897}"

# 采样窗口要 > 容器启动(~10s) + workload(SECONDS_RUN)。多留 15s 余量。
COUNT="${COUNT:-$(( (SECONDS_RUN + 25) * 1000 / DMON_MS ))}"

LOGDIR="$HERE/logs/freq$((1000/DMON_MS))"; mkdir -p "$LOGDIR"
SW_LOG="$LOGDIR/dcgmi_switch_${NAME}.txt"
GPU_LOG="$LOGDIR/dcgmi_gpu_${NAME}.txt"
WL_LOG="$LOGDIR/workload_${NAME}.txt"
IMG="nvcr.io/nvidia/pytorch:25.12-py3"

echo "[run_probe] name=$NAME src_gpu=$SRC dst_gpu=$DST size=${SIZE_MB}MB run=${SECONDS_RUN}s interval=${DMON_MS}ms count=$COUNT"
echo "[run_probe] switch fields=$SW_FIELDS -> $SW_LOG"
echo "[run_probe] gpu    fields=$GPU_FIELDS (-i $SRC,$DST) -> $GPU_LOG"

# 1) 两条 dmon 后台起（先起，天然覆盖 workload 头尾 idle）
dcgmi dmon -e "$SW_FIELDS"  -i "$SWITCHES" -d "$DMON_MS" -c "$COUNT" > "$SW_LOG"  2>&1 &
SW_PID=$!
dcgmi dmon -e "$GPU_FIELDS" -i "$SRC,$DST" -d "$DMON_MS" -c "$COUNT" > "$GPU_LOG" 2>&1 &
GPU_PID=$!

# 给 dmon 一点起头 idle 基线再上负载
sleep 3

# 2) 前台跑 workload 容器（暴露源/目的物理卡；容器内重编号 cuda:0=src, cuda:1=dst）
docker run --rm --gpus "\"device=$SRC,$DST\"" -v "$HERE/workloads":/w "$IMG" \
  python /w/nvlink_p2p_saturate.py "$SIZE_MB" "$SECONDS_RUN" 2>&1 | tee "$WL_LOG" \
  | grep -E "START|END|props|P2P|CANACCESS|Error|error|WARN" || true

echo "[run_probe] workload done, 等两条 dmon 收尾 ..."
wait "$SW_PID" "$GPU_PID"
echo "[run_probe] 完成。switch 日志: $SW_LOG ; gpu 日志: $GPU_LOG ; workload: $WL_LOG"
