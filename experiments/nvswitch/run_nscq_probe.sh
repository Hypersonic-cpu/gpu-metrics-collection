#!/usr/bin/env bash
# 直连 NSCQ 读 NVSwitch per-port throughput_counters —— DCGM 没接的那条路，但**真能拿到过交换机的流量**。
# 编译 workloads/nscq_throughput_probe.cpp（host 侧，链接 libnvidia-nscq），read#1 -> workload -> read#2，差分。
# NSCQ/dcgmi 都是宿主机二进制（CLAUDE.md），本 probe 同理在宿主机跑；本机免 sudo 即可读。
#
# 用法：run_nscq_probe.sh <name> <workload> <container_devices> [size_mb] [seconds]
#   workload : p2p | nvls | ring | idle
#   例：run_nscq_probe.sh p2p  p2p  1,2      2048 20
#       run_nscq_probe.sh nvls nvls 1,2,3,4 512  20
#       run_nscq_probe.sh idle idle -        -    20     # 纯 idle 基线
set -euo pipefail

NAME="$1"; WL="$2"; CDEV="${3:--}"; SIZE_MB="${4:-2048}"; SECONDS_RUN="${5:-20}"
HERE="$(cd "$(dirname "$0")" && pwd)"
NSCQ_INC="${NSCQ_INC:-$HOME/Repos/DCGM/sdk/nvidia/nscq}"
IMG="nvcr.io/nvidia/pytorch:25.12-py3"
LOGDIR="$HERE/logs/nscq_direct"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/nscq_${NAME}.txt"
BIN="${TMPDIR:-/tmp}/nscq_probe"   # 编译产物放 tmp，不塞进 logs/

echo "[nscq_probe] build probe (host g++, -lnvidia-nscq) ..."
g++ -O2 -I"$NSCQ_INC" "$HERE/workloads/nscq_throughput_probe.cpp" -o "$BIN" -L/usr/lib/x86_64-linux-gnu -lnvidia-nscq

# probe 自己 read#1 -> sleep(SLEEP) -> read#2；SLEEP 要覆盖容器启动(~8s)+workload
SLEEP=$(( SECONDS_RUN + 15 ))
echo "[nscq_probe] name=$NAME workload=$WL devices=$CDEV size=${SIZE_MB}MB run=${SECONDS_RUN}s window=${SLEEP}s -> $LOG"
"$BIN" "$SLEEP" > "$LOG" 2>&1 &
PROBE=$!
sleep 5   # 让 read#1 先落地

case "$WL" in
  idle) echo "[nscq_probe] idle baseline (no workload)";;
  p2p)  docker run --rm --gpus "\"device=$CDEV\"" -v "$HERE/workloads":/w "$IMG" \
          python /w/nvlink_p2p_saturate.py "$SIZE_MB" "$SECONDS_RUN" 2>&1 | grep -E "CANACCESS|END|GB/s" || true;;
  nvls) docker run --rm --gpus "\"device=$CDEV\"" --ipc=host --shm-size=1g -e NCCL_ALGO=NVLS \
          -e NCCL_DEBUG=WARN -v "$HERE/workloads":/w "$IMG" \
          python /w/nccl_allreduce.py "$SIZE_MB" "$SECONDS_RUN" 2>&1 | grep -E "_END|GB/s|ALGO" || true;;
  ring) docker run --rm --gpus "\"device=$CDEV\"" --ipc=host --shm-size=1g -e NCCL_ALGO=Ring \
          -e NCCL_DEBUG=WARN -v "$HERE/workloads":/w "$IMG" \
          python /w/nccl_allreduce.py "$SIZE_MB" "$SECONDS_RUN" 2>&1 | grep -E "_END|GB/s|ALGO" || true;;
esac

echo "[nscq_probe] 等 probe read#2 ..."
wait $PROBE
echo "--- SUMMARY ---"; grep -E "SUMMARY|read#" "$LOG"
echo "[nscq_probe] 完整日志: $LOG"
