#!/usr/bin/env bash
# NCCL 集合通信版跑测器：多卡 all_reduce（NVLS/multimem 走 NVSwitch 计算引擎），宿主机同采 switch+GPU 计数器。
# 与 run_probe.sh（单卡 P2P）对照，验证 switch 侧字段在"真的过交换机计算引擎"的流量下是否点亮。
#
# 用法：run_nccl_probe.sh <name> <container_devices> [size_mb] [seconds]
#   name              : 实验基名
#   container_devices : 暴露给容器的物理卡列表，如 1,2,3,4（world_size=卡数）
# 环境变量：
#   NCCL_ALGO : 强制算法（NVLS / Ring / Tree；不设=auto）。NVLS=走 multimem。
#   DMON_MS   : 采样间隔 ms（默认 1000；正式用 100）
#   SWITCHES  : 采哪些 switch（默认全 12 台，wildcard 不支持须显式列）
set -euo pipefail

NAME="$1"; CDEV="$2"; SIZE_MB="${3:-512}"; SECONDS_RUN="${4:-30}"
DMON_MS="${DMON_MS:-1000}"
NCCL_ALGO_ENV="${NCCL_ALGO:-}"
SWITCHES="${SWITCHES:-nvswitch:0,nvswitch:1,nvswitch:2,nvswitch:3,nvswitch:4,nvswitch:5,nvswitch:6,nvswitch:7,nvswitch:8,nvswitch:9,nvswitch:10,nvswitch:11}"
HERE="$(cd "$(dirname "$0")" && pwd)"

SW_FIELDS="${SW_FIELDS:-861,862,780,781}"
GPU_FIELDS="${GPU_FIELDS:-1011,1012,449}"
GPU_ENTITIES="$CDEV"   # GPU 侧按物理 id 采（与容器暴露的相同）
COUNT="${COUNT:-$(( (SECONDS_RUN + 35) * 1000 / DMON_MS ))}"   # NCCL 启动比裸 memcpy 慢，多留余量

LOGDIR="$HERE/logs/freq$((1000/DMON_MS))"; mkdir -p "$LOGDIR"
SW_LOG="$LOGDIR/dcgmi_switch_${NAME}.txt"
GPU_LOG="$LOGDIR/dcgmi_gpu_${NAME}.txt"
WL_LOG="$LOGDIR/workload_${NAME}.txt"
NCCL_LOG="$LOGDIR/nccl_debug_${NAME}.txt"
IMG="nvcr.io/nvidia/pytorch:25.12-py3"

echo "[nccl_probe] name=$NAME devices=$CDEV size=${SIZE_MB}MB run=${SECONDS_RUN}s algo=${NCCL_ALGO_ENV:-auto} interval=${DMON_MS}ms count=$COUNT"
echo "[nccl_probe] switch fields=$SW_FIELDS -> $SW_LOG ; gpu fields=$GPU_FIELDS (-i $GPU_ENTITIES) -> $GPU_LOG"
echo "[nccl_probe] NCCL_DEBUG -> $NCCL_LOG"

dcgmi dmon -e "$SW_FIELDS"  -i "$SWITCHES"      -d "$DMON_MS" -c "$COUNT" > "$SW_LOG"  2>&1 &
SW_PID=$!
dcgmi dmon -e "$GPU_FIELDS" -i "$GPU_ENTITIES" -d "$DMON_MS" -c "$COUNT" > "$GPU_LOG" 2>&1 &
GPU_PID=$!
sleep 3

# NCCL 需要够大 shm + ipc=host；NCCL_DEBUG=INFO 抓算法选择（NVLS 证据）。
docker run --rm --gpus "\"device=$CDEV\"" --ipc=host --shm-size=1g \
  -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NVLS,TUNING \
  ${NCCL_ALGO_ENV:+-e NCCL_ALGO=$NCCL_ALGO_ENV} \
  -v "$HERE/workloads":/w "$IMG" \
  python /w/nccl_allreduce.py "$SIZE_MB" "$SECONDS_RUN" > "$WL_LOG" 2>&1 || true

# 拆出 NCCL debug 到单独文件，workload 摘要打屏
grep -E "NCCL INFO|NVLS|Algorithm|Connected|via" "$WL_LOG" > "$NCCL_LOG" 2>/dev/null || true
echo "--- workload summary ---"; grep -E "props|_START|_END|GB/s|ERROR|Error" "$WL_LOG" || true
echo "--- NVLS evidence (NCCL_DEBUG grep) ---"; grep -iE "NVLS|multimem|Connected all|Algorithm" "$WL_LOG" | head -12 || true

echo "[nccl_probe] 等两条 dmon 收尾 ..."
wait "$SW_PID" "$GPU_PID"
echo "[nccl_probe] 完成。switch: $SW_LOG ; gpu: $GPU_LOG ; nccl_debug: $NCCL_LOG"
