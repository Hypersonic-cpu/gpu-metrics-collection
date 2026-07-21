#!/usr/bin/env bash
# DCGM overhead 实验(含极限频率补测)：对一个 workload，在 [无 dcgmi] 与 [不同频率/字段的 dcgmi] 下各跑若干遍，
# workload 用 CUDA event 自计时报告 kernel 时间；对比 median 时间的变化 = dcgmi 的性能开销。
#
# 用法:
#   run_overhead.sh <gemm|mem|comm> [host_gpus] [container_devices]               # 正常 5 条件矩阵
#   EXTREME_MS=10 run_overhead.sh --extreme <gemm|mem|comm> [host_gpus] [cdev]    # 极限频率补测(dev/full @ 最低间隔)
#   gemm/mem 默认单卡物理6；comm 默认物理6,7。结果 tsv 追加到 ./results/raw.tsv(summarize.py 一并汇总)。
#   环境变量: REPEATS(默认2) TARGET_S(默认18) N_CHUNKS(默认360) EXTREME_MS(--extreme 时的最低间隔,默认10)
set -euo pipefail

MODE=normal
if [[ "${1:-}" == "--extreme" ]]; then MODE=extreme; shift; fi

WL="$1"; HOST_GPUS="${2:-}"; CDEV="${3:-}"
REPEATS="${REPEATS:-2}"; TARGET_S="${TARGET_S:-18}"; N_CHUNKS="${N_CHUNKS:-360}"
EXTREME_MS="${EXTREME_MS:-10}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HERE/logs"; RESDIR="$HERE/results"; mkdir -p "$LOGDIR" "$RESDIR"
RAW="$RESDIR/raw.tsv"
IMG="nvcr.io/nvidia/pytorch:25.12-py3"

# 默认选卡（避开被 sglang 占用的 GPU0）
if [[ "$WL" == "comm" ]]; then HOST_GPUS="${HOST_GPUS:-6,7}"; CDEV="${CDEV:-6,7}";
else HOST_GPUS="${HOST_GPUS:-6}"; CDEV="${CDEV:-6}"; fi

# 字段集
F_DEV="449,252,250"                                                # device-only, 最轻
F_PASS1="204,1005,1002,1009,1010,449,1011,1012"                    # pass1 混合(含 profiling)
F_FULL="1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,449,204,252,250"  # 重(逼近 multiplexing)

# 条件矩阵: name interval_ms fields   (baseline 用 "-" 表示不开 dcgmi)
if [[ "$MODE" == "extreme" ]]; then
  # 极限：间隔拉到 dmon 能接受的最低值(EXTREME_MS)。dev-only 不受 100ms profiling 下限约束(真·高频)，
  # full 是狂刷 host 查询的最坏情况。含一次新 baseline 便于公平对比。
  CONDS=(
    "baseline - -"
    "dev@${EXTREME_MS} $EXTREME_MS $F_DEV"
    "full@${EXTREME_MS} $EXTREME_MS $F_FULL"
  )
else
  CONDS=(
    "baseline - -"
    "dev@100 100 $F_DEV"
    "pass1@1000 1000 $F_PASS1"
    "pass1@100 100 $F_PASS1"
    "full@100 100 $F_FULL"
  )
fi

# workload 容器命令
run_workload() {  # $1=输出日志
  local out="$1"
  if [[ "$WL" == "gemm" ]]; then
    docker run --rm --gpus "\"device=$CDEV\"" -v "$HERE/workloads":/w "$IMG" \
      python /w/gemm.py 8192 "$TARGET_S" > "$out" 2>&1
  elif [[ "$WL" == "mem" ]]; then
    docker run --rm --gpus "\"device=$CDEV\"" -v "$HERE/workloads":/w "$IMG" \
      python /w/mem_bound.py 512 "$TARGET_S" > "$out" 2>&1
  elif [[ "$WL" == "comm" ]]; then
    docker run --rm --gpus "\"device=$CDEV\"" --ipc=host -v "$HERE/workloads":/w "$IMG" \
      torchrun --nproc_per_node=2 /w/comm_nccl.py 256 "$N_CHUNKS" > "$out" 2>&1
  else echo "unknown workload $WL" >&2; exit 2; fi
}

pkill -f "dcgmi dmon" 2>/dev/null || true   # 清理可能的残留
echo "[overhead:$MODE] workload=$WL host_gpus=$HOST_GPUS cdev=$CDEV repeats=$REPEATS target_s=$TARGET_S n_chunks=$N_CHUNKS extreme_ms=$EXTREME_MS"

for rep in $(seq 1 "$REPEATS"); do
  for cond in "${CONDS[@]}"; do
    read -r cname civ cfields <<< "$cond"
    wlog="$LOGDIR/wl_${WL}_${cname//[@]/_}_rep${rep}.txt"
    dmon_pid=""
    if [[ "$cname" != "baseline" ]]; then
      dlog="$LOGDIR/dmon_${WL}_${cname//[@]/_}_rep${rep}.txt"
      dcgmi dmon -e "$cfields" -i "$HOST_GPUS" -d "$civ" > "$dlog" 2>&1 &
      dmon_pid=$!
    fi
    run_workload "$wlog" || echo "  [warn] workload run failed, see $wlog"
    [[ -n "$dmon_pid" ]] && { kill "$dmon_pid" 2>/dev/null || true; wait "$dmon_pid" 2>/dev/null || true; }
    line=$(grep '^RESULT' "$wlog" | tail -1 || true)
    if [[ -z "$line" ]]; then line="RESULT ERROR(see $wlog)"; fi
    printf '%s\t%s\t%s\t%s\t%s\n' "$WL" "$cname" "$civ" "$rep" "$line" >> "$RAW"
    echo "  rep$rep $cname -> ${line#RESULT }"
  done
done
echo "[overhead:$MODE] done. raw -> $RAW"
