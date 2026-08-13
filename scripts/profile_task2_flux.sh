#!/usr/bin/env bash
set -euo pipefail

# Flux Pass-2-only profiler:
#   1) Acquire Nsight Systems GPU HW counters FIRST.
#   2) Launch Flux normally on the requested GPUs.
#   3) Keep collecting HBM/NVLink/PCIe/SM time-series metrics.
#   4) Stop Nsight, generate report.nsys-rep, diagnose, and export post_metrics.csv.
#
# Default:
#   workload GPUs : 2,3,4,5 (TP4)
#   metric GPU    : 2 @ 200 kHz
#   workload      : AG+GEMM M=8 N=5120 K=2048
#
# Usage:
#   ./profile_flux_task2_only.sh [M N K]
#
# Examples:
#   ./profile_flux_task2_only.sh 8 5120 2048
#   METRIC_GPUS=2 METRIC_FREQ=100000 ./profile_flux_task2_only.sh 8 5120 2048
#   GPU_LIST=2,3,4,5 NPROC=4 METRIC_GPUS=2 HW_WINDOW=5 \
#     ./profile_flux_task2_only.sh 4096 5120 2048

FLUX_DIR="${FLUX_DIR:-$HOME/Repos/flux}"
METRIC_DIR="${METRIC_DIR:-$HOME/Repos/metric_collection}"
NSYS_TOOL="${NSYS_TOOL:-$METRIC_DIR/tool/nsys/nsys-tool}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/.venvs/flux/bin/activate}"

GPU_LIST="${GPU_LIST:-2,3,4,5}"
NPROC="${NPROC:-4}"

# HW counters: start with ONE representative GPU for 200 kHz debugging.
METRIC_GPUS="${METRIC_GPUS:-2}"
METRIC_FREQ="${METRIC_FREQ:-200000}"

DTYPE="${DTYPE:-bfloat16}"
RING_MODE="${RING_MODE:-ring1d}"
WARMUP="${WARMUP:-100}"

# Keep the workload alive long enough for the metric window.
HW_ITERS="${HW_ITERS:-100000}"

# Counters start before Flux. We intentionally include startup/warmup in the
# report, then keep collecting for HW_WINDOW seconds after HW_INIT_WAIT.
HW_INIT_WAIT="${HW_INIT_WAIT:-3}"
HW_WINDOW="${HW_WINDOW:-5}"

FLUX_M="${1:-8}"
FLUX_N="${2:-5120}"
FLUX_K="${3:-2048}"

TAG="${TAG:-flux-ag-m${FLUX_M}-n${FLUX_N}-k${FLUX_K}-tp${NPROC}}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$METRIC_DIR/runs/task2_${TAG}__${STAMP}}"
HW_SESSION="flux_hw_${$}_${STAMP//-/}"
HW_LOG="$RUN_ROOT/${TAG}-hw-workload.log"
MARKS="$RUN_ROOT/phase_marks.tsv"

[[ -x "$NSYS_TOOL" ]] || {
  echo "ERROR: nsys-tool not executable: $NSYS_TOOL" >&2
  exit 2
}
[[ -f "$VENV_ACTIVATE" ]] || {
  echo "ERROR: Flux venv not found: $VENV_ACTIVATE" >&2
  exit 2
}

mkdir -p "$RUN_ROOT"
cd "$METRIC_DIR"
export PATH="$METRIC_DIR/tool/nsys:$PATH"

export FLUX_DIR VENV_ACTIVATE HW_ITERS
export FLUX_M FLUX_N FLUX_K GPU_LIST NPROC DTYPE RING_MODE WARMUP

run_flux() {
  cd "$FLUX_DIR"
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"

  export OMP_NUM_THREADS=1
  export NVSHMEM_BOOTSTRAP=UID
  export NVSHMEM_DISABLE_CUDA_VMM=1
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export CUDA_MODULE_LOADING=LAZY
  export BYTED_TORCH_BYTECCL=O0
  export NCCL_IB_TIMEOUT=23
  export NCCL_IB_GID_INDEX=3
  export NVSHMEM_IB_GID_INDEX=3
  export NVSHMEM_HCA_LIST=mlx5_bond_0:1

  exec env CUDA_VISIBLE_DEVICES="$GPU_LIST" \
    torchrun --standalone --nproc_per_node="$NPROC" \
      test/python/ag_gemm/test_ag_kernel.py \
      "$FLUX_M" "$FLUX_N" "$FLUX_K" \
      --dtype="$DTYPE" \
      --ring_mode="$RING_MODE" \
      --warmup "$WARMUP" \
      --iters "$HW_ITERS"
}

HW_PID=""
COLLECTOR_STARTED=0
COLLECTOR_STOPPED=0

mark() {
  printf '%s\t%s\n' "$(date +%s.%3N)" "$1" >> "$MARKS"
}

kill_workload() {
  if [[ -n "${HW_PID:-}" ]] && kill -0 "$HW_PID" 2>/dev/null; then
    kill -INT -- "-$HW_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$HW_PID" 2>/dev/null || break
      sleep 0.2
    done
    kill -TERM -- "-$HW_PID" 2>/dev/null || true
    sleep 0.2
    kill -KILL -- "-$HW_PID" 2>/dev/null || true
    wait "$HW_PID" 2>/dev/null || true
  fi
  HW_PID=""
}

stop_collector() {
  if [[ "$COLLECTOR_STARTED" == 1 && "$COLLECTOR_STOPPED" == 0 ]]; then
    set +e
    "$NSYS_TOOL" stop \
      -o "$RUN_ROOT" \
      --session "$HW_SESSION"
    local rc=$?
    set -e
    COLLECTOR_STOPPED=1
    return "$rc"
  fi
}

cleanup() {
  local rc=$?
  set +e
  if [[ "$COLLECTOR_STARTED" == 1 && "$COLLECTOR_STOPPED" == 0 ]]; then
    echo "[cleanup] stopping HW collector so a partial report can be written..."
    stop_collector >/dev/null 2>&1 || true
  fi
  kill_workload
  exit "$rc"
}
trap cleanup EXIT INT TERM

latest_hw_run() {
  find "$RUN_ROOT" -maxdepth 1 -type d \
    -name "nsys_cuda_iface_full_${TAG}-iface-hw__*" \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-
}

echo
echo "=== Flux task 2/2 only: HW time-series ==="
echo "Shape            : M=$FLUX_M N=$FLUX_N K=$FLUX_K"
echo "Workload GPUs    : $GPU_LIST  (nproc=$NPROC)"
echo "HW metric GPUs   : $METRIC_GPUS @ $METRIC_FREQ Hz"
echo "Warmup / iters   : $WARMUP / $HW_ITERS"
echo "Init wait        : ${HW_INIT_WAIT}s"
echo "Stable window    : ${HW_WINDOW}s"
echo "Output root      : $RUN_ROOT"
echo

# IMPORTANT:
# Acquire the periodic sampler BEFORE Flux starts. If this fails with
# "Already under profiling", we do not launch the workload at all.
echo "=== 1/4 Acquire GPU HW counters FIRST ==="
set +e
"$NSYS_TOOL" start \
  -o "$RUN_ROOT" \
  --session "$HW_SESSION" \
  -g cuda_iface_full \
  --trace none \
  -i "$METRIC_GPUS" \
  --freq "$METRIC_FREQ" \
  --name "${TAG}-iface-hw" \
  --note "HW-only Flux pass; workload GPUs=$GPU_LIST; metric GPUs=$METRIC_GPUS; M=$FLUX_M N=$FLUX_N K=$FLUX_K"
START_RC=$?
set -e

if [[ "$START_RC" -ne 0 ]]; then
  echo
  echo "ERROR: Could not acquire HW counters."
  echo "The Flux workload was NOT started."
  echo "If the message says 'Already under profiling', another profiler is"
  echo "currently holding a required profiling resource."
  exit "$START_RC"
fi

COLLECTOR_STARTED=1
mark "hw_counter_acquired"

echo
echo "=== 2/4 Launch Flux normally ==="
setsid bash -c "$(declare -f run_flux); run_flux" >"$HW_LOG" 2>&1 &
HW_PID=$!
mark "workload_launch"
echo "[task2] Flux pid/pgid=$HW_PID"
echo "[task2] workload log=$HW_LOG"

# Give torchrun/NCCL/NVSHMEM/warmup time to settle while counters are already on.
sleep "$HW_INIT_WAIT"

if ! kill -0 "$HW_PID" 2>/dev/null; then
  echo "ERROR: Flux ended during initialization/warmup." >&2
  tail -120 "$HW_LOG" >&2 || true
  stop_collector || true
  COLLECTOR_STOPPED=1
  exit 3
fi

mark "analysis_window_begin"
echo
echo "=== 3/4 Collect steady-state HW metrics for ${HW_WINDOW}s ==="
sleep "$HW_WINDOW"
mark "analysis_window_end"

echo
echo "=== 4/4 Stop collector and write .nsys-rep ==="
stop_collector
STOP_RC=$?
COLLECTOR_STOPPED=1

# The metric report is complete; the long-running workload is no longer needed.
kill_workload
trap - EXIT INT TERM

HW_RUN="$(latest_hw_run)"
if [[ -z "$HW_RUN" || ! -f "$HW_RUN/report.nsys-rep" ]]; then
  echo "ERROR: No HW report found under $RUN_ROOT" >&2
  echo "Workload log: $HW_LOG" >&2
  exit 4
fi

echo
echo "[task2] report:"
echo "  $HW_RUN/report.nsys-rep"
echo "[task2] phase marks:"
echo "  $MARKS"

echo
echo "=== Diagnose ==="
"$NSYS_TOOL" diagnose "$HW_RUN" || true

echo
echo "=== Export HW time-series ==="
python3 "$METRIC_DIR/tool/nsys/post_export.py" "$HW_RUN" || true
if [[ -f "$HW_RUN/post_metrics.csv" ]]; then
  echo "[task2] time series:"
  echo "  $HW_RUN/post_metrics.csv"
else
  echo "[task2] WARNING: post_metrics.csv was not generated."
fi

echo
echo "Done."
echo "The .nsys-rep is the raw Pass-2 HW-metric report."
echo "post_metrics.csv is the convenient time-series export."
exit "$STOP_RC"
