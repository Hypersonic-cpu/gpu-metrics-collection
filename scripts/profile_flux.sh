#!/usr/bin/env bash
set -euo pipefail

# Flux two-pass Nsight Systems profiler
#
# Pass 1: CUDA + NVTX only, with NO CPU sampling and NO GPU HW counters.
#         -> kernel start/end, streams, overlap, CUDA/NVTX timeline.
# Pass 2: acquire system-scope GPU HW metrics, then start Flux normally while
#         the collector is active, with CUDA tracing disabled.
#         -> HBM/NVLink/PCIe/SM time-series metrics.
#
# Default workload:
#   Qwen3-30B AG+GEMM, M=8 N=5120 K=2048
#   Flux on physical GPUs 2,3,4,5 (TP4)
#   HW metrics sampled only on physical GPU 2 @ 200 kHz
#
# Usage:
#   ./scripts/profile_flux.sh [M N K]
#
# Examples:
#   ./scripts/profile_flux.sh 8 5120 2048
#   METRIC_GPUS=2 METRIC_FREQ=100000 ./scripts/profile_flux.sh 8 5120 2048
#   GPU_LIST=2,3,4,5 NPROC=4 METRIC_GPUS=2,3 METRIC_FREQ=50000 \
#     ./scripts/profile_flux.sh 4096 5120 2048

# -----------------------------------------------------------------------------
# Paths / fixed defaults
# -----------------------------------------------------------------------------
FLUX_DIR="${FLUX_DIR:-$HOME/Repos/flux}"
METRIC_DIR="${METRIC_DIR:-$HOME/Repos/metric_collection}"
NSYS_TOOL="${NSYS_TOOL:-$METRIC_DIR/tool/nsys/nsys-tool}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/.venvs/flux/bin/activate}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
METRIC_GPUS="${METRIC_GPUS:-0}"
METRIC_FREQ="${METRIC_FREQ:-100000}"

DTYPE="${DTYPE:-bfloat16}"
RING_MODE="${RING_MODE:-ring1d}"
WARMUP="${WARMUP:-20}"

# These are deliberately large for the M=8 decode case so the process stays
# alive while nsys creates/opens the capture window. Reduce for large shapes.
TRACE_ITERS="${TRACE_ITERS:-100000}"
HW_ITERS="${HW_ITERS:-100000}"

TRACE_PREWAIT="${TRACE_PREWAIT:-1}"
TRACE_WINDOW="${TRACE_WINDOW:-3}"
HW_PREWAIT="${HW_PREWAIT:-2}"
HW_WINDOW="${HW_WINDOW:-1}"
INTERPASS_WAIT="${INTERPASS_WAIT:-2}"

# -----------------------------------------------------------------------------
# Internal child mode.
# IMPORTANT: handle this BEFORE parsing M/N/K from positional arguments.
# The parent exports FLUX_M/N/K; the child receives only __run_flux <iters>.
# This fixes the old bug where "__run_flux" accidentally became M.
# -----------------------------------------------------------------------------
run_flux() {
    local iters="$1"

    : "${FLUX_M:?FLUX_M is not set}"
    : "${FLUX_N:?FLUX_N is not set}"
    : "${FLUX_K:?FLUX_K is not set}"

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
        --iters "$iters"
}

if [[ "${1:-}" == "__run_flux" ]]; then
    run_flux "${2:?missing iteration count}"
    exit $?
fi

# -----------------------------------------------------------------------------
# User-facing positional arguments
# -----------------------------------------------------------------------------
FLUX_M="${1:-8}"
FLUX_N="${2:-5120}"
FLUX_K="${3:-2048}"

export GPU_LIST NPROC FLUX_M FLUX_N FLUX_K DTYPE RING_MODE WARMUP

TAG="${TAG:-flux-ag-m${FLUX_M}-n${FLUX_N}-k${FLUX_K}-tp${NPROC}}"
SELF="$(readlink -f "$0")"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$METRIC_DIR/runs/two_pass_${TAG}__${STAMP}}"
TRACE_SESSION="flux_trace_${$}_${STAMP//-/}"
HW_SESSION="flux_hw_${$}_${STAMP//-/}"

[[ -x "$NSYS_TOOL" ]] || { echo "ERROR: nsys-tool not executable: $NSYS_TOOL" >&2; exit 2; }
[[ -f "$VENV_ACTIVATE" ]] || { echo "ERROR: Flux venv not found: $VENV_ACTIVATE" >&2; exit 2; }
mkdir -p "$RUN_ROOT"

cd "$METRIC_DIR"
export PATH="$METRIC_DIR/tool/nsys:$PATH"

# -----------------------------------------------------------------------------
# Create a true CUDA-only group if it does not already exist.
# Existing cuda_lite also samples CPU process-tree data; on machines with
# perf_event_paranoid=4 that produces the CPU profiling errors seen earlier.
# -----------------------------------------------------------------------------
TRACE_GROUP="cuda_trace_only"
TRACE_GROUP_FILE="$METRIC_DIR/tool/nsys/groups/${TRACE_GROUP}.conf"
if [[ ! -f "$TRACE_GROUP_FILE" ]]; then
    cat > "$TRACE_GROUP_FILE" <<'GROUP_EOF'
# CUDA/NVTX only: no CPU sampling, no GPU HW counters.
trace            = cuda,nvtx
cuda_graph_trace = node
sample           = none
cpuctxsw         = none
gpu_metrics_set  = none
gpu_metrics_freq = 10000
GROUP_EOF
    echo "[setup] created $TRACE_GROUP_FILE"
fi

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
session_exists() {
    local sess="$1"
    "$NSYS_TOOL" sessions 2>/dev/null | grep -Fq "$sess"
}

wait_session_gone() {
    local sess="$1" timeout_s="${2:-10}"
    local n
    for ((n=0; n<timeout_s*2; n++)); do
        if ! session_exists "$sess"; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

wait_pid_gone() {
    local pid="$1" timeout_s="${2:-30}"
    local n stat
    [[ -n "$pid" ]] || return 0
    for ((n=0; n<timeout_s*2; n++)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
        stat="$(ps -o stat= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
        [[ "$stat" == Z* ]] && return 0
        sleep 0.5
    done
    return 1
}

cleanup_session() {
    local sess="$1"
    local launch_pid=""
    local state_file="$RUN_ROOT/.nsys-tool.state"

    # `nsys shutdown` removes the session before the application launched by
    # `nsys launch` has necessarily exited.  Keep the launch PID before
    # shutdown so a following capture cannot race the old CUDA process or its
    # profiling teardown.
    if [[ -f "$state_file" ]]; then
        launch_pid="$(sed -n -E 's/^launch_pid=(.*)$/\1/p' "$state_file" | head -1)"
    fi

    # First ask for a clean shutdown; if a session remains, cancel it.
    "$NSYS_TOOL" shutdown -o "$RUN_ROOT" --session "$sess" >/dev/null 2>&1 || true
    if ! wait_session_gone "$sess" 20; then
        "$NSYS_TOOL" cancel -o "$RUN_ROOT" --session "$sess" >/dev/null 2>&1 || true
        wait_session_gone "$sess" 20 || true
    fi
    if [[ -n "$launch_pid" ]] && ! wait_pid_gone "$launch_pid" 30; then
        echo "[cleanup] WARNING: nsys launch pid $launch_pid is still alive after shutdown" >&2
    fi
    rm -f "$RUN_ROOT/.nsys-tool.state"
}

latest_run() {
    local pattern="$1"
    find "$RUN_ROOT" -maxdepth 1 -type d -name "$pattern" -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | head -1 | cut -d' ' -f2-
}

HW_PID=""
HW_COLLECTOR_STARTED=0
HW_COLLECTOR_STOPPED=0
HW_CAPTURE_LOG=""

stop_hw_collector() {
    [[ "$HW_COLLECTOR_STARTED" == 1 && "$HW_COLLECTOR_STOPPED" == 0 ]] || return 0
    local log_file="${HW_CAPTURE_LOG:-/dev/null}"
    local rc
    set +e
    "$NSYS_TOOL" stop \
      -o "$RUN_ROOT" \
      --session "$HW_SESSION" 2>&1 | tee -a "$log_file"
    rc=${PIPESTATUS[0]}
    set -e
    HW_COLLECTOR_STOPPED=1
    return "$rc"
}

cleanup_hw_workload() {
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

cleanup_all() {
    local rc=$?
    set +e
    if [[ "$HW_COLLECTOR_STARTED" == 1 && "$HW_COLLECTOR_STOPPED" == 0 ]]; then
        echo "[cleanup] stopping HW collector so a partial report can be written..."
        stop_hw_collector || true
    fi
    cleanup_hw_workload
    exit "$rc"
}
trap cleanup_all EXIT INT TERM

printf '\n=== Configuration ===\n'
printf 'Flux shape      : M=%s N=%s K=%s\n' "$FLUX_M" "$FLUX_N" "$FLUX_K"
printf 'Workload GPUs   : %s  (nproc=%s)\n' "$GPU_LIST" "$NPROC"
printf 'HW metric GPUs  : %s @ %s Hz\n' "$METRIC_GPUS" "$METRIC_FREQ"
printf 'Pass1 window    : %ss\n' "$TRACE_WINDOW"
printf 'Pass2 window    : %ss\n' "$HW_WINDOW"
printf 'Output root     : %s\n\n' "$RUN_ROOT"

# =============================================================================
# PASS 1: CUDA/NVTX timeline only
# =============================================================================
echo "=== PASS 1/2: CUDA/NVTX timeline only (NO CPU sampling, NO HW counters) ==="

# Remove only a stale session with our unique name (normally none).
cleanup_session "$TRACE_SESSION"

"$NSYS_TOOL" launch \
  -o "$RUN_ROOT" \
  --session "$TRACE_SESSION" \
  -g "$TRACE_GROUP" \
  -i none \
  --name "${TAG}-timeline-launch" \
  -- "$SELF" __run_flux "$TRACE_ITERS"

sleep "$TRACE_PREWAIT"

"$NSYS_TOOL" gen \
  -o "$RUN_ROOT" \
  --session "$TRACE_SESSION" \
  -t "$TRACE_WINDOW" \
  --name "${TAG}-timeline"

# gen intentionally leaves a launch session alive. Explicitly terminate it,
# then WAIT until Nsight reports that the session is gone before Pass 2.
cleanup_session "$TRACE_SESSION"

TRACE_RUN="$(latest_run "nsys_${TRACE_GROUP}_${TAG}-timeline__*")"
if [[ -n "$TRACE_RUN" && -f "$TRACE_RUN/report.nsys-rep" ]]; then
    echo "[pass1] report: $TRACE_RUN/report.nsys-rep"
    "$NSYS_TOOL" diagnose "$TRACE_RUN" || true
else
    echo "[pass1] WARNING: could not locate the timeline report automatically."
fi

# Give Nsight's daemon/profiling context a short grace period to release all
# resources before opening the GPU periodic-sampler pass.
echo "[between] waiting ${INTERPASS_WAIT}s for Nsight resources to be released..."
sleep "$INTERPASS_WAIT"

# Best-effort extra cleanup of our private state. This does not touch unrelated
# users/sessions because both pass session names are unique to this invocation.
rm -f "$RUN_ROOT/.nsys-tool.state"

# =============================================================================
# PASS 2: normal Flux workload + system-scope HW metrics
# =============================================================================
echo
echo "=== PASS 2/2: HW counters while Flux runs normally ==="

HW_LOG="$RUN_ROOT/${TAG}-hw-workload.log"
HW_CAPTURE_LOG="$RUN_ROOT/${TAG}-hw-capture.log"

# Acquire the system-scope GPU metrics collector before starting Flux.  The
# collector does not need an application to exist, and this avoids racing the
# CUDA/NVTX session from Pass 1 while it is still being torn down.
cleanup_session "$HW_SESSION"

HW_RC=1
for attempt in 1 2; do
    echo "[pass2] HW capture attempt $attempt/2 (collector first)"
    set +e
    "$NSYS_TOOL" start \
      -o "$RUN_ROOT" \
      --session "$HW_SESSION" \
      -g cuda_iface_full \
      --trace none \
      -i "$METRIC_GPUS" \
      --freq "$METRIC_FREQ" \
      --name "${TAG}-iface-hw" \
      --note "Flux HW-only pass; workload GPUs=$GPU_LIST; metric GPUs=$METRIC_GPUS; M=$FLUX_M N=$FLUX_N K=$FLUX_K" \
      2>&1 | tee -a "$HW_CAPTURE_LOG"
    HW_RC=${PIPESTATUS[0]}
    set -e

    if [[ "$HW_RC" -eq 0 ]]; then
        HW_COLLECTOR_STARTED=1
        break
    fi

    echo "[pass2] capture failed; see $HW_CAPTURE_LOG" >&2
    cleanup_session "$HW_SESSION"
    sleep 2
done

if [[ "$HW_COLLECTOR_STARTED" -ne 1 ]]; then
    echo "ERROR: Could not acquire HW counters; Flux was not started." >&2
    echo "Collector log: $HW_CAPTURE_LOG" >&2
    exit "$HW_RC"
fi

# Start Flux outside Nsight in its own process group.  The collector is already
# active and captures startup, warmup, and the steady-state window.
setsid "$SELF" __run_flux "$HW_ITERS" >"$HW_LOG" 2>&1 &
HW_PID=$!
echo "[pass2] Flux workload pid/pgid=$HW_PID; log=$HW_LOG"

sleep "$HW_PREWAIT"

if ! kill -0 "$HW_PID" 2>/dev/null; then
    echo "ERROR: Flux workload ended before HW collection began." >&2
    echo "Increase HW_ITERS or reduce HW_PREWAIT. Last log lines:" >&2
    tail -100 "$HW_LOG" >&2 || true
    exit 3
fi

# Stop the collector while Flux is still running so the report includes the
# requested steady-state tail, then stop torchrun and all ranks.
sleep "$HW_WINDOW"
echo "[pass2] stopping HW collector"
set +e
stop_hw_collector
HW_RC=$?
set -e
cleanup_hw_workload

HW_RUN="$(latest_run "nsys_cuda_iface_full_${TAG}-iface-hw__*")"
if [[ -n "$HW_RUN" && -f "$HW_RUN/report.nsys-rep" ]]; then
    echo "[pass2] report: $HW_RUN/report.nsys-rep"
    # The report check is critical: it is the only place that discovers the
    # overflow/error state that nsys leaves inside report.nsys-rep.  Exporting
    # the CSV only reads the finalized report, so it cannot change raw data or
    # the diagnostic result; leave it running after the shell returns.
    DIAG_RC=0
    "$NSYS_TOOL" diagnose "$HW_RUN" || DIAG_RC=$?

    POST_LOG="$HW_RUN/post_export.log"
    setsid nohup python3 "$METRIC_DIR/tool/nsys/post_export.py" "$HW_RUN" \
        >"$POST_LOG" 2>&1 < /dev/null &
    POST_PID=$!
    echo "[pass2] post_export started in background (pid=$POST_PID)"
    echo "[pass2] post-export log: $POST_LOG"
    if [[ "$DIAG_RC" -ne 0 && "$HW_RC" -eq 0 ]]; then
        HW_RC="$DIAG_RC"
    fi
else
    echo "[pass2] WARNING: no HW report was found."
    echo "[pass2] workload log: $HW_LOG"
    echo "[pass2] collector log: $HW_CAPTURE_LOG"
    [[ "$HW_RC" -eq 0 ]] && HW_RC=4
fi

# Do not leave our own nsys state/session behind.
cleanup_session "$HW_SESSION"

echo
echo "=== Done ==="
echo "Pass 1 .nsys-rep: CUDA/NVTX kernel timeline only."
echo "Pass 2 .nsys-rep: HBM/NVLink/PCIe/SM HW-metric time series only."
echo "The reports live in different directories under:"
echo "  $RUN_ROOT"

exit "$HW_RC"
