#!/usr/bin/env bash
set -euo pipefail

# Flux four-pass profiler. This file is a copied/extended variant of
# scripts/profile_flux.sh; the original two-pass entry point is intentionally
# left unchanged.
#
# Pass 1: CUDA + NVTX timeline only, with no CPU sampling or GPU HW counters.
# Pass 2: NSYS interface metrics + GPC/SYS clocks, one GPU at 100 kHz.
# Pass 3: the same NSYS metric set on all eight GPUs at 25 kHz.
# Pass 4: long-window DCGM absolute NVLink/HBM bandwidth collection.
#
# Passes 2--4 each start Flux normally while their own collector is active.
# They are sequential because DCGM profiling and NSYS GPU Metrics cannot share
# the same hardware profiling resource at the same time.
#
# Default workload:
#   Qwen3-30B AG+GEMM, M=8 N=5120 K=2048
#   Flux on physical GPUs 0,1,2,3,4,5,6,7 (TP8)
#
# Usage:
#   ./scripts/profile_flux_4pass.sh [M N K]
#   ./scripts/profile_flux_4pass.sh -- torchrun ... --iters __PROFILE_ITERS__
#
# Important: this script is deliberately not run as part of repository checks.
# Start it only after all eight workload GPUs and the DCGM profiling resource
# are free.

# -----------------------------------------------------------------------------
# Paths and pass defaults
# -----------------------------------------------------------------------------
FLUX_DIR="${FLUX_DIR:-$HOME/Repos/flux}"
METRIC_DIR="${METRIC_DIR:-$HOME/Repos/metric_collection}"
NSYS_TOOL="${NSYS_TOOL:-$METRIC_DIR/tool/nsys/nsys-tool}"
PROFILE_SH="${PROFILE_SH:-$METRIC_DIR/tool/profile.sh}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/.venvs/flux/bin/activate}"

GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"

NSYS_PASS2_GPUS="${NSYS_PASS2_GPUS:-0}"
NSYS_PASS2_FREQ="${NSYS_PASS2_FREQ:-100000}"
NSYS_PASS3_GPUS="${NSYS_PASS3_GPUS:-0,1,2,3,4,5,6,7}"
NSYS_PASS3_FREQ="${NSYS_PASS3_FREQ:-25000}"
NSYS_GROUP="${NSYS_GROUP:-cuda_iface_clock}"

DCGM_GPUS="${DCGM_GPUS:-$GPU_LIST}"
DCGM_FIELDS="${DCGM_FIELDS:-flux_absolute_bw}"
DCGM_INTERVAL_MS="${DCGM_INTERVAL_MS:-100}"
DCGM_WINDOW="${DCGM_WINDOW:-60}"
DCGM_HBM_PEAK_GBPS="${DCGM_HBM_PEAK_GBPS:-3352.32}"
START_POSTPROCESS="${START_POSTPROCESS:-1}"

DTYPE="${DTYPE:-bfloat16}"
RING_MODE="${RING_MODE:-ring1d}"
WARMUP="${WARMUP:-20}"

# Keep each workload alive while Nsight opens its system-scope sampler. The
# long DCGM pass uses a larger iteration count because its window is 60 s by
# default; the run is still stopped by this script after the requested window.
TRACE_ITERS="${TRACE_ITERS:-100000}"
NSYS_ITERS="${NSYS_ITERS:-100000}"
DCGM_ITERS="${DCGM_ITERS:-1000000}"

RC_PROFILING_OVERFLOW=20
RC_PROFILING_OTHERERR=21
RC_WORKLOAD_FAIL=22

TRACE_PREWAIT="${TRACE_PREWAIT:-1}"
TRACE_WINDOW="${TRACE_WINDOW:-3}"
PASS2_PREWAIT="${PASS2_PREWAIT:-2}"
PASS2_WINDOW="${PASS2_WINDOW:-3}"
PASS3_PREWAIT="${PASS3_PREWAIT:-2}"
PASS3_WINDOW="${PASS3_WINDOW:-3}"
DCGM_START_WAIT="${DCGM_START_WAIT:-2}"
INTERPASS_WAIT="${INTERPASS_WAIT:-3}"
ROI_TIMEOUT="${ROI_TIMEOUT:-300}"
ROI_COLLECTOR_LEAD="${ROI_COLLECTOR_LEAD:-0.2}"

# -----------------------------------------------------------------------------
# Internal child mode
# -----------------------------------------------------------------------------
run_workload() {
    local iters="$1"
    shift
    local raw_arg
    local command=()

    [[ "$#" -gt 0 ]] || { echo "ERROR: empty workload command" >&2; return 2; }
    for raw_arg in "$@"; do
        if [[ "$raw_arg" == "__PROFILE_ITERS__" ]]; then
            command+=("$iters")
        else
            command+=("$raw_arg")
        fi
    done

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

    exec env CUDA_VISIBLE_DEVICES="$GPU_LIST" "${command[@]}"
}

# Handle this before parsing the user-facing command. The child receives the
# pass-specific iteration count followed by the complete workload argv.
if [[ "${1:-}" == "__run_workload" ]]; then
    shift
    run_workload "${1:?missing iteration count}" "${@:2}"
    exit $?
fi

# -----------------------------------------------------------------------------
# User-facing arguments and run paths
# -----------------------------------------------------------------------------
WORKLOAD_CMD=()
if [[ "${1:-}" == "--" ]]; then
    shift
    [[ "$#" -gt 0 ]] || { echo "ERROR: -- requires a workload command" >&2; exit 2; }
    WORKLOAD_CMD=("$@")
    FLUX_M="custom"
    FLUX_N="custom"
    FLUX_K="custom"
else
    FLUX_M="${1:-8}"
    FLUX_N="${2:-5120}"
    FLUX_K="${3:-2048}"
    WORKLOAD_CMD=(
        torchrun --standalone "--nproc_per_node=$NPROC"
        test/python/ag_gemm/test_ag_kernel.py
        "$FLUX_M" "$FLUX_N" "$FLUX_K"
        --dtype "$DTYPE"
        --ring_mode "$RING_MODE"
        --warmup "$WARMUP"
        --iters __PROFILE_ITERS__
    )
fi

export GPU_LIST NPROC FLUX_M FLUX_N FLUX_K DTYPE RING_MODE WARMUP

TAG="${TAG:-flux-ag-m${FLUX_M}-n${FLUX_N}-k${FLUX_K}-tp${NPROC}}"
SELF="$(readlink -f "$0")"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$METRIC_DIR/runs/four_pass_${TAG}__${STAMP}}"

TRACE_SESSION="flux_trace_4p_${$}_${STAMP//-/}"
PASS2_SESSION="flux_pass2_${$}_${STAMP//-/}"
PASS3_SESSION="flux_pass3_${$}_${STAMP//-/}"

[[ -x "$NSYS_TOOL" ]] || { echo "ERROR: nsys-tool not executable: $NSYS_TOOL" >&2; exit 2; }
[[ -x "$PROFILE_SH" ]] || { echo "ERROR: profile.sh not executable: $PROFILE_SH" >&2; exit 2; }
[[ -f "$VENV_ACTIVATE" ]] || { echo "ERROR: Flux venv not found: $VENV_ACTIVATE" >&2; exit 2; }
[[ -f "$METRIC_DIR/tool/nsys/groups/cuda_trace_only.conf" ]] || {
    echo "ERROR: missing CUDA-only group: $METRIC_DIR/tool/nsys/groups/cuda_trace_only.conf" >&2
    exit 2
}
[[ -f "$METRIC_DIR/tool/nsys/groups/${NSYS_GROUP}.conf" ]] || {
    echo "ERROR: missing NSYS group: $METRIC_DIR/tool/nsys/groups/${NSYS_GROUP}.conf" >&2
    exit 2
}
[[ -f "$METRIC_DIR/tool/dcgmi/${DCGM_FIELDS}.txt" ]] || {
    echo "ERROR: missing DCGM field group: $METRIC_DIR/tool/dcgmi/${DCGM_FIELDS}.txt" >&2
    exit 2
}

mkdir -p "$RUN_ROOT"
cd "$METRIC_DIR"
export PATH="$METRIC_DIR/tool/nsys:$PATH"

# -----------------------------------------------------------------------------
# Common session/workload helpers
# -----------------------------------------------------------------------------
session_exists() {
    local sess="$1"
    "$NSYS_TOOL" sessions 2>/dev/null | grep -Fq "$sess"
}

wait_session_gone() {
    local sess="$1" timeout_s="${2:-20}"
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

    if [[ -f "$state_file" ]]; then
        launch_pid="$(sed -n -E 's/^launch_pid=(.*)$/\1/p' "$state_file" | head -1)"
    fi

    "$NSYS_TOOL" shutdown -o "$RUN_ROOT" --session "$sess" >/dev/null 2>&1 || true
    if ! wait_session_gone "$sess" 20; then
        "$NSYS_TOOL" cancel -o "$RUN_ROOT" --session "$sess" >/dev/null 2>&1 || true
        wait_session_gone "$sess" 20 || true
    fi
    if [[ -n "$launch_pid" ]] && ! wait_pid_gone "$launch_pid" 30; then
        echo "[cleanup] WARNING: nsys launch pid $launch_pid is still alive" >&2
    fi
    rm -f "$state_file"
}

latest_run() {
    local pattern="$1"
    find "$RUN_ROOT" -maxdepth 1 -type d -name "$pattern" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -1 | cut -d' ' -f2-
}

latest_dcgm_run() {
    find "$RUN_ROOT/pass4_dcgm" -mindepth 1 -maxdepth 1 -type d \
        -name '20*' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -1 | cut -d' ' -f2-
}

WORKLOAD_PID=""
ACTIVE_SESSION=""
ACTIVE_COLLECTOR_STARTED=0
ACTIVE_COLLECTOR_STOPPED=0
ACTIVE_CAPTURE_LOG=""
DCGM_PID=""
DIAG_PIDS=()
DIAG_PASSES=()
DIAG_FILES=()
DIAG_LOCK="$METRIC_DIR/runs/.nsys-diagnose.lock"

cleanup_workload() {
    if [[ -n "${WORKLOAD_PID:-}" ]] && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
        kill -INT -- "-$WORKLOAD_PID" 2>/dev/null || true
        for _ in $(seq 1 20); do
            kill -0 "$WORKLOAD_PID" 2>/dev/null || break
            sleep 0.2
        done
        kill -TERM -- "-$WORKLOAD_PID" 2>/dev/null || true
        sleep 0.2
        kill -KILL -- "-$WORKLOAD_PID" 2>/dev/null || true
        wait "$WORKLOAD_PID" 2>/dev/null || true
    fi
    WORKLOAD_PID=""
}

workload_is_running() {
    local stat
    [[ -n "${WORKLOAD_PID:-}" ]] || return 1
    kill -0 "$WORKLOAD_PID" 2>/dev/null || return 1
    stat="$(ps -o stat= -p "$WORKLOAD_PID" 2>/dev/null | tr -d '[:space:]')"
    [[ -n "$stat" && "$stat" != Z* ]]
}

prepare_roi() {
    local label="$1"
    export FLUX_PROFILE_ROI=1
    export FLUX_PROFILE_READY_FILE="$RUN_ROOT/.roi-${label}.ready"
    export FLUX_PROFILE_GO_FILE="$RUN_ROOT/.roi-${label}.go"
    export FLUX_PROFILE_ROI_TIMEOUT="$ROI_TIMEOUT"
    rm -f "$FLUX_PROFILE_READY_FILE" "$FLUX_PROFILE_GO_FILE"
}

wait_for_roi() {
    local label="$1"
    local ticks=$((ROI_TIMEOUT * 10))
    local n
    for ((n=0; n<ticks; n++)); do
        if [[ -f "$FLUX_PROFILE_READY_FILE" ]]; then
            echo "[$label] ROI ready: $FLUX_PROFILE_READY_FILE"
            return 0
        fi
        sleep 0.1
    done
    echo "[$label] ERROR: workload did not reach ROI within ${ROI_TIMEOUT}s" >&2
    return 1
}

release_roi() {
    local label="$1"
    : > "$FLUX_PROFILE_GO_FILE"
    echo "[$label] ROI released: $FLUX_PROFILE_GO_FILE"
}

stop_active_collector() {
    [[ "$ACTIVE_COLLECTOR_STARTED" == 1 && "$ACTIVE_COLLECTOR_STOPPED" == 0 ]] || return 0
    local log_file="${ACTIVE_CAPTURE_LOG:-/dev/null}"
    local rc
    if (
        set +e
        "$NSYS_TOOL" stop -o "$RUN_ROOT" --session "$ACTIVE_SESSION" 2>&1 \
            | tee -a "$log_file"
        exit "${PIPESTATUS[0]}"
    ); then
        rc=0
    else
        rc=$?
    fi
    ACTIVE_COLLECTOR_STOPPED=1
    return "$rc"
}

cleanup_all() {
    local rc=$?
    set +e
    if [[ "$ACTIVE_COLLECTOR_STARTED" == 1 && "$ACTIVE_COLLECTOR_STOPPED" == 0 ]]; then
        echo "[cleanup] stopping active NSYS collector..." >&2
        stop_active_collector || true
    fi
    cleanup_workload
    if [[ -n "${DCGM_PID:-}" ]] && kill -0 "$DCGM_PID" 2>/dev/null; then
        kill -INT "$DCGM_PID" 2>/dev/null || true
        wait "$DCGM_PID" 2>/dev/null || true
    fi
    exit "$rc"
}
trap cleanup_all EXIT INT TERM

queue_nsys_diagnostic() {
    local pass_no="$1"
    local report_dir="$2"
    local status_file="$report_dir/profile_diagnostic.rc"
    local diag_log="$report_dir/profile_diagnostic.log"
    local post_log="$report_dir/post_export.log"
    rm -f "$status_file"

    (
        trap - EXIT INT TERM
        set +e
        local_diag_rc=0
        mapped_rc=0
        flock "$DIAG_LOCK" "$NSYS_TOOL" diagnose "$report_dir" || local_diag_rc=$?
        if [[ "$local_diag_rc" -ne 0 ]]; then
            if grep -Eiq 'overflow|溢出' "$report_dir/diagnostics.txt" 2>/dev/null; then
                mapped_rc="$RC_PROFILING_OVERFLOW"
            else
                mapped_rc="$RC_PROFILING_OTHERERR"
            fi
        elif [[ "$START_POSTPROCESS" == 0 ]]; then
            echo "[nsys] post_export deferred (START_POSTPROCESS=0)"
        else
            setsid nohup python3 "$METRIC_DIR/tool/nsys/post_export.py" "$report_dir" \
                >"$post_log" 2>&1 < /dev/null &
            echo "[nsys] post_export started in background (pid=$!)"
            echo "[nsys] post-export log: $post_log"
        fi
        printf '%s\n' "$mapped_rc" > "$status_file"
    ) >"$diag_log" 2>&1 &

    DIAG_PIDS+=("$!")
    DIAG_PASSES+=("$pass_no")
    DIAG_FILES+=("$status_file")
    echo "[pass${pass_no}] diagnostic queued in background (pid=$!, log=$diag_log)"
}

wait_nsys_diagnostics() {
    local i pid pass_no status_file diag_rc
    echo "[diagnostics] waiting for ${#DIAG_PIDS[@]} queued NSYS report check(s)..."
    for i in "${!DIAG_PIDS[@]}"; do
        pid="${DIAG_PIDS[$i]}"
        pass_no="${DIAG_PASSES[$i]}"
        status_file="${DIAG_FILES[$i]}"
        wait "$pid" 2>/dev/null || true
        diag_rc="$RC_PROFILING_OTHERERR"
        [[ -s "$status_file" ]] && read -r diag_rc < "$status_file"
        echo "[diagnostics] pass${pass_no} rc=$diag_rc"
        case "$pass_no" in
            1) if [[ "$PASS1_RC" -eq 0 ]]; then PASS1_RC="$diag_rc"; fi ;;
            2) if [[ "$PASS2_RC" -eq 0 ]]; then PASS2_RC="$diag_rc"; fi ;;
            3) if [[ "$PASS3_RC" -eq 0 ]]; then PASS3_RC="$diag_rc"; fi ;;
        esac
    done
}

# -----------------------------------------------------------------------------
# Pass 1: CUDA/NVTX timeline only
# -----------------------------------------------------------------------------
run_pass1() {
    local trace_group="cuda_trace_only"
    local trace_run=""

    echo
    echo "=== PASS 1/4: CUDA/NVTX timeline only ==="
    cleanup_session "$TRACE_SESSION"
    prepare_roi pass1

    "$NSYS_TOOL" launch \
        -o "$RUN_ROOT" \
        --session "$TRACE_SESSION" \
        -g "$trace_group" \
        -i none \
        --name "${TAG}-timeline-launch" \
        -- "$SELF" __run_workload "$TRACE_ITERS" "${WORKLOAD_CMD[@]}"

    if ! wait_for_roi pass1; then
        cleanup_session "$TRACE_SESSION"
        return "$RC_WORKLOAD_FAIL"
    fi
    if ! "$NSYS_TOOL" start \
        -o "$RUN_ROOT" \
        --session "$TRACE_SESSION" \
        -g "$trace_group" \
        -i none \
        --name "${TAG}-timeline"; then
        cleanup_session "$TRACE_SESSION"
        return "$RC_PROFILING_OTHERERR"
    fi
    sleep "$TRACE_PREWAIT"
    release_roi pass1
    sleep "$TRACE_WINDOW"
    if ! "$NSYS_TOOL" stop -o "$RUN_ROOT" --session "$TRACE_SESSION"; then
        cleanup_session "$TRACE_SESSION"
        return "$RC_PROFILING_OTHERERR"
    fi

    # Stop finalizes the report; shutdown also terminates the launched app so
    # no rank can leak into the next hardware-counter pass.
    cleanup_session "$TRACE_SESSION"
    trace_run="$(latest_run "nsys_${trace_group}_${TAG}-timeline__*")"
    if [[ -n "$trace_run" && -f "$trace_run/report.nsys-rep" ]]; then
        echo "[pass1] report: $trace_run/report.nsys-rep"
        queue_nsys_diagnostic 1 "$trace_run"
    else
        echo "[pass1] ERROR: timeline report was not generated" >&2
        return "$RC_PROFILING_OTHERERR"
    fi
}

# -----------------------------------------------------------------------------
# Pass 2/3: NSYS system-scope GPU metrics
# -----------------------------------------------------------------------------
run_nsys_pass() {
    local pass_no="$1"
    local metric_gpus="$2"
    local metric_freq="$3"
    local window_s="$4"
    local iters="$6"
    local label="$7"
    local session="$8"
    local capture_log="$RUN_ROOT/${TAG}-${label}-capture.log"
    local workload_log="$RUN_ROOT/${TAG}-${label}-workload.log"
    local report_dir=""
    local start_rc=0
    local stop_rc=0
    local workload_failed=0

    echo
    echo "=== PASS ${pass_no}/4: NSYS ${label} ==="
    echo "[pass${pass_no}] workload GPUs=$GPU_LIST; metric GPUs=$metric_gpus @ ${metric_freq}Hz"

    ACTIVE_SESSION="$session"
    ACTIVE_CAPTURE_LOG="$capture_log"
    ACTIVE_COLLECTOR_STARTED=0
    ACTIVE_COLLECTOR_STOPPED=0
    : > "$capture_log"
    cleanup_session "$session"

    prepare_roi "pass${pass_no}"
    setsid "$SELF" __run_workload "$iters" "${WORKLOAD_CMD[@]}" >"$workload_log" 2>&1 &
    WORKLOAD_PID=$!
    echo "[pass${pass_no}] Flux workload pid/pgid=$WORKLOAD_PID"
    echo "[pass${pass_no}] workload log=$workload_log"
    if ! wait_for_roi "pass${pass_no}"; then
        tail -100 "$workload_log" >&2 || true
        cleanup_workload
        cleanup_session "$session"
        ACTIVE_SESSION=""
        return "$RC_WORKLOAD_FAIL"
    fi

    if (
        set +e
        "$NSYS_TOOL" start \
            -o "$RUN_ROOT" \
            --session "$session" \
            -g "$NSYS_GROUP" \
            --trace none \
            -i "$metric_gpus" \
            --freq "$metric_freq" \
            --name "${TAG}-${label}" \
            --note "Flux ${label}; workload GPUs=$GPU_LIST; metric GPUs=$metric_gpus; frequency=${metric_freq}Hz; M=$FLUX_M N=$FLUX_N K=$FLUX_K" \
            2>&1 | tee -a "$capture_log"
        exit "${PIPESTATUS[0]}"
    ); then
        start_rc=0
    else
        start_rc=$?
    fi

    if [[ "$start_rc" -ne 0 ]]; then
        echo "[pass${pass_no}] ERROR: NSYS collector did not start" >&2
        cleanup_workload
        cleanup_session "$session"
        ACTIVE_SESSION=""
        return "$RC_PROFILING_OTHERERR"
    fi
    ACTIVE_COLLECTOR_STARTED=1

    sleep "$ROI_COLLECTOR_LEAD"
    release_roi "pass${pass_no}"

    sleep "$window_s"
    if ! workload_is_running; then
        workload_failed=1
        echo "[pass${pass_no}] ERROR: workload ended during the profiling window" >&2
        tail -100 "$workload_log" >&2 || true
    fi
    echo "[pass${pass_no}] stopping NSYS collector"
    stop_active_collector
    stop_rc=$?
    cleanup_workload

    report_dir="$(latest_run "nsys_${NSYS_GROUP}_${TAG}-${label}__*")"
    if [[ -z "$report_dir" || ! -f "$report_dir/report.nsys-rep" ]]; then
        echo "[pass${pass_no}] ERROR: no NSYS report found" >&2
        echo "[pass${pass_no}] capture log: $capture_log" >&2
        cleanup_session "$session"
        ACTIVE_SESSION=""
        return "$RC_PROFILING_OTHERERR"
    fi

    echo "[pass${pass_no}] report: $report_dir/report.nsys-rep"
    queue_nsys_diagnostic "$pass_no" "$report_dir"
    cleanup_session "$session"
    ACTIVE_SESSION=""
    ACTIVE_COLLECTOR_STARTED=0
    ACTIVE_COLLECTOR_STOPPED=0
    ACTIVE_CAPTURE_LOG=""

    [[ "$stop_rc" -ne 0 ]] && return "$RC_PROFILING_OTHERERR"
    [[ "$workload_failed" -ne 0 ]] && return "$RC_WORKLOAD_FAIL"
    return 0
}

# -----------------------------------------------------------------------------
# Pass 4: long-window DCGM bandwidth
# -----------------------------------------------------------------------------
run_pass4_dcgm() {
    local dcgm_root="$RUN_ROOT/pass4_dcgm"
    local dcgm_log="$RUN_ROOT/${TAG}-pass4-dcgm.log"
    local workload_log="$RUN_ROOT/${TAG}-pass4-dcgm-workload.log"
    local dcgm_run=""
    local collector_rc=0
    local workload_failed=0

    echo
    echo "=== PASS 4/4: DCGM absolute NVLink/HBM bandwidth (${DCGM_WINDOW}s) ==="
    echo "[pass4] workload GPUs=$GPU_LIST; DCGM GPUs=$DCGM_GPUS @ ${DCGM_INTERVAL_MS}ms"
    mkdir -p "$dcgm_root"

    prepare_roi pass4
    setsid "$SELF" __run_workload "$DCGM_ITERS" "${WORKLOAD_CMD[@]}" >"$workload_log" 2>&1 &
    WORKLOAD_PID=$!
    echo "[pass4] Flux workload pid/pgid=$WORKLOAD_PID"
    echo "[pass4] workload log=$workload_log"
    if ! wait_for_roi pass4; then
        tail -100 "$workload_log" >&2 || true
        cleanup_workload
        return "$RC_WORKLOAD_FAIL"
    fi

    # Allocation and warmup have completed. Start DCGM while the workload is
    # held at the ROI barrier, then release only the measured Flux loop.
    set +e
    "$PROFILE_SH" \
        --backend dcgm \
        --gpus "$DCGM_GPUS" \
        --fields "$DCGM_FIELDS" \
        --interval-ms "$DCGM_INTERVAL_MS" \
        --out "$dcgm_root" \
        --duration "$DCGM_WINDOW" \
        --note "Flux Pass 4; absolute NVLink/HBM bandwidth; HBM is dram_active*${DCGM_HBM_PEAK_GBPS}GB/s estimate" \
        >"$dcgm_log" 2>&1 &
    DCGM_PID=$!
    set -e

    local ready=0
    for _ in $(seq 1 60); do
        dcgm_run="$(latest_dcgm_run)"
        if [[ -n "$dcgm_run" && -f "$dcgm_run/dcgm_raw.log" ]]; then
            ready=1
            break
        fi
        if ! kill -0 "$DCGM_PID" 2>/dev/null; then
            break
        fi
        sleep 0.5
    done
    if [[ "$ready" -ne 1 ]]; then
        echo "[pass4] ERROR: DCGM collector did not create a run" >&2
        tail -100 "$dcgm_log" >&2 || true
        wait "$DCGM_PID" 2>/dev/null || true
        DCGM_PID=""
        cleanup_workload
        return "$RC_PROFILING_OTHERERR"
    fi
    sleep "$DCGM_START_WAIT"
    release_roi pass4
    if ! workload_is_running; then
        echo "[pass4] ERROR: Flux ended before the DCGM window" >&2
        tail -100 "$workload_log" >&2 || true
        cleanup_workload
        kill -INT "$DCGM_PID" 2>/dev/null || true
        wait "$DCGM_PID" 2>/dev/null || true
        DCGM_PID=""
        return "$RC_WORKLOAD_FAIL"
    fi

    if wait "$DCGM_PID"; then
        collector_rc=0
    else
        collector_rc=$?
    fi
    DCGM_PID=""
    if ! workload_is_running; then
        workload_failed=1
        echo "[pass4] ERROR: workload ended during the DCGM window" >&2
        tail -100 "$workload_log" >&2 || true
    fi
    cleanup_workload

    dcgm_run="$(latest_dcgm_run)"
    if [[ -z "$dcgm_run" || ! -f "$dcgm_run/dcgm_raw.log" ]]; then
        echo "[pass4] ERROR: no DCGM raw log found under $dcgm_root" >&2
        echo "[pass4] collector log: $dcgm_log" >&2
        return "$RC_PROFILING_OTHERERR"
    fi
    echo "[pass4] DCGM run: $dcgm_run"

    [[ "$collector_rc" -ne 0 ]] && return "$RC_PROFILING_OTHERERR"
    [[ "$workload_failed" -ne 0 ]] && return "$RC_WORKLOAD_FAIL"

    if [[ "$START_POSTPROCESS" == 0 ]]; then
        echo "[pass4] DCGM post-processing deferred (START_POSTPROCESS=0)"
        return 0
    fi

    # Both steps only consume the finalized DCGM log. Keep them off the
    # collector's critical path; the log records any later parse failure.
    local post_log="$dcgm_run/dcgm_postprocess.log"
    setsid nohup bash -c '
        set -euo pipefail
        metric_dir="$1"
        run_dir="$2"
        hbm_peak="$3"
        python3 "$metric_dir/tool/metrics.py" parse "$run_dir"
        python3 "$metric_dir/tool/dcgmi/absolute_bandwidth.py" \
            "$run_dir/metrics.csv" --hbm-peak-gbps "$hbm_peak"
    ' _ "$METRIC_DIR" "$dcgm_run" "$DCGM_HBM_PEAK_GBPS" \
        >"$post_log" 2>&1 < /dev/null &
    local post_pid=$!
    echo "[pass4] DCGM post-processing started in background (pid=$post_pid)"
    echo "[pass4] post-processing log: $post_log"
    return 0
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
printf '\n=== Four-pass configuration ===\n'
printf 'Flux shape       : M=%s N=%s K=%s\n' "$FLUX_M" "$FLUX_N" "$FLUX_K"
printf 'Workload GPUs    : %s  (nproc=%s)\n' "$GPU_LIST" "$NPROC"
printf 'Workload command :'; printf ' %q' "${WORKLOAD_CMD[@]}"; printf '\n'
printf 'Pass 2 NSYS      : %s @ %s Hz, group=%s\n' "$NSYS_PASS2_GPUS" "$NSYS_PASS2_FREQ" "$NSYS_GROUP"
printf 'Pass 3 NSYS      : DISABLED\n'
printf 'Pass 4 DCGM      : %s @ %sms, window=%ss, fields=%s\n' "$DCGM_GPUS" "$DCGM_INTERVAL_MS" "$DCGM_WINDOW" "$DCGM_FIELDS"
printf 'Output root      : %s\n\n' "$RUN_ROOT"

PASS1_RC=0
PASS2_RC=0
PASS3_RC=0
PASS4_RC=0

run_pass1 || PASS1_RC=$?

echo "[between] waiting ${INTERPASS_WAIT}s before Pass 2..."
sleep "$INTERPASS_WAIT"
set +e
run_nsys_pass 2 "$NSYS_PASS2_GPUS" "$NSYS_PASS2_FREQ" "$PASS2_WINDOW" \
    "$PASS2_PREWAIT" "$NSYS_ITERS" "pass2-iface-clock" "$PASS2_SESSION"
PASS2_RC=$?
set -e

# Pass 3 is intentionally disabled. Keep PASS3_RC=0 so the existing status
# schema remains backward-compatible with scheduler summaries.
echo "[between] waiting ${INTERPASS_WAIT}s before Pass 3..."
sleep "$INTERPASS_WAIT"
set +e
run_nsys_pass 3 "$NSYS_PASS3_GPUS" "$NSYS_PASS3_FREQ" "$PASS3_WINDOW" \
    "$PASS3_PREWAIT" "$NSYS_ITERS" "pass3-allgpu-clock" "$PASS3_SESSION"
PASS3_RC=$?
set -e

echo "[between] waiting ${INTERPASS_WAIT}s before Pass 4..."
sleep "$INTERPASS_WAIT"
set +e
run_pass4_dcgm
PASS4_RC=$?
set -e

GPU_RELEASE_MARKER="$RUN_ROOT/gpu_resource_released"
printf '%s\n' "$(date +%s.%3N)" > "$GPU_RELEASE_MARKER"
echo "GPU_RESOURCE_RELEASED run_root=$RUN_ROOT marker=$GPU_RELEASE_MARKER" >&2

wait_nsys_diagnostics

echo
echo "=== Four-pass done ==="
echo "Pass 1 rc=$PASS1_RC (CUDA/NVTX timeline)"
echo "Pass 2 rc=$PASS2_RC (NSYS GPU0 @ ${NSYS_PASS2_FREQ}Hz + clocks)"
echo "Pass 3 rc=$PASS3_RC (ENABLED)"
echo "Pass 4 rc=$PASS4_RC (DCGM long-window absolute bandwidth)"
echo "Outputs: $RUN_ROOT"

FINAL_STATUS="PASS"
FINAL_RC=0
HAS_OVERFLOW=0
HAS_OTHERERR=0
HAS_WORKLOAD_FAIL=0
for rc in "$PASS1_RC" "$PASS2_RC" "$PASS3_RC" "$PASS4_RC"; do
    case "$rc" in
        0) ;;
        "$RC_PROFILING_OVERFLOW") HAS_OVERFLOW=1 ;;
        "$RC_WORKLOAD_FAIL") HAS_WORKLOAD_FAIL=1 ;;
        *) HAS_OTHERERR=1 ;;
    esac
done
if [[ "$HAS_WORKLOAD_FAIL" -eq 1 ]]; then
    FINAL_STATUS="WORKLOAD_FAIL"
    FINAL_RC="$RC_WORKLOAD_FAIL"
elif [[ "$HAS_OTHERERR" -eq 1 ]]; then
    FINAL_STATUS="PROFILING_OTHERERR"
    FINAL_RC="$RC_PROFILING_OTHERERR"
elif [[ "$HAS_OVERFLOW" -eq 1 ]]; then
    FINAL_STATUS="PROFILING_OVERFLOW"
    FINAL_RC="$RC_PROFILING_OVERFLOW"
fi
printf '{"status":"%s","pass1_rc":%d,"pass2_rc":%d,"pass3_rc":%d,"pass4_rc":%d}\n' \
    "$FINAL_STATUS" "$PASS1_RC" "$PASS2_RC" "$PASS3_RC" "$PASS4_RC" \
    > "$RUN_ROOT/profile_status.json"
echo "FINAL_STATUS $FINAL_STATUS"
exit "$FINAL_RC"
