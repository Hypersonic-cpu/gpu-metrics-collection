#!/usr/bin/env bash
# GPU5 空闲后启动 1e9 次 MHA；30s 后采 1s，释放全部资源 20s，
# 再用独立 session 启动 1e9 次 MHA 并采 1s；总超时 5 分钟。
set -euo pipefail

if [[ ${1:-} != --inside-timeout ]]; then
  exec timeout --foreground --signal=TERM --kill-after=15s 300s "$0" --inside-timeout
fi

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
nsys_bin=${NSYS_BIN:-/usr/local/cuda/bin/nsys}
tk_python=${TK_PYTHON:-$HOME/Repos/ThunderKittens/.venv/bin/python3}
stamp=$(date +%Y%m%d-%H%M%S)
out_root=${OUT_ROOT:-$repo/runs/tk_mha_delayed_$stamp}
session_base=${NSYS_SESSION:-tkmha5_$stamp}
session=""
iterations=1000000000
min_active_seconds=${MIN_ACTIVE_SECONDS:-75}
session_live=0
launch_pid=""

cleanup() {
  if [[ $session_live == 1 && -n $session ]]; then
    sudo -n -E "$nsys_bin" shutdown --session="$session" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

gpu5_free() {
  local processes
  processes=$(nvidia-smi -i 5 --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null) || return 1
  [[ -z ${processes//[[:space:]]/} ]]
}

echo "[driver] waiting for physical GPU5 (poll interval 1s, total timeout 300s)"
until gpu5_free; do sleep 1; done
gpu5_free || exec "$0" --inside-timeout
mkdir -p "$out_root"

start_workload() {
  local name=$1
  session="${session_base}_${name}"
  launch_log="$out_root/workload_${name}.log"
  sudo -n -E "$nsys_bin" launch \
    --session-new="$session" \
    --trace=cuda,nvtx \
    --cuda-graph-trace=node -- \
    "$tk_python" "$repo/scripts/thunderkittens/launch_case.py" \
      --case MHA_SMOKE --devices 5 --num-gpus 1 --iterations "$iterations" \
    >"$launch_log" 2>&1 &
  launch_pid=$!
  session_live=1

  for _ in $(seq 1 100); do
    sudo -n -E "$nsys_bin" sessions list 2>/dev/null | grep -q "$session" && break
    sleep 0.2
  done
  sudo -n -E "$nsys_bin" sessions list 2>/dev/null | grep -q "$session" || {
    echo "nsys session failed to start; see $launch_log" >&2
    exit 1
  }

  echo "[driver] waiting for MHA ROI_BEGIN ($name)"
  local ready=0
  for _ in $(seq 1 120); do
    if [[ -f $launch_log ]] && grep -q 'ROI_BEGIN case=MHA_SMOKE' "$launch_log"; then
      ready=1
      echo "[driver] MHA active ($name) at $(date --iso-8601=seconds)"
      break
    fi
    sleep 1
  done
  [[ $ready == 1 ]] || { echo "MHA did not enter ROI; see $launch_log" >&2; exit 1; }
}

stop_workload() {
  sudo -n -E "$nsys_bin" shutdown --session="$session"
  wait "$launch_pid" 2>/dev/null || true
  session_live=0
}

collect_window() {
  local name=$1
  local run_dir="$out_root/$name"
  mkdir -p "$run_dir"
  echo "[driver] start $name at $(date --iso-8601=seconds)"
  sudo -n -E "$nsys_bin" start \
    --session="$session" \
    --output="$run_dir/report" \
    --force-overwrite=true \
    --sample=none \
    --cpuctxsw=none \
    --gpu-metrics-devices=5 \
    --gpu-metrics-set="file:$repo/tool/nsys/sets/iface_full.config" \
    --gpu-metrics-frequency=10000
  sleep 1
  sudo -n -E "$nsys_bin" stop --session="$session"
  sudo -n chown -R "$(id -u):$(id -g)" "$run_dir"
  "$nsys_bin" export --type sqlite --force-overwrite=true \
    --output "$run_dir/report.sqlite" "$run_dir/report.nsys-rep"
  "$nsys_bin" stats --force-export=true --report cuda_gpu_kern_sum \
    --format csv --output "$run_dir/kernel" "$run_dir/report.nsys-rep"
}

start_workload window1
active_start=$(date +%s)
sleep 30
collect_window window1
stop_workload

echo "[driver] GPU5/profiler resources released; sleeping 20s"
sleep 20
until gpu5_free; do sleep 1; done
start_workload window2
collect_window window2
stop_workload

elapsed=$(( $(date +%s) - active_start ))
if (( elapsed < min_active_seconds )); then
  echo "[driver] keeping workload alive for $((min_active_seconds - elapsed))s"
  sleep $((min_active_seconds - elapsed))
fi

mapfile -t reports < <(find "$out_root" -name report.nsys-rep -type f | sort)
[[ ${#reports[@]} -eq 2 ]] || {
  echo "expected 2 reports, found ${#reports[@]}" >&2
  exit 1
}

for report in "${reports[@]}"; do
  run_dir=$(dirname "$report")
  NSYS_HOST_BIN=/usr/local/cuda/bin/nsys python3 \
    "$repo/tool/nsys/check_report.py" "$report"
  [[ -f $run_dir/report.sqlite ]] || {
    /usr/local/cuda/bin/nsys export --type sqlite --force-overwrite=true \
      --output "$run_dir/report.sqlite" "$report"
  }
  python3 "$repo/scripts/validate_nsys_kernel.py" "$run_dir/report.sqlite" \
    --kernel fwd_attend_ker --require-metrics
done

trap - EXIT INT TERM
echo "[driver] PASS: two GPU5 windows validated under $out_root"
