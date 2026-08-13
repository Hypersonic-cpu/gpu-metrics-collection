#!/usr/bin/env bash
# 单次长跑完成 native(前10次)、Nsys、DCGM；支持 1-GPU MHA 和 manifest 中的 8-GPU case。
set -euo pipefail

if [[ ${1:-} != --inside-timeout ]]; then
  exec timeout --foreground --signal=TERM --kill-after=15s 300s "$0" --inside-timeout "$@"
fi
shift

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
case_id=${CASE_ID:-${1:-AG_SMOKE}}
devices=${DEVICES:-${2:-0,1,2,3,4,5,6,7}}
nsys_gpus=${NSYS_GPUS:-${3:-0}}
iterations=${ITERATIONS:-1000000000}
nsys_delay=${NSYS_DELAY:-30}
nsys_seconds=${NSYS_SECONDS:-1}
dcgm_seconds=${DCGM_SECONDS:-5}
nsys_bin=${NSYS_BIN:-/usr/local/cuda/bin/nsys}
tk_python=${TK_PYTHON:-$HOME/Repos/ThunderKittens/.venv/bin/python3}
stamp=$(date +%Y%m%d-%H%M%S)
out=${OUT_ROOT:-$repo/runs/tk_profile_${case_id}_$stamp}
session=${NSYS_SESSION:-tkprofile_$stamp}
session_live=0
launch_pid=""
watchdog_pid=""

cleanup() {
  [[ -n $watchdog_pid ]] && kill "$watchdog_pid" >/dev/null 2>&1 || true
  if [[ $session_live == 1 ]]; then
    sudo -n -E "$nsys_bin" shutdown --session="$session" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

gpu_set_free() {
  local pids
  # nvidia-smi 启动很慢：每轮只调用一次，-i 一次传入全部目标卡。
  pids=$(nvidia-smi -i "$devices" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null) || return 1
  [[ -z ${pids//[[:space:]]/} ]]
}

echo "[driver] waiting for GPUs $devices (1s polling; total timeout 300s)"
if [[ ${ALLOW_BUSY:-0} == 1 ]]; then
  echo "[driver] ALLOW_BUSY=1: explicit test override, skipping the free-GPU wait"
else
  until gpu_set_free; do sleep 1; done
fi
mkdir -p "$out/nsys" "$out/dcgm"

workload_log="$out/workload.log"
sudo -n -E "$nsys_bin" launch \
  --session-new="$session" --trace=cuda,nvtx --cuda-graph-trace=node -- \
  "$tk_python" "$repo/scripts/thunderkittens/profile_launch_case.py" \
    --case "$case_id" --devices "$devices" --iterations "$iterations" \
    --native-samples 10 >"$workload_log" 2>&1 &
launch_pid=$!
session_live=1
# 外层 timeout 的 TERM/KILL 可能中断 EXIT trap；独立 watchdog 确保 session/app 仍会被关闭。
(
  for _ in $(seq 1 285); do sleep 1; done
  sudo -n -E "$nsys_bin" shutdown --session="$session" >/dev/null 2>&1 || true
) &
watchdog_pid=$!

for _ in $(seq 1 100); do
  sudo -n -E "$nsys_bin" sessions list 2>/dev/null | grep -q "$session" && break
  sleep 0.2
done
sudo -n -E "$nsys_bin" sessions list 2>/dev/null | grep -q "$session" || {
  echo "nsys session failed; see $workload_log" >&2; exit 1;
}
for _ in $(seq 1 120); do
  grep -q 'ROI_LOOP_BEGIN' "$workload_log" 2>/dev/null && break
  sleep 1
done
grep -q 'ROI_LOOP_BEGIN' "$workload_log" || {
  echo "workload did not enter long loop; see $workload_log" >&2; exit 1;
}

sleep "$nsys_delay"
echo "[driver] Nsys window: metrics GPUs=$nsys_gpus duration=${nsys_seconds}s"
sudo -n -E "$nsys_bin" start \
  --session="$session" --output="$out/nsys/report" --force-overwrite=true \
  --sample=none --cpuctxsw=none \
  --gpu-metrics-devices="$nsys_gpus" \
  --gpu-metrics-set="file:$repo/tool/nsys/sets/iface_full.config" \
  --gpu-metrics-frequency=10000
sleep "$nsys_seconds"
sudo -n -E "$nsys_bin" stop --session="$session"
sudo -n chown -R "$(id -u):$(id -g)" "$out"

echo "[driver] DCGM window: GPUs=$devices duration=${dcgm_seconds}s"
"$repo/tool/profile.sh" --backend dcgm --gpus "$devices" \
  --fields pass1_core --interval-ms 100 --duration "$dcgm_seconds" \
  --out "$out/dcgm" --note "TK $case_id long-loop attach after Nsys"
dcgm_run=$(find "$out/dcgm" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
python3 "$repo/tool/metrics.py" parse "$dcgm_run" | tee "$out/dcgm_summary.txt"

sudo -n -E "$nsys_bin" shutdown --session="$session"
wait "$launch_pid" 2>/dev/null || true
session_live=0
kill "$watchdog_pid" >/dev/null 2>&1 || true
watchdog_pid=""
sudo -n chown -R "$(id -u):$(id -g)" "$out"

echo "[driver] workload killed; validating Nsys report"
"$nsys_bin" export --type sqlite --force-overwrite=true \
  --output "$out/nsys/report.sqlite" "$out/nsys/report.nsys-rep"
"$nsys_bin" stats --force-export=true --report cuda_gpu_kern_sum \
  --format csv --output "$out/nsys/kernel" "$out/nsys/report.nsys-rep"
NSYS_HOST_BIN="$nsys_bin" python3 "$repo/tool/nsys/check_report.py" \
  "$out/nsys/report.nsys-rep"

family=$($tk_python "$repo/scripts/thunderkittens/profile_launch_case.py" \
  --case "$case_id" --devices "$devices" --iterations 1 --dry-run | \
  sed -n 's/.*family=\([^ ]*\).*/\1/p')
case "$family" in
  MHA) kernel=fwd_attend_ker;;
  AG) kernel=main_kernel;;
  RS) kernel=matmul_reduce_scatter;;
  MoE) kernel=dispatch_group_gemm_kernel;;
  *) echo "unknown family: $family" >&2; exit 1;;
esac
python3 "$repo/scripts/validate_nsys_kernel.py" "$out/nsys/report.sqlite" \
  --kernel "$kernel" --require-metrics
python3 "$repo/tool/nsys/post_export.py" "$out/nsys" --reuse-sqlite

python3 "$repo/scripts/summarize_tk_profile.py" \
  --workload-log "$workload_log" \
  --kernel-csv "$out/nsys/kernel_cuda_gpu_kern_sum.csv" \
  --nsys-metrics "$out/nsys/post_metrics.csv" \
  --dcgm-metrics "$dcgm_run/metrics.csv" \
  --output "$out/summary.json" | tee "$out/summary.txt"

trap - EXIT INT TERM
echo "[driver] PASS -> $out"
