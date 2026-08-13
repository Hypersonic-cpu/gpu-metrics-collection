#!/usr/bin/env bash
# 官方 ag_gemm benchmark.py：短 native run + 独立超长 run 上顺序采 Nsys/DCGM。
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
nsys_bin=${NSYS_BIN:-/usr/local/cuda/bin/nsys}
tk_python=${TK_PYTHON:-$HOME/Repos/ThunderKittens/.venv/bin/python3}
devices=0,1,2,3,4,5,6,7
nsys_gpus=${NSYS_GPUS:-0}
size=${SIZE:-2048}
warmup=${WARMUP:-10}
native_iters=${NATIVE_ITERS:-20}
profile_iters=${PROFILE_ITERS:-1000000000}
stamp=${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}
out=${OUT_ROOT:-$repo/runs/tk_official_ag_${stamp}}
session=officialag_$stamp
session_live=0
watchdog_pid=""

cleanup_broker_sockets() {
  local socket_path
  for socket_path in /tmp/kittens_broker.sock{0..7}; do
    [[ -S $socket_path ]] || continue
    if sudo -n lsof "$socket_path" 2>/dev/null | grep -q .; then
      echo "[driver] refusing to unlink active broker socket: $socket_path" >&2
      return 1
    fi
    sudo -n unlink "$socket_path"
  done
}

if [[ ${1:-} != --inside-timeout ]]; then
  mkdir -p "$out/native"
  cleanup_broker_sockets
  echo "[driver] native official AG: size=$size warmup=$warmup iterations=$native_iters"
  ARCH=SM90 CUDA_VISIBLE_DEVICES="$devices" \
    "$HOME/Repos/ThunderKittens/.venv/bin/torchrun" --standalone --nproc_per_node=8 \
    "$repo/scripts/thunderkittens/official_ag_case.py" \
      --size "$size" --warmup "$warmup" --iterations "$native_iters" \
    >"$out/native/run.log" 2>&1
  export OUT_ROOT="$out" RUN_STAMP="$stamp"
  exec timeout --foreground --signal=TERM --kill-after=15s 300s \
    "$0" --inside-timeout
fi

cleanup() {
  [[ -n $watchdog_pid ]] && kill "$watchdog_pid" >/dev/null 2>&1 || true
  if [[ $session_live == 1 ]]; then
    sudo -n -E "$nsys_bin" shutdown --session="$session" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM
mkdir -p "$out/native" "$out/nsys" "$out/dcgm"
cleanup_broker_sockets

echo "[driver] launch official AG long run: iterations=$profile_iters"
sudo -n -E "$nsys_bin" launch --session-new="$session" \
  --trace=cuda,nvtx --cuda-graph-trace=node -- \
  /usr/bin/env ARCH=SM90 CUDA_VISIBLE_DEVICES="$devices" \
  "$HOME/Repos/ThunderKittens/.venv/bin/torchrun" --standalone --nproc_per_node=8 \
  "$repo/scripts/thunderkittens/official_ag_case.py" \
    --size "$size" --warmup "$warmup" --iterations "$profile_iters" \
  >"$out/workload.log" 2>&1 &
launch_pid=$!
session_live=1
(
  for _ in $(seq 1 285); do sleep 1; done
  sudo -n -E "$nsys_bin" shutdown --session="$session" >/dev/null 2>&1 || true
) &
watchdog_pid=$!

for _ in $(seq 1 100); do
  sudo -n -E "$nsys_bin" sessions list 2>/dev/null | grep -q "$session" && break
  sleep 0.2
done
for _ in $(seq 1 120); do
  grep -q 'OFFICIAL_AG_BEGIN' "$out/workload.log" 2>/dev/null && break
  sleep 1
done
grep -q 'OFFICIAL_AG_BEGIN' "$out/workload.log" || {
  echo "official benchmark did not start; see $out/workload.log" >&2; exit 1;
}

# OFFICIAL_AG_BEGIN 在 run() 前打印；给 30s 覆盖分配、通信初始化和 10 次 warmup。
sleep 30
echo "[driver] Nsys: GPU metrics=$nsys_gpus @10kHz, ~1s"
sudo -n -E "$nsys_bin" start --session="$session" \
  --output="$out/nsys/report" --force-overwrite=true \
  --sample=none --cpuctxsw=none \
  --gpu-metrics-devices="$nsys_gpus" \
  --gpu-metrics-set="file:$repo/tool/nsys/sets/iface_full.config" \
  --gpu-metrics-frequency=10000
sleep 1
sudo -n -E "$nsys_bin" stop --session="$session"
sudo -n chown -R "$(id -u):$(id -g)" "$out"

echo "[driver] DCGM: GPUs=$devices @10Hz, 5s"
"$repo/tool/profile.sh" --backend dcgm --gpus "$devices" \
  --fields pass1_core --interval-ms 100 --duration 5 \
  --out "$out/dcgm" --note "official AG size=$size after Nsys"
dcgm_run=$(find "$out/dcgm" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
python3 "$repo/tool/metrics.py" parse "$dcgm_run" >"$out/dcgm_summary.txt"

sudo -n -E "$nsys_bin" shutdown --session="$session"
wait "$launch_pid" 2>/dev/null || true
session_live=0
cleanup_broker_sockets
kill "$watchdog_pid" >/dev/null 2>&1 || true
watchdog_pid=""
sudo -n chown -R "$(id -u):$(id -g)" "$out"

"$nsys_bin" export --type sqlite --force-overwrite=true \
  --output "$out/nsys/report.sqlite" "$out/nsys/report.nsys-rep"
"$nsys_bin" stats --force-export=true --report cuda_gpu_kern_sum \
  --format csv --output "$out/nsys/kernel" "$out/nsys/report.nsys-rep"
NSYS_HOST_BIN="$nsys_bin" python3 "$repo/tool/nsys/check_report.py" \
  "$out/nsys/report.nsys-rep"
python3 "$repo/scripts/validate_nsys_kernel.py" "$out/nsys/report.sqlite" \
  --kernel main_kernel --require-metrics
python3 "$repo/tool/nsys/post_export.py" "$out/nsys" --reuse-sqlite

python3 "$repo/scripts/summarize_official_ag.py" \
  --native-log "$out/native/run.log" \
  --kernel-csv "$out/nsys/kernel_cuda_gpu_kern_sum.csv" \
  --nsys-metrics "$out/nsys/post_metrics.csv" \
  --dcgm-metrics "$dcgm_run/metrics.csv" \
  --output "$out/summary.json" | tee "$out/summary.txt"

trap - EXIT INT TERM
echo "[driver] PASS -> $out"
