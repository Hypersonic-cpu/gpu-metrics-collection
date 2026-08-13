#!/usr/bin/env bash
# Collect native timing, one Nsys window, and one DCGM window for one official P0 case.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
manifest=${1:?usage: collect_tk_p0_case.sh MANIFEST CASE OUT_DIR}
case_id=${2:?usage: collect_tk_p0_case.sh MANIFEST CASE OUT_DIR}
out=${3:?usage: collect_tk_p0_case.sh MANIFEST CASE OUT_DIR}
group=${GROUP:-p0}
devices=${DEVICES:-0,1,2,3,4,5,6,7}
nsys_gpus=${NSYS_GPUS:-0}
nsys_duration=${NSYS_DURATION:-0.2}
nsys_frequency=${NSYS_FREQUENCY:-100000}
nsys_delay=${NSYS_DELAY:-30}
dcgm_duration=${DCGM_DURATION:-5}
warmup=${WARMUP_ITERS:-10}
native_iters=${NATIVE_ITERS:-20}
profile_iters=${PROFILE_ITERS:-1000000000}
profile_timeout=${PROFILE_TIMEOUT:-300}
nsys_bin=${NSYS_BIN:-/usr/local/cuda/bin/nsys}
tk_python=${TK_PYTHON:-$HOME/Repos/ThunderKittens/.venv/bin/python3}
torchrun=${TORCHRUN:-$HOME/Repos/ThunderKittens/.venv/bin/torchrun}
variant_root=${MOE_VARIANT_ROOT:-$repo/scripts/thunderkittens/moe_variants}
session=tkp0_${case_id}_$$_$(date +%s)
session_live=0
watchdog_pid=""
launch_pid=""

cleanup_broker_sockets() {
  local socket_path
  for socket_path in /tmp/kittens_broker.sock{0..7}; do
    [[ -S $socket_path ]] || continue
    if sudo -n lsof "$socket_path" 2>/dev/null | grep -q .; then
      echo "[COLLECT][$case_id] active broker socket retained: $socket_path" >&2
      continue
    fi
    sudo -n unlink "$socket_path"
  done
}

cleanup() {
  [[ -n $watchdog_pid ]] && kill "$watchdog_pid" >/dev/null 2>&1 || true
  if [[ $session_live == 1 ]]; then
    sudo -n -E "$nsys_bin" shutdown --session="$session" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

all_gpus_free() {
  local apps
  # One global query per polling round. Empty means all GPUs are free.
  apps=$(nvidia-smi --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null) || return 1
  [[ -z ${apps//[[:space:]]/} ]]
}

wait_for_all_gpus() {
  echo "[WAIT][$case_id] all GPUs must be free; one nvidia-smi query per 1s round"
  until all_gpus_free; do sleep 1; done
  echo "[WAIT][$case_id] free -> launch immediately"
}

mkdir -p "$out/native" "$out/nsys" "$out/dcgm"
wait_for_all_gpus
cleanup_broker_sockets

echo "[NATIVE LAUNCH][$case_id] warmup=$warmup iterations=$native_iters devices=$devices"
ARCH=SM90 CUDA_VISIBLE_DEVICES="$devices" TK_ROOT="${TK_ROOT:-$HOME/Repos/ThunderKittens}" \
  "$torchrun" --standalone --nproc_per_node=8 \
  "$repo/scripts/thunderkittens/official_p0_case.py" \
    --manifest "$manifest" --group "$group" --case "$case_id" \
    --warmup "$warmup" --iterations "$native_iters" \
    --variant-root "$variant_root" >"$out/native/run.log" 2>&1
echo "[NATIVE DONE][$case_id] log=$out/native/run.log"
cleanup_broker_sockets

echo "[PROFILE LAUNCH][$case_id] iterations=$profile_iters devices=$devices"
sudo -n -E "$nsys_bin" launch --session-new="$session" \
  --trace=cuda,nvtx --cuda-graph-trace=node -- \
  /usr/bin/env ARCH=SM90 CUDA_VISIBLE_DEVICES="$devices" \
    TK_ROOT="${TK_ROOT:-$HOME/Repos/ThunderKittens}" \
  "$torchrun" --standalone --nproc_per_node=8 \
  "$repo/scripts/thunderkittens/official_p0_case.py" \
    --manifest "$manifest" --group "$group" --case "$case_id" \
    --warmup "$warmup" --iterations "$profile_iters" \
    --variant-root "$variant_root" >"$out/workload.log" 2>&1 &
launch_pid=$!
session_live=1
(
  sleep "$profile_timeout"
  sudo -n -E "$nsys_bin" shutdown --session="$session" >/dev/null 2>&1 || true
) &
watchdog_pid=$!

for _ in $(seq 1 100); do
  sudo -n -E "$nsys_bin" sessions list 2>/dev/null | grep -q "$session" && break
  sleep 0.2
done
sudo -n -E "$nsys_bin" sessions list 2>/dev/null | grep -q "$session" || {
  echo "[COLLECT FAIL][$case_id] Nsys session did not appear" >&2; exit 1;
}
for _ in $(seq 1 180); do
  grep -q "OFFICIAL_P0_RUN_BEGIN case=$case_id" "$out/workload.log" 2>/dev/null && break
  sleep 1
done
grep -q "OFFICIAL_P0_RUN_BEGIN case=$case_id" "$out/workload.log" || {
  echo "[COLLECT FAIL][$case_id] official benchmark did not start" >&2; exit 1;
}
echo "[PROFILE READY][$case_id] official benchmark entered run(); delay=${nsys_delay}s"
sleep "$nsys_delay"

echo "[NSYS START][$case_id] duration=${nsys_duration}s gpus=$nsys_gpus frequency=${nsys_frequency}Hz"
sudo -n -E "$nsys_bin" start --session="$session" \
  --output="$out/nsys/report" --force-overwrite=true \
  --sample=none --cpuctxsw=none \
  --gpu-metrics-devices="$nsys_gpus" \
  --gpu-metrics-set="file:$repo/tool/nsys/sets/iface_full.config" \
  --gpu-metrics-frequency="$nsys_frequency"
sleep "$nsys_duration"
sudo -n -E "$nsys_bin" stop --session="$session"
sudo -n chown -R "$(id -u):$(id -g)" "$out"
echo "[NSYS DONE][$case_id] report=$out/nsys/report.nsys-rep"

echo "[DCGM START][$case_id] duration=${dcgm_duration}s gpus=$devices"
"$repo/tool/profile.sh" --backend dcgm --gpus "$devices" \
  --fields pass1_core --interval-ms 100 --duration "$dcgm_duration" \
  --out "$out/dcgm" --note "official P0 $case_id after Nsys"
dcgm_run=$(find "$out/dcgm" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
python3 "$repo/tool/metrics.py" parse "$dcgm_run" >"$out/dcgm_summary.txt"
echo "[DCGM DONE][$case_id] run=$dcgm_run"

echo "[KILL][$case_id] shutting down long official benchmark"
sudo -n -E "$nsys_bin" shutdown --session="$session"
wait "$launch_pid" 2>/dev/null || true
session_live=0
kill "$watchdog_pid" >/dev/null 2>&1 || true
watchdog_pid=""
cleanup_broker_sockets
sudo -n chown -R "$(id -u):$(id -g)" "$out"
{
  printf 'CASE_ID=%q\n' "$case_id"
  printf 'DCGM_RUN=%q\n' "$dcgm_run"
  printf 'NSYS_GPUS=%q\n' "$nsys_gpus"
  printf 'NSYS_DURATION=%q\n' "$nsys_duration"
  printf 'NSYS_FREQUENCY=%q\n' "$nsys_frequency"
} >"$out/collection.env"
trap - EXIT INT TERM
echo "[COLLECT DONE][$case_id] out=$out"

