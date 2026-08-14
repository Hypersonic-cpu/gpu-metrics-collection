#!/usr/bin/env bash
# Native -> long rotating workload -> Nsys -> DCGM -> kill for one P1 case.
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
case_id=${1:?usage: collect_tk_p1_case.sh CASE OUT_DIR CSV ROTATION_CONFIG}
out=${2:?usage: collect_tk_p1_case.sh CASE OUT_DIR CSV ROTATION_CONFIG}
case_csv=${3:?usage: collect_tk_p1_case.sh CASE OUT_DIR CSV ROTATION_CONFIG}
rotation_config=${4:?usage: collect_tk_p1_case.sh CASE OUT_DIR CSV ROTATION_CONFIG}
world_size=${WORLD_SIZE:?WORLD_SIZE is required}
devices=${CASE_DEVICES:?CASE_DEVICES is required}
nsys_gpus=${NSYS_GPUS:-0}
nsys_duration=${NSYS_DURATION:-0.2}
nsys_frequency=${NSYS_FREQUENCY:-100000}
nsys_delay=${NSYS_DELAY:-15}
dcgm_duration=${DCGM_DURATION:-5}
rotation_copies=${ROTATION_COPIES:?ROTATION_COPIES is required}
rotation_seed_base=${ROTATION_SEED_BASE:?ROTATION_SEED_BASE is required}
native_iters=${NATIVE_ITERS:-20}
profile_iters=${PROFILE_ITERS:-1000000000}
profile_timeout=${PROFILE_TIMEOUT:-3600}
ready_timeout=${READY_TIMEOUT:-300}
nsys_bin=${NSYS_BIN:-/usr/local/cuda/bin/nsys}
tk_python=${TK_PYTHON:-$HOME/Repos/ThunderKittens/.venv/bin/python3}
torchrun=${TORCHRUN:-$HOME/Repos/ThunderKittens/.venv/bin/torchrun}
extension_root=${P1_EXTENSION_ROOT:-$repo/scripts/thunderkittens/p1_extensions}
runner=$repo/scripts/thunderkittens/p1_rotation_case.py
session=tkp1_${case_id}_$$_$(date +%s)
session_live=0
watchdog_pid=""
launch_pid=""

cleanup_broker_sockets() {
  [[ $world_size == 8 ]] || return 0
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
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    2>/dev/null) || return 1
  [[ -z ${apps//[[:space:]]/} ]]
}

echo "[WAIT][$case_id] all GPUs must be free; one nvidia-smi query per 1s round"
until all_gpus_free; do sleep 1; done
echo "[WAIT][$case_id] free -> launch immediately"
cleanup_broker_sockets
mkdir -p "$out/native" "$out/nsys" "$out/dcgm"
start_file=$out/measure.start
unlink "$start_file" 2>/dev/null || true

case_args=(
  "$runner" --csv "$case_csv" --rotation-config "$rotation_config"
  --case "$case_id" --warmup 1 --rotation-copies "$rotation_copies"
  --rotation-seed-base "$rotation_seed_base" --extension-root "$extension_root"
)
if [[ $world_size == 8 ]]; then
  app=("$torchrun" --standalone --nproc_per_node=8 "${case_args[@]}")
else
  app=("$tk_python" "${case_args[@]}")
fi

echo "[NATIVE LAUNCH][$case_id] warmup=1 iterations=$native_iters world=$world_size devices=$devices rotation_copies=$rotation_copies seed_base=$rotation_seed_base"
ARCH=SM90 CUDA_VISIBLE_DEVICES="$devices" TK_ROOT="${TK_ROOT:-$HOME/Repos/ThunderKittens}" \
  "${app[@]}" --iterations "$native_iters" --check-correctness \
  >"$out/native/run.log" 2>&1
echo "[NATIVE DONE][$case_id] log=$out/native/run.log"
cleanup_broker_sockets

echo "[PROFILE LAUNCH][$case_id] iterations=$profile_iters world=$world_size devices=$devices"
sudo -n -E "$nsys_bin" launch --session-new="$session" \
  --trace=cuda,nvtx --cuda-graph-trace=node -- \
  /usr/bin/env ARCH=SM90 CUDA_VISIBLE_DEVICES="$devices" \
    TK_ROOT="${TK_ROOT:-$HOME/Repos/ThunderKittens}" \
    ROTATION_START_FILE="$start_file" \
    "${app[@]}" --iterations "$profile_iters" \
    >"$out/workload.log" 2>&1 &
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
ready_marker="ROTATION_PROFILE_READY case=$case_id"
legacy_marker="ROTATION_MEASURE_BEGIN case=$case_id"
gate_mode=""
for _ in $(seq 1 "$ready_timeout"); do
  if grep -q "$ready_marker" "$out/workload.log" 2>/dev/null; then
    gate_mode=1
    break
  fi
  if grep -q "$legacy_marker" "$out/workload.log" 2>/dev/null; then
    gate_mode=0
    break
  fi
  kill -0 "$launch_pid" 2>/dev/null || break
  sleep 1
done
[[ -n $gate_mode ]] || {
  echo "[COLLECT FAIL][$case_id] benchmark readiness marker missing after ${ready_timeout}s" >&2
  exit 1
}
if [[ $gate_mode == 1 ]]; then
  printf '[PROFILE READY][%s] marker=%s delay=%ss gate=start-file\n' \
    "$case_id" "$ready_marker" "$nsys_delay"
  sleep "$nsys_delay"
else
  printf '[PROFILE READY][%s] marker=%s delay=0s gate=legacy\n' \
    "$case_id" "$legacy_marker"
fi

echo "[NSYS START][$case_id] duration=${nsys_duration}s gpus=$nsys_gpus frequency=${nsys_frequency}Hz"
nsys_control_log=$out/nsys/control.log
sudo -n -E "$nsys_bin" start --session="$session" \
  --output="$out/nsys/report" --force-overwrite=true \
  --sample=none --cpuctxsw=none \
  --gpu-metrics-devices="$nsys_gpus" \
  --gpu-metrics-set="file:$repo/tool/nsys/sets/iface_full.config" \
  --gpu-metrics-frequency="$nsys_frequency" >"$nsys_control_log" 2>&1
[[ $gate_mode == 0 ]] || touch "$start_file"
sleep "$nsys_duration"
sudo -n -E "$nsys_bin" stop --session="$session" >>"$nsys_control_log" 2>&1
sudo -n chown -R "$(id -u):$(id -g)" "$out"
echo "[NSYS DONE][$case_id] report=$out/nsys/report.nsys-rep"

echo "[DCGM START][$case_id] duration=${dcgm_duration}s gpus=$devices"
"$repo/tool/profile.sh" --backend dcgm --gpus "$devices" \
  --fields pass1_core --interval-ms 100 --duration "$dcgm_duration" \
  --out "$out/dcgm" --note "P1 rotation $case_id after Nsys" \
  >"$out/dcgm_collect.log" 2>&1
dcgm_run=$(find "$out/dcgm" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
python3 "$repo/tool/metrics.py" parse "$dcgm_run" >"$out/dcgm_summary.txt"
echo "[DCGM DONE][$case_id] run=$dcgm_run summary=$out/dcgm_summary.txt raw_console=$out/dcgm_collect.log"

echo "[KILL][$case_id] shutting down long rotating benchmark"
sudo -n -E "$nsys_bin" shutdown --session="$session"
wait "$launch_pid" 2>/dev/null || true
session_live=0
kill "$watchdog_pid" >/dev/null 2>&1 || true
watchdog_pid=""
cleanup_broker_sockets
sudo -n chown -R "$(id -u):$(id -g)" "$out"
{
  printf 'CASE_ID=%q\n' "$case_id"
  printf 'WORLD_SIZE=%q\n' "$world_size"
  printf 'CASE_DEVICES=%q\n' "$devices"
  printf 'NSYS_GPUS=%q\n' "$nsys_gpus"
  printf 'NSYS_DURATION=%q\n' "$nsys_duration"
  printf 'NSYS_FREQUENCY=%q\n' "$nsys_frequency"
  printf 'ROTATION_COPIES=%q\n' "$rotation_copies"
  printf 'ROTATION_SEED_BASE=%q\n' "$rotation_seed_base"
  printf 'READY_TIMEOUT=%q\n' "$ready_timeout"
  printf 'DCGM_RUN=%q\n' "$dcgm_run"
} >"$out/collection.env"
trap - EXIT INT TERM
echo "[COLLECT DONE][$case_id] out=$out"
