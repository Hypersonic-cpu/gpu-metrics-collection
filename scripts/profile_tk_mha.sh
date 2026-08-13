#!/usr/bin/env bash
# GPU5 上全程采 CUDA/NVTX + 10kHz HW metrics，再按 HTM_TARGET:MHA_SMOKE 离线筛 ROI。
set -euo pipefail

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
nsys_bin=${NSYS_BIN:-/usr/local/cuda/bin/nsys}
tk_python=${TK_PYTHON:-$HOME/Repos/ThunderKittens/.venv/bin/python3}
iterations=${ITERATIONS:-30000}
output=${OUTPUT:-/tmp/tk_mha_gpu5}

sudo -n -E "$nsys_bin" profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=none \
  --gpu-metrics-devices=5 \
  --gpu-metrics-frequency=10000 \
  --force-overwrite=true \
  -o "$output" \
  "$tk_python" "$repo/scripts/thunderkittens/launch_case.py" \
    --case MHA_SMOKE --devices 5 --num-gpus 1 --iterations "$iterations"

"$nsys_bin" export --type sqlite --force-overwrite=true \
  --output "$output.sqlite" "$output.nsys-rep"
"$nsys_bin" stats --force-export=true --report cuda_gpu_kern_sum \
  "$output.nsys-rep"
python3 "$repo/scripts/validate_nsys_kernel.py" "$output.sqlite" \
  --kernel fwd_attend_ker --roi-prefix HTM_TARGET:MHA_SMOKE --require-metrics
python3 "$repo/tool/nsys/check_report.py" "$output.nsys-rep"
