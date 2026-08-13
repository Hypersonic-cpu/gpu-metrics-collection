#!/usr/bin/env bash
# 已验证策略：CUDA/NVTX 与 GPU Metrics 全程采集，再按 NVTX 离线筛 ROI。
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
nsys_bin=${NSYS_BIN:-/usr/local/cuda/bin/nsys}
python_bin=${PYTHON_BIN:-$HOME/.venvs/flux/bin/python}
gpu=${GPU:-0}
output=${OUTPUT:-/tmp/roi_probe_trace_metrics}

sudo -n -E "$nsys_bin" profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=none \
  --gpu-metrics-devices="$gpu" \
  --gpu-metrics-frequency=10000 \
  --force-overwrite=true \
  -o "$output" \
  "$python_bin" "$script_dir/nsys_roi_probe.py"

"$nsys_bin" stats --force-export=true --report cuda_gpu_kern_sum \
  "$output.nsys-rep"
