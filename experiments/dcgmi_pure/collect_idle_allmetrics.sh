#!/usr/bin/env bash
# 采一段"纯 idle（无 workload）"的全字段 dcgmi dmon，用来定性回答：
#   除了 PCIe，其它接口/计算指标在空卡上有没有 idle 底噪？
# 字段一次全采（HBM/compute/PCIe/NVLink/显存并列），同一张卡同一拍横向对比，
# 谁非 0 一眼可见。dcgmi 是宿主机二进制（本账号 passwordless sudo），无 workload 不进容器。
#
# 用法: collect_idle_allmetrics.sh <gpu_id> [count] [interval_ms]
#   gpu_id     : 物理卡号（务必选空卡：nvidia-smi 里 mem=0 util=0）
#   count      : 采几拍（默认 350 拍 @10Hz ≈ 35s）
#   interval_ms: 采样间隔（默认 100ms = 10Hz，profiling 有效地板；见 README/§3.1）
set -euo pipefail
GPU="${1:?need gpu id}"; COUNT="${2:-350}"; IV="${3:-100}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# 字段分组（短名见 docs/host_probe/dcgmi_fields_list.txt）：
#  HBM      1005 dram_active(DRAMA)
#  compute  1002 sm_active(SMACT) 1003 sm_occupancy(SMOCC) 1001 gr_engine_active(GRACT) 1004 tensor_active(TENSO)
#  HBM 粗   204  mem_copy_util(MCUTL, device)
#  PCIe     1009 pcie_tx_bytes(PCITX) 1010 pcie_rx_bytes(PCIRX)
#  NVLink   449  nvlink_bandwidth_total(NBWLT, device) 1011 nvlink_tx_bytes(NVLTX) 1012 nvlink_rx_bytes(NVLRX)
#  显存容量 252  fb_used(FBUSD)
F=1005,1002,1003,1001,1004,204,1009,1010,449,1011,1012,252
OUT="$HERE/idle_allmetrics_${IV}ms_gpu${GPU}.txt"

echo "[idle-allmetrics] gpu=$GPU count=$COUNT interval=${IV}ms -> $OUT"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i "$GPU"
sudo dcgmi dmon -e "$F" -i "$GPU" -d "$IV" -c "$COUNT" > "$OUT"
echo "[done] $(wc -l < "$OUT") 行 -> $OUT"
