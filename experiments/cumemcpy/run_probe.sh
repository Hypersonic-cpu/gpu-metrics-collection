#!/usr/bin/env bash
# cumemcpy 实验跑测器(三合一)：核心引擎 probe + 两个预设子命令 10hz / l2。
# 宿主机起 dcgmi dmon，同时在 pytorch 容器里跑一个 copy workload，日志按频率落到 ./logs/freq<Hz>/。
# 这是采集器的雏形(attach + wrap 的简化版)，方便复现 metrics 可用性实验。
#
# 用法：
#   # ① 引擎(通用跑测)：
#   run_probe.sh <name> <host_gpu_ids> <field_ids> <container_devices> <workload.py> [dmon_count] [workload_args...]
#   # ② 预设：10Hz 复刷两种大小×三个 copy，共 6 个 log：
#   run_probe.sh 10hz
#   # ③ 预设：L2-fit 复跑(1Hz)：
#   run_probe.sh l2 [size_mb]        # 默认 16MB(src+dst=32MB<50MB L2)
#
# 统一字段集 F=204,1005,1002,1009,1010,449,1011,1012
#   (mem_copy_util, dram_active, sm_active, pcie_tx/rx_bytes, nvlink_bw_total, nvlink_tx/rx_bytes)
# 引擎三个 copy-engine 测试(物理卡 GPU0 常被占，用空卡)：
#   1) H2D+D2H(物理卡4)：      run_probe.sh copy_h2d_d2h  4   $F 4   copy_h2d_d2h.py  40
#   2) 单卡内 D2D(物理卡4)：   run_probe.sh copy_d2d_intra 4  $F 4   copy_d2d_intra.py 30
#   3) 卡间 D2D(物理卡1->2)：  run_probe.sh copy_d2d_inter 1,2 $F 1,2 copy_d2d_inter.py 30
# 第 7+ 个参数会**原样透传给 workload**(如拷贝大小 MB，用于 L2-fit 实验)：
#   run_probe.sh copy_d2d_intra_l2fit 4 $F 4 copy_d2d_intra.py 30 16   # -> workload 收到 "16"
# 环境变量 DMON_MS：采样间隔(ms)，默认 1000(1Hz)。10Hz 传 DMON_MS=100(官方下限)。
# 注：保留字 "10hz"/"l2" 是子命令，别拿它们当 <name>。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
IMG="nvcr.io/nvidia/pytorch:25.12-py3"
F=204,1005,1002,1009,1010,449,1011,1012

# 核心引擎：后台起 dmon(先起、workload 后跑、结尾 wait，天然把整个 workload 夹在 idle 头尾之间=全程覆盖)，
# 前台跑 workload 容器。日志按采样频率分子目录 logs/freq<Hz>/(1000ms->freq1, 100ms->freq10)。
probe() {
  local NAME="$1" HOST_GPUS="$2" FIELDS="$3" CDEV="$4" WL="$5" COUNT="${6:-40}"
  local WL_ARGS=("${@:7}")   # 透传给 workload 的额外参数(可空)
  local DMON_MS="${DMON_MS:-1000}"
  local LOGDIR="$HERE/logs/freq$((1000/DMON_MS))"; mkdir -p "$LOGDIR"
  local LOG="$LOGDIR/dcgmi_dmon_${NAME}.txt"

  echo "[run_probe] name=$NAME host_gpus=$HOST_GPUS fields=$FIELDS container_devices=$CDEV workload=$WL interval=${DMON_MS}ms count=$COUNT"
  echo "[run_probe] log -> $LOG"

  # COUNT×DMON_MS 要 > 容器启动+workload 时长，才不漏尾。容器启动的几秒天然充当 profiling warmup。
  dcgmi dmon -e "$FIELDS" -i "$HOST_GPUS" -d "$DMON_MS" -c "$COUNT" > "$LOG" 2>&1 &
  local DMON=$!
  # --gpus '"device=..."' 只暴露选定物理卡，容器内重编号为 cuda:0..
  docker run --rm --gpus "\"device=$CDEV\"" -v "$HERE/workloads":/w "$IMG" python /w/"$WL" "${WL_ARGS[@]}" \
    2>&1 | grep -E "START|END|DONE|L2|props|Error|error" || true
  echo "[run_probe] workload done, 等 dmon 收尾 ..."
  wait "$DMON"
  echo "[run_probe] 完成，日志见 $LOG"
}

# 预设①：10Hz(100ms, 官方 profiling 下限) 复刷 2GB(busts L2) 与 16MB(fits L2) × 三个 copy -> logs/freq10/。
# 目的：更细的时间序列，看清 H2D→sleep→D2H 过渡及 L2-fit 稳态。COUNT 放大到覆盖(容器启动+workload+尾巴)。
preset_10hz() {
  export DMON_MS=100
  echo "########## 2GB (busts L2) @ 10Hz ##########"
  probe copy_h2d_d2h   4   $F 4   copy_h2d_d2h.py  480   # 28s workload -> 48s 窗
  probe copy_d2d_intra 4   $F 4   copy_d2d_intra.py 380  # 20s workload -> 38s 窗
  probe copy_d2d_inter 1,2 $F 1,2 copy_d2d_inter.py 380
  echo "########## 16MB (fits L2) @ 10Hz ##########"
  probe copy_h2d_d2h_l2fit   4   $F 4   copy_h2d_d2h.py  480 16
  probe copy_d2d_intra_l2fit 4   $F 4   copy_d2d_intra.py 380 16
  probe copy_d2d_inter_l2fit 1,2 $F 1,2 copy_d2d_inter.py 380 16
  echo "########## done. logs: $HERE/logs/freq10/dcgmi_dmon_copy_*.txt ##########"
}

# 预设②：L2-fit 复跑(1Hz)，把拷贝大小缩到 << L2(H100=50MB)，与原始 2GB(busts L2) 版对照。
# 观察：接口计量(pcie/nvlink) 不受影响；HBM 的 dram_active/mem_copy_util 在 L2-fit 时应显著下降(甚至趋 0)，
# 而拷贝带宽反而更高。结论=L2 不可关(见 README 实验4 / docs/metrics_reference.md §5.1)。
preset_l2() {
  local SIZE_MB="${1:-16}"
  echo "=== L2-fit sweep, copy_size=${SIZE_MB}MB (原始对照 = 2048MB busts-L2 版) ==="
  probe copy_h2d_d2h_l2fit   4   $F 4   copy_h2d_d2h.py  40 "$SIZE_MB"   # 控制组：dram_active 本就≈0
  probe copy_d2d_intra_l2fit 4   $F 4   copy_d2d_intra.py 30 "$SIZE_MB"  # ★核心★ SM kernel 走 L2，HBM 被挡
  probe copy_d2d_inter_l2fit 1,2 $F 1,2 copy_d2d_inter.py 30 "$SIZE_MB"  # NVLink 计量应照旧≈400GB/s
  echo "=== done. logs: $HERE/logs/freq1/dcgmi_dmon_copy_*_l2fit.txt (1Hz) ==="
}

case "${1:-}" in
  10hz) shift; preset_10hz "$@";;
  l2)   shift; preset_l2 "$@";;
  "")   echo "用法见脚本头注释：run_probe.sh <name> <host_gpus> <fields> <cdev> <wl.py> [count] [args] | run_probe.sh 10hz | run_probe.sh l2 [size_mb]" >&2; exit 2;;
  *)    probe "$@";;
esac
