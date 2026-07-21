# experiments/cumemcpy —— Copy Engine 三类 memcpy 的 metric 点亮验证（可复现，留档）

三个纯 memory-copy 测试，各自看**哪些 DCGM metric 会被点亮**，作为 `docs/metrics_reference.md`
结论的本机 ground-truth。均用 `ctypes` 直接调 `cudaMemcpy*` API（确保走 Copy Engine，而非误用 torch kernel）。

本机：`zkrh-58`，8× H100 80GB HBM3（HBM3 峰值≈3350 GB/s），NVLink-switched（12×NVSwitch），
DCGM 4.2.3，driver 590.48.01。镜像 `nvcr.io/nvidia/pytorch:25.12-py3`。

## 怎么跑
统一字段集 `F=204,1005,1002,1009,1010,449,1011,1012`
（mem_copy_util, dram_active, sm_active, pcie_tx/rx_bytes, nvlink_bw_total, nvlink_tx/rx_bytes）：
```bash
F=204,1005,1002,1009,1010,449,1011,1012
experiments/cumemcpy/run_probe.sh copy_h2d_d2h  4   $F 4   copy_h2d_d2h.py  40   # 实验1
experiments/cumemcpy/run_probe.sh copy_d2d_intra 4  $F 4   copy_d2d_intra.py 30  # 实验2
experiments/cumemcpy/run_probe.sh copy_d2d_inter 1,2 $F 1,2 copy_d2d_inter.py 30 # 实验3
```
> GPU0 常被别的租户占，用空卡（1–7）。byte/profiling 字段头 ~2 拍是 N/A 的 warmup。
>
> **原始日志已打包为 `logs.tar.gz`**（解压后即 `logs/` 树，下文 `logs/...` 路径均指压缩包内）。**按采样频率分目录**：`logs/freq1/`（1Hz, `-d 1000`）、`logs/freq10/`（10Hz, `-d 100`）。
> `run_probe.sh` 按 `DMON_MS` 自动落到 `logs/freq<Hz>/`，文件名用实验基名。三个 workload 参数化：`copy_*.py [size_mb]`，
> 默认 2048（=2GB busts-L2，向后兼容）；L2-fit 传 16（见实验4）。高频复刷：`run_probe.sh 10hz`；L2-fit 复刷：`run_probe.sh l2 16`。

## 三个实验（本机实测数值）

### 实验1 — `copy_h2d_d2h.py`：H2D + D2H（Copy Engine 过 PCIe）
`cudaMemcpyAsync(HostToDevice/DeviceToHost)`，pinned host 内存，2GB/次，各 ~57 GB/s。
- **H2D**：`pcie_rx_bytes`≈59 GB/s 点亮（device 收）；`pcie_tx`≈0。
- **D2H**：`pcie_tx_bytes`≈60 GB/s 点亮（device 发）；`pcie_rx`≈0。
- 两段都：`sm_active=0`（纯 Copy Engine，不占 SM）；`dram_active≈0.017`、`mem_copy_util=2`（HBM 只被轻碰）；`nvlink=0`。
- 日志：`logs/freq1/dcgmi_dmon_copy_h2d_d2h.txt`（10Hz 版见 `logs/freq10/`）。

### 实验2 — `copy_d2d_intra.py`：单卡内 D2D
`cudaMemcpyAsync(DeviceToDevice)` 同一张卡内，2GB/次，拷贝 ~1525 GB/s（HBM 读+写≈3050 GB/s）。
- **`sm_active=0.995`！** → 同卡 D2D **由 SM（kernel）搬运，不是 Copy Engine**（这是 CUDA 的实现：
  同卡拷贝用 SM 拿满 HBM 带宽；只有跨边界 H2D/D2H/peer 才用 Copy Engine）。
- `dram_active≈0.91`、`mem_copy_util=100`（HBM 近饱和）；`pcie/nvlink=0`（不出卡）。
- 日志：`logs/freq1/dcgmi_dmon_copy_d2d_intra.txt`（10Hz 版见 `logs/freq10/`）。

### 实验3 — `copy_d2d_inter.py`：卡间 D2D（Copy Engine 过 NVLink）
`cudaMemcpyPeerAsync` GPU1→GPU2 单向，**启用 P2P**后 ~399 GB/s。
- `nvlink_tx_bytes`(源 GPU1)≈407 GB/s、`nvlink_rx_bytes`(目的 GPU2)≈407 GB/s、`nvlink_bw_total`(449)≈388 GB/s 点亮，方向分明。
- `sm_active=0`（纯 Copy Engine）；`dram_active≈0.119`、`mem_copy_util=15`（HBM 仅 ~12%，NVLink 才是瓶颈）；`pcie≈0`。
- ⚠️ **坑**：若不启 P2P，`cudaMemcpyPeer` 退化成 **host 中转（PCIe）**——实测只有 37 GB/s，且 `pcie` 点亮、`nvlink=0`。
  "卡间拷贝"未必真走 NVLink，务必用 nvlink metric 核对。
- 日志：`logs/freq1/dcgmi_dmon_copy_d2d_inter.txt`（10Hz 版见 `logs/freq10/`）。

### 实验4 — L2 缓存的影响：`cudaMemcpy` 能绕过 L2 吗？（`run_probe.sh l2`）
把上面三个 copy 的拷贝大小从 2GB（busts L2）缩到 **16MB/卡**（src+dst 一起 < L2；H100 L2=**50MB**），复跑对照。
- **先纠正一个说法**：`cudaMemcpy` **绕不过 L2**。NVIDIA 工程师（论坛）："ALL accesses to DRAM take place
  through the L2 cache. And the L2 cache cannot be disabled." L2 无法关闭，唯一 bypass 是 SM 直读 pinned sysmem(zero-copy)，非 memcpy。
  - 出处（论坛，非正式文档，故以本机实测为准）：<https://forums.developer.nvidia.com/t/cudamemcpy-and-l2-cache/42817>、
    <https://forums.developer.nvidia.com/t/cudamemcpy-devicetodevice-and-l2-cache-usage/315370>。
- **但拷贝若整块住进 L2，HBM 就被挡掉**（大 buffer >> 50MB 时 L2 被冲刷、命中≈0，所以原始 2GB 版才满打 HBM）：

| 场景 | 指标 | 2GB(busts L2) | 16MB(fits L2) | 说明 |
|---|---|---|---|---|
| 单卡 D2D | DRAMA/MCUTL/BW | 0.907 / 100 / 1525 GB/s | **0.185 / 26 / 1895 GB/s** | HBM 掉 5×、带宽反升＝L2 在供数据 |
| 卡间 D2D 源 | DRAMA / NVLTX | 0.118 / ~405 | **0.003 / ~283** | HBM≈0 但 **NVLink 照旧** |
| 卡间 D2D 目的 | DRAMA / NVLRX | 0.113 / ~405 | **0.000 / ~281** | HBM≈0 但 **NVLink 照旧** |
| H2D(写) | DRAMA / PCIRX | 0.016 / ~59 | 0.016 / ~58 | 写仍落 HBM（不变）|
| D2H(读) | DRAMA / PCITX | 0.016 / ~60 | **0.001 / ~48** | 读命中 L2，HBM≈0 |

- **结论**：①接口计数器（pcie/nvlink/449）在接口处计量、**与 L2 无关**，永远如实反映过线流量 → 判"有没有在搬"认它；
  ②`dram_active`/`mem_copy_util` 是 **HBM** 指标、**会被 L2 挡掉**（cache-resident copy 让它≈0 却在猛搬）→ 别拿它判"有没有在拷贝"。
  读写不对称：所有读 + 卡间目的写被 L2 吸收，唯独 H2D 写 device 仍落 HBM（copy-engine 写策略无正式文档，仅记实测）。
- 跑法：`experiments/cumemcpy/run_probe.sh l2 16`；日志 `logs/freq1/dcgmi_dmon_copy_*_l2fit.txt`（10Hz: `logs/freq10/`）。指标层面的理解（"L2 会挡住 HBM 指标"）见 `docs/metrics_reference.md` §5.1。
- workload 已参数化：`copy_*.py [size_mb]`，默认 2048（＝原始 2GB，向后兼容，原始日志可复现）。

## 点亮矩阵（一图看尽）
| metric | H2D | D2H | 单卡 D2D | 卡间 D2D(NVLink) |
|---|---|---|---|---|
| `sm_active`(1002) | 0 | 0 | **0.995** | 0 |
| `dram_active`(1005) | 0.017 | 0.016 | **0.91** | 0.119 |
| `mem_copy_util`(204) | 2 | 2 | **100** | 15 |
| `pcie_rx_bytes`(1010) | **59 GB/s** | ~0 | ~0 | ~0 |
| `pcie_tx_bytes`(1009) | ~0 | **60 GB/s** | ~0 | ~0 |
| `nvlink_tx_bytes`(1011) | 0 | 0 | 0 | **407(源)** |
| `nvlink_rx_bytes`(1012) | 0 | 0 | 0 | **407(目的)** |
| `nvlink_bw_total`(449) | 0 | 0 | 0 | **388** |

## 结论
- **引擎归属**：H2D/D2H/卡间D2D = Copy Engine（`sm_active=0`）；**单卡 D2D = SM kernel（`sm_active≈1`）**。
  → 别用 `sm_active` 判断"有没有搬数据"：跨边界拷贝时它=0。
- **接口计量不漏 copy engine**：`pcie_bytes`/`nvlink_bytes`/`dram_active` 按接口计量，Copy Engine 的流量全被计到。
- **`mem_copy_utilization`(204) = 本卡 HBM 总带宽利用率的粗粒度整数百分比**（≈`dram_active×100`：58GB/s→2、
  406GB/s→15、3050GB/s→100）。它把**所有** HBM 读写（各种 memcpy + 计算 kernel）**混成一个标量、无接口拆分**。
  - **卡间拷贝为何只 15**：那次仅 406 GB/s，占 HBM 峰值(3350)的 ~12%——瓶颈是 NVLink 不是 HBM，HBM 很闲。
  - **`MCUTL ≥ DRAMA×100`、且负载越高差越多**（本机 +0.3→+3.2→+9.3，单卡满载 MCUTL 早早饱和到 100 而 DRAMA=0.91）：
    MCUTL 按"内存控制器在忙的时间占比"、DRAMA 按"数据总线真在传的周期占比"——结构性差异，非噪声。深入解析见 metrics_reference §5.1。
  - **本工具不采它**：接口性能靠 `dram_active`(HBM)+`pcie_bytes`+`nvlink_bytes/449` 分接口刻画，204 拆不出接口、又比 dram_active 粗。见 metrics_reference §5.1。
- **对采集字段的影响**：第一遍 `dram_active + nvlink_tx/rx_bytes(+449) + pcie_bytes` 覆盖全部接口带宽，
  对 Copy Engine 与 SM-kernel 驱动都不漏。
