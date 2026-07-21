# experiments/dcgmi_pure —— dcgmi 纯 idle 实测：各指标底噪 + 采样频率上限（无 workload）

> **原始 dmon/探针日志已打包为 `logs.tar.gz`**（下文出现的 `idle_*.txt`/`freqtest_*.txt`/`nvml_pcie_probe_gpu6.txt` 等文件名＝解压后路径）；
> CSV 产物在 `results/`、图在 `analysis/`、探针与负载脚本在 `workloads/`。

**两个结论先行**：

1. **空卡底噪只有 PCIe 一个**：纯 idle（无 workload）下，只有 `pcie_tx/rx_bytes`(1009/1010) 有 **~2 MB/样本
   ≈ 20 MB/s @10Hz** 的底噪（host↔GPU 的遥测/管理轮询本身走 PCIe）；**HBM(`dram_active`)、NVLink(`nvlink_tx/rx_bytes`
   +`nvlink_bandwidth_total`)、SM/计算(`sm_active`/`sm_occupancy`/`gr_engine_active`/`tensor_active`)、
   `mem_copy_util`、`fb_used` 全部 idle=0 / N/A，无底噪**（GPU7+GPU3 双卡实测，见下"idle 底噪总览"）。

2. **采样有效上限 = 10Hz(100ms)**：`dcgmi dmon -d` 虽然能设到 1ms（那只是客户端重复读缓存），但 **byte/profiling
   字段的有效采样率被 NVML GPM 的 ~100ms 内部刷新率钉死**（**这是 NVML/硬件的地板，不是 DCGM 的策略**；H100 上这批
   字段的数据源就是 NVML GPM，见 `docs/metrics_reference.md` §0）。超过 10Hz 只会在真数据之间插入**空行**，不提高时间
   分辨率——所以 **100ms(10Hz) 是 profiling/byte 字段的实际频率上限**，再快是无效采样。

---

## idle 底噪总览（全字段实测：只有 PCIe 有底噪）

采一段纯 idle 的**全字段** `dcgmi dmon`（HBM/计算/PCIe/NVLink/显存并列，同卡同拍横向对比，谁非 0 一眼可见），
空卡上逐字段判定"有没有底噪"。字段集 `1005,1002,1003,1001,1004,204,1009,1010,449,1011,1012,252`，
GPU7（长采 350 拍）+ GPU3（交叉验证 150 拍），10Hz。采集器 `collect_idle_allmetrics.sh`，
日志 `idle_allmetrics_100ms_gpu{7,3}.txt`，判定脚本 `analysis/analyze_idle_noise_allmetrics.py` → `results/idle_noise_allmetrics_summary.csv`。

| 接口/类别 | 字段 (id) | 短名 | **idle 底噪?** | idle 实测值 | 说明 |
|---|---|---|---|---|---|
| **PCIe** | `pcie_tx_bytes`(1009) | PCITX | **✅ 有** | mean **2.19 MB/样本 ≈ 21.9 MB/s** | host↔GPU 遥测/管理轮询走 PCIe（详见 Task1） |
| **PCIe** | `pcie_rx_bytes`(1010) | PCIRX | **✅ 有** | mean **1.96 MB/样本 ≈ 19.6 MB/s** | 同上 |
| HBM | `dram_active`(1005) | DRAMA | ❌ 无 | 0.000（空读 N/A） | 无 HBM 活动=真 0；ratio 字段空读返 N/A |
| 计算 | `sm_active`(1002) | SMACT | ❌ 无 | 0.000 | 无 kernel |
| 计算 | `sm_occupancy`(1003) | SMOCC | ❌ 无 | 0.000 | 无驻留 warp |
| 计算 | `gr_engine_active`(1001) | GRACT | ❌ 无 | 0.000 | = nvidia-smi GPU-Util 的近亲，idle 归 0 |
| 计算 | `tensor_active`(1004) | TENSO | ❌ 无 | 0.000 | 无 Tensor 管线活动 |
| HBM(粗) | `mem_copy_util`(204) | MCUTL | ❌ 无 | 0（整数%） | device 字段，idle 归 0 |
| **NVLink** | `nvlink_bandwidth_total`(449) | NBWLT | ❌ 无 | 0 | **device 计数器 idle=合法 0** |
| **NVLink** | `nvlink_tx_bytes`(1011) | NVLTX | ❌ 无 | 0 | **Fabric Manager 不产生可测的 NVLink 数据流量** |
| **NVLink** | `nvlink_rx_bytes`(1012) | NVLRX | ❌ 无 | 0 | 同上 |
| 显存容量 | `fb_used`(252) | FBUSD | ❌ 无 | 0 MB | 空卡真的 0 占用（非底噪，是容量） |

**为什么只有 PCIe 有底噪**：PCIe 是 host↔GPU 的必经管理通道——dcgmi/nv-hostengine/驱动的遥测、寄存器轮询、
心跳这些"看护流量"本身要过 PCIe，于是被 `pcie_*_bytes` 记成 ~20 MB/s 的常态底噪。HBM/NVLink/SM 这些是
**片内/片间的数据面**，idle 时没有任何 kernel/拷贝/跨卡通信去驱动它们，计数器就干净地停在 0。
**NVLink 特别确认**：本机是 NVLink-switched 节点（12×NVSwitch + Fabric Manager active），但 FM 只维持链路
train/健康态（控制面），**不在 `nvlink_*_bytes` 上留下数据字节**，故 idle NVLink 三个字段全 0（双卡一致）。

**对采集器的影响**：
- **PCIe 带宽曲线要扣底噪/设阈值**（详见下 Task1 三种去噪法；量级差真实传输 ~3 个数量级，也可忽略）。
- **HBM/NVLink/SM/显存无需扣底噪**：idle 即 0，任何非 0 都是真活动 → 判"有没有在用这条接口"可直接看是否 >0
  （NVLink 尤其干净，可当"是否真走了 NVLink"的判据，配合 §5.3 的 P2P 坑）。

## 采样频率上限 · 怎么测（>10Hz 空读塌陷）
纯 idle（无 workload）下用不同 `-d` 采一段，看 `PCITX/PCIRX`(pcie_tx/rx_bytes, profiling) 等
byte 字段有多少行是"空读"。字段集 `204,1005,1002,1009,1010,449,1011,1012`（含 MCUTL/DRAMA/SMACT/
PCITX/PCIRX/NBWLT/NVLTX/NVLRX），GPU4。日志：`idle_pcie_check_{5,10,20,100}hz.txt`。
（这批 `idle_pcie_check_*` 其实**整组字段都采了**，只是当时只分析 PCIe；上面"idle 底噪总览"里其它字段
idle=0 的结论，这批日志与新 `idle_allmetrics_*` 双向印证。）

## 实测：有效样本占比随频率的塌陷
统计 `PCITX`(pcie_tx_bytes) 列中"真值 vs 空读"的比例：

| 文件 | 频率 | 间隔 | 总行 | PCITX 真值 | 空读(=0) | **有效占比** | 说明 |
|---|---|---|---|---|---|---|---|
| `..._5hz.txt`   | 5Hz  | 200ms | 97  | 92  | 3   | **95%** | 低于内部率，基本全有效 |
| `..._10hz.txt`  | 10Hz | 100ms | 125 | 121 | 2   | **97%** | = NVML GPM 内部刷新率（本机 H100），刚好全有效 |
| `..._20hz.txt`  | 20Hz | 50ms  | 103 | 57  | 46  | **55%** | 2× 内部率 → ~一半空行 |
| `..._100hz.txt` | 100Hz| 10ms  | 150 | 20  | 130 | **13%** | 10× 内部率 → 只剩 ~1/10 真数据 |

有效占比 ≈ 100ms/间隔，与"内部 100ms 更新一次"完全吻合。

## 空读的两种signature（重要，parser 要认）
超过内部更新率时，同一次"空读"里不同字段的表现不同：
- **byte 计数器**（`PCITX/PCIRX/NVLTX/NVLRX`, 1009-1012）：空读返回 **`0`**。
  → ⚠️ **和"真的没流量"无法区分**，高频下 byte 字段的 `0` 是二义的，最坑。
- **ratio 字段**（`DRAMA`=dram_active 1005）：空读返回 **`N/A`**（本表 20Hz 有 47 个 N/A、100Hz 有 129 个）。
  → 至少能和真值 `0.000` 区分开。

## 对本工具的影响（写进 metrics_reference §3 / PLAN）
1. **profiling/byte 字段的默认最高频率 = 10Hz(100ms)**，不要设更快：更快不增分辨率、只产空行，
   还要 parser 额外过滤，且 byte 空读的 `0` 会污染带宽曲线。
2. 若确实要 100ms 以内的分辨率：**DCGM 与 NVML GPM 这条路都做不到**（同一个 ~100ms 地板，绕过 DCGM
   直调 `nvmlGpmSampleGet` 也没用）；PCIe 可退到**经典 NVML `nvmlDeviceGetPcieThroughput` ~25–50Hz**
   （见下 §分析 Task3，注意那是**经典 NVML 设备查询、不是 GPM**），HBM/NVLink 想更快只能 Nsight/CUPTI（见 metrics_reference §6）。
3. 性能开销不是限制频率的原因——`experiments/dcgmi_overhead` 实测即便 `-d 1`(1000Hz) 对 workload
   的拖慢也 <0.3%。**限制频率的是数据有效性，不是开销。**

---

# 分析（`analysis/`）—— PCIe 底噪 + 频率上限 + 能否更高

> 分析脚本在**容器内**跑（宿主机 pip 被 OOM kill）：
> `docker run --rm -v "$PWD":/work -w /work nvcr.io/nvidia/pytorch:25.12-py3 python /work/experiments/dcgmi_pure/analysis/analyze_idle_and_freq.py`
> 频率探针在**宿主机**跑（dcgmi 是 host 二进制）：`python3 workloads/dcgm_refresh_probe.py 6 3000 .`
> 图 = `analysis/*.png`；统计 = `results/stats.txt`；探针原始日志 = `logs.tar.gz` 内 `freqtest_*_gpu6.txt`。

## Task1 · PCIe 空载底噪（idle，无 workload）
数据源：`idle_pcie_check_{gpu45(1Hz),5hz,10hz}.txt` 的 PCITX/PCIRX 有效样本
（新 `idle_allmetrics_100ms_gpu{7,3}.txt` 复核一致：PCITX mean 2.19MB≈21.9MB/s、PCIRX 1.96MB≈19.6MB/s）。

- **每样本底噪 ≈ 1.7–2.9 MB**（合并 1/5/10Hz：mean 2.38MB、median 2.35MB、p95 4.6MB），**与采样间隔近似无关**
  —— 每个非空样本 ≈ 一个 100ms 内部窗累积的 PCIe housekeeping（DCGM/驱动自身轮询走 PCIe）。
- **折算成带宽**（采集器 bytes/interval）：**10Hz 运行点 ≈ TX 17 MB/s、RX 17 MB/s**（低频 1Hz 会欠采到 ~3MB/s，
  因为该字段只报最近一个 ~100ms 窗、不是整秒累积——低频反而低估，另见 Task3）。
- **对照真实传输差 ~3 个数量级**：idle ~17 MB/s vs H2D workload ~59 000 MB/s（cumemcpy 实测）。
- **怎么去掉**（图 `task1_idle_vs_workload_scale.png`）：底噪是**平稳窄带噪声**（图 `task1_idle_timeseries_10hz.png`），
  三选一即可，且对精度影响 <0.1%：
  1. **阈值最省事**：<100 MB/s 视为"无真实传输"→ idle 直接归零，不显示成假流量。
  2. **常数扣除**：在运行频率上先采一段 idle 测底噪均值（10Hz≈1.7MB/样本），逐样本减掉。
  3. **忽略**：量级差 3 个数量级，不减也不影响带宽结论。
- 图：`task1_idle_hist_10hz.png`（分布）、`task1_persample_vs_freq_box.png`（每样本 ~ 与间隔无关）。

## Task1b · 其它指标 idle 无底噪（HBM/NVLink/SM/计算/显存）
Task1 只查了 PCIe。这一步把**全字段**一起采（`collect_idle_allmetrics.sh`，GPU7 350 拍 + GPU3 150 拍 @10Hz），
逐字段判定有无底噪，脚本 `analysis/analyze_idle_noise_allmetrics.py`（纯 stdlib 宿主机跑）→ `results/idle_noise_allmetrics_summary.csv`。
**结论：除 PCIe 外全部 CLEAN（idle=0/N/A，0 个非零样本）**——两卡 348/148 个有效样本里，DRAMA/SMACT/SMOCC/
GRACT/TENSO/MCUTL/NBWLT/NVLTX/NVLRX/FBUSD 无一非零。完整逐字段判定表见文首"idle 底噪总览"。
- 只有 PCIe 有底噪的原因、NVLink 对 Fabric Manager 的特别确认：见文首"为什么只有 PCIe 有底噪"。
- **没有像 PCIe 那样的去噪分析/图**，因为这些字段 idle 就是 0——没有底噪可扣，任何非零即真活动。

## Task2 · >10Hz 就是"好多 0 的无效行"（你的猜想 ✅ 成立）
把 `dcgmi_pure`(idle) 与 `dcgmi_overhead`(有 workload) 的高频日志一起统计，**空读占比只由频率决定、与有没有流量无关**：

| 来源 | 文件 | 频率 | PCITX=0 空读 | DRAMA=N/A 空读 | 有效% |
|---|---|---|---|---|---|
| idle | `idle_pcie_check_10hz.txt`   | 10Hz   | 1.6%  | 3.2%  | **96.8%** |
| idle | `idle_pcie_check_20hz.txt`   | 20Hz   | 44.7% | 45.6% | **55.3%** |
| idle | `idle_pcie_check_100hz.txt`  | 100Hz  | 86.7% | 86.0% | **13.3%** |
| workload | `dmon_mem_full_100_rep1.txt`(overhead) | 10Hz | 7.3% | 7.3% | **92.7%** |
| workload | `dmon_mem_full_1_rep1.txt`(overhead)   | 1000Hz | 87.8% | 87.5% | **12.2%** |
| workload | `dmon_gemm_full_1_rep1.txt`(overhead)  | 1000Hz | 91.6% | 93.3% | **8.4%** |

→ 有效% ≈ `min(1, 10Hz/freq)`，idle 与 workload 两条线**完全重合**（图 `task2_validfrac_vs_freq.png`）。
**结论：是的，freq>10Hz 就有大量 0 无效行**。根因是 **NVML GPM 靠两张快照相减、GPU 内部计数器 ~100ms 才刷新
一次**（见 metrics_reference §0.4），窗口比它短就没有新增量可算；这也是 DCGM feature-overview 记的**官方已知行为**
（"collection at higher frequencies will result in zeroes"，
<https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html>）。

## Task3 · DCGM 采样真就这么低吗？能不能高点？（联网复核 + 本机实测）
**官方文档**：DCGM feature-overview 记 profiling 默认 1Hz、**可配置最小 100ms(10Hz)**、"不为高频采样设计"；
而这 100ms 的**真正来源**是 **NVML GPM 文档**明说的 *"the interval between two nvmlGpmSampleGet() calls should
be greater than 100ms due to the internal sample refresh rate"*
（<https://docs.nvidia.com/deploy/nvml-api/group__nvmlGpmFunctions.html>）。无任何 flag/env/config 能把它压到 100ms 以下。

**本机实测（GPU6，持续 H2D 负载，探针 `dcgm_refresh_probe.py` / `nvml_pcie_probe.py`）**：

| 路径 | 实测有效(非空/真刷新)率 | 备注 |
|---|---|---|
| DCGM profiling `-d 100`(req 10Hz) | **~10 Hz clean** | = 内部更新率，几乎全有效 |
| DCGM profiling `-d 1`(req 1000Hz) | **~3 Hz（塌了！）** | dmon 实际只出 ~300–345 行/s；2500 行里仅 23 行非零(59.4GB/s)，其余 99% 是 0。**超采反而更差** |
| **NVML `nvmlDeviceGetPcieThroughput`** | **~24 Hz（双读）/ ~50Hz（单读）** | 每次读都返回真值(288/288 非零)、读到 59.3 GB/s；官方=20ms 窗，**只有 PCIe** |
| Nsight Systems GPU Metrics / CUPTI PM Sampling | 10³–10⁵ Hz | 真高频，但**抢占**硬件计数器，与 DCGM profiling 互斥 |

图 `task3_paths_clean_rate.png`。**要点**：
1. **10Hz 是 NVML GPM（= DCGM profiling 在 H100 的底层数据源）的硬地板**（官方 + 实测双证），HBM/NVLink 想更快没有轻量路子。
2. **别把 dcgmi 拉到 >10Hz**：不仅无用，`-d 1` 下有效交付率反而从 10Hz 掉到 ~3Hz（multiplex 返回 0 越发严重）。
3. **唯一轻量的"高一点"= 经典 NVML `nvmlDeviceGetPcieThroughput` ~25–50Hz**（**属经典 NVML 设备查询、不是 GPM**，
   所以不吃 GPM 那个 100ms 地板；官方 20ms 窗），且**无空行**（每次都是真值），但**仅限 PCIe**；代价是每次调用阻塞
   ~20ms。要 HBM/NVLink 的 >10Hz，只能上 Nsight/CUPTI（抢计数器，不能和 DCGM 同跑）。

### 文件清单（脚本 / 产物 / 日志）
**`analysis/`（脚本 + 图）**
- `analyze_idle_noise_allmetrics.py` —— **Task1b** 全字段 idle 逐字段底噪判定（纯 stdlib，宿主机跑）→
  `results/idle_noise_allmetrics_summary.csv`（每 卡×字段 一行：verdict FLOOR/CLEAN + 非零统计）。
  输入 = `logs.tar.gz` 内 `idle_allmetrics_100ms_gpu{7,3}.txt`（`../collect_idle_allmetrics.sh` 采）
- `analyze_idle_and_freq.py` —— Task1+2 解析与出图（容器内跑）→ `task1_*.png`/`task2_*.png`、`results/stats.txt`
- `make_idle_csv.py` —— 把 Task1 底噪的**原始 PCITX/PCIRX** 导出 CSV（纯 stdlib，宿主机跑）→
  `results/idle_pcie_raw.csv`（每样本一行：原始 bytes + MB + MB·s⁻¹ + 状态 warmup/empty/valid）、
  `results/idle_pcie_summary.csv`（每 频率×方向 的 mean/median/std/分位 + 1/5/10Hz 合并行）
- `task3_plot.py` —— Task3 汇总图 → `task3_paths_clean_rate.png`

**`workloads/`（探针 + 负载，运行产原始日志）**
- `dcgm_refresh_probe.py` + `h2d_loop.py` —— Task3 内部刷新率探针（经 DCGM 观测 NVML GPM 的 ~100ms 刷新；host 探针 + 容器持续 H2D 负载）
  → `logs.tar.gz` 内 `freqtest_{1,16}field_1ms_{idle,h2dload}_gpu6.txt`
- `nvml_pcie_probe.py` —— Task3 NVML 替代路径刷新率（容器内 pynvml）→ `logs.tar.gz` 内 `nvml_pcie_probe_gpu6.txt`

**`results/`（产物）**：`idle_noise_allmetrics_summary.csv`、`idle_pcie_raw.csv`、`idle_pcie_summary.csv`、`stats.txt`、`freqtest_summary_gpu6.txt`（探针汇总）
**`logs.tar.gz`（原始 dump）**：`idle_allmetrics_100ms_gpu{7,3}.txt`、`idle_pcie_check_{gpu45,5hz,10hz,20hz,100hz}.txt`、`freqtest_{1,16}field_1ms_{idle,h2dload}_gpu6.txt`、`nvml_pcie_probe_gpu6.txt`
