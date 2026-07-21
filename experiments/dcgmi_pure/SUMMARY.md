# dcgmi_pure —— 区域摘要 / 上手索引

> 在 `experiments/dcgmi_pure/` 下工作前先读这份 + `README.md`（结论与表格全在 README，这里是**地图 + 复现 + 待办**）。
> 采样频率地板的根因（NVML GPM 100ms）不在本区域重复，见 `CLAUDE.md` 关键坑 与 `docs/metrics_reference.md` §0.4/§3.1。

## 这个区域确立了什么（一句话）
1. **DCGM profiling/byte 字段的有效采样地板 = 10Hz(100ms)**；超过只产空读无效行，甚至 `-d 1` 下有效交付率反而塌到 ~3Hz。
2. **idle 底噪只有 PCIe 一个**：全字段实测（GPU7+GPU3），只有 `pcie_tx/rx_bytes` ~2MB/样本≈**20MB/s @10Hz** 有底噪
   （DCGM/驱动轮询走 PCIe）；**HBM(`dram_active`)、NVLink(449/1011/1012)、SM/计算(1001-1004)、`mem_copy_util`、`fb_used` 全部 idle=0/NA，无底噪**。
3. **想 >10Hz 的轻量路子只有 NVML 经典 `nvmlDeviceGetPcieThroughput`（~25–50Hz，仅 PCIe，无空行）**；HBM/NVLink 更快只能 Nsight/CUPTI（抢计数器）。

## 关键数（引用请回原始日志/CSV；**原始日志在 `logs.tar.gz` 内、CSV 在 `results/`、png 在 `analysis/`**）
| 项 | 值 | 出处 |
|---|---|---|
| **idle 底噪 verdict（全 12 字段）** | 只有 PCITX/PCIRX = FLOOR；其余 10 个字段 CLEAN（0 非零样本） | `results/idle_noise_allmetrics_summary.csv`；`idle_allmetrics_100ms_gpu{7,3}.txt` |
| 有效样本占比 vs 频率 | 5Hz 95% / 10Hz 97% / 20Hz 55% / 100Hz 13% ≈ `min(1,10Hz/freq)` | `idle_pcie_check_*hz.txt`；图 `analysis/task2_validfrac_vs_freq.png` |
| PCIe idle 底噪(每样本) | 合并 1/5/10Hz：TX mean 2.38MB、RX 2.25MB（median≈2.3，σ≈1.3）；全字段复核 TX 2.19/RX 1.96MB | `results/idle_pcie_summary.csv`、`idle_noise_allmetrics_summary.csv` |
| PCIe idle 底噪(带宽@10Hz) | TX ~17 MB/s、RX ~17 MB/s | 同上 |
| DCGM `-d 1`(~345Hz 拉) 实际交付 | ~3Hz clean（99% 行是 0） | `freqtest_1field_1ms_h2dload_gpu6.txt` |
| NVML pcie throughput 交付 | ~24Hz(双读)/~50Hz(单读)，288/288 非零 | `nvml_pcie_probe_gpu6.txt` |
| dcgmi 采集开销 | <0.3%（即便 1000Hz+16 字段） | `experiments/dcgmi_overhead` |

## 文件地图
**原始日志（已打包为 `logs.tar.gz`，解压即区域根目录下述文件）**
- `idle_allmetrics_100ms_gpu{7,3}.txt` —— **全字段 idle 底噪扫描**（12 字段并列，@10Hz；GPU7 长采 350 拍、GPU3 交叉验证 150 拍）
- `idle_pcie_check_gpu45.txt`(1Hz,GPU4+5) / `_5hz` / `_10hz` / `_20hz` / `_100hz.txt` —— 不同频率 idle dmon，字段集 `204,1005,1002,1009,1010,449,1011,1012`（整组都采、当时只析 PCIe）
- `freqtest_{1,16}field_1ms_{idle,h2dload}_gpu6.txt` —— Task3 host 探针原始（带相对时间戳，追踪某 byte 字段的刷新）
- `nvml_pcie_probe_gpu6.txt` —— NVML 替代路径探针输出

**`workloads/`（探针 + 负载，运行产上面的原始日志）**
- `dcgm_refresh_probe.py` + `h2d_loop.py` → `logs.tar.gz` 内 `freqtest_*`（host 探针 + 容器持续 H2D 负载；负载让每个 100ms 窗都有真流量，才能干净数内部刷新率）
- `nvml_pcie_probe.py` → `logs.tar.gz` 内 `nvml_pcie_probe_gpu6.txt`（容器内 pynvml）

**`analysis/`（脚本 + 图）**
- `analyze_idle_noise_allmetrics.py` → `results/idle_noise_allmetrics_summary.csv`（全字段 idle 逐字段 FLOOR/CLEAN 判定；**宿主机** stdlib）
- `analyze_idle_and_freq.py` → `task1_*.png` `task2_*.png`、`results/stats.txt`（**容器内**跑）
- `make_idle_csv.py` → `results/idle_pcie_raw.csv`（每样本原始 PCITX/PCIRX+MB+MB/s+状态）、`results/idle_pcie_summary.csv`（每 频率×方向 统计）（**宿主机** stdlib）
- `task3_plot.py` → `task3_paths_clean_rate.png`

**`results/`（产物）**：`idle_noise_allmetrics_summary.csv`、`idle_pcie_raw.csv`、`idle_pcie_summary.csv`、`stats.txt`、`freqtest_summary_gpu6.txt`（探针汇总＝最后一次运行＝h2dload）

## 怎么复现（环境分工：装依赖进容器，dcgmi 在宿主机）
```bash
# 选一张空卡（显存/util 都 0；GPU0 常被占）
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

# Task1b 全字段 idle 底噪扫描（宿主机采集 + 宿主机 stdlib 判定；无 workload 不进容器）
experiments/dcgmi_pure/collect_idle_allmetrics.sh 7 350 100    # 采：空卡 GPU7
python3 experiments/dcgmi_pure/analysis/analyze_idle_noise_allmetrics.py   # 判定 -> summary.csv

# Task1+2 解析出图（容器；宿主机 pip 会 OOM，exit137）
docker run --rm -v "$PWD":/work -w /work nvcr.io/nvidia/pytorch:25.12-py3 \
  python /work/experiments/dcgmi_pure/analysis/analyze_idle_and_freq.py
# 底噪 CSV（宿主机，纯 stdlib）
python3 experiments/dcgmi_pure/analysis/make_idle_csv.py
# Task3 DCGM 内部刷新率探针（宿主机；idle 版）
python3 experiments/dcgmi_pure/workloads/dcgm_refresh_probe.py 6 3000 experiments/dcgmi_pure
# Task3 加载版 = 先在容器起 h2d_loop.py 制造 PCIe 流量，再 TRACK=1010 TAG=h2dload 跑上面的探针
# Task3 NVML 替代路径（容器 pynvml，需同时有 h2d 负载）
```
- 采集/探针用**物理卡号**（`-i <gpu>`）；本账号 **passwordless sudo**；容器采 profiling 需 `--cap-add SYS_ADMIN`。
- 容器无中文字体：图内文字用英文，中文留 README/CSV/控制台。

## parser 须知（两种日志布局）
- **header 驱动解析**：数据行 `GPU <id> <值...>`，列名取自以 `#Entity` 开头的表头；跳过 `ID  MB/` 单位行。
- 两种列布局：**9 列**（pure：MCUTL/DRAMA/SMACT/PCITX/PCIRX/NBWLT/NVLTX/NVLRX）、**16 字段**（overhead full；PCITX 在第 11 列、DRAMA 第 7）。
- warmup 头 ~2 拍 N/A；超采空读 = **byte 字段返 `0`（与真无流量二义）**、**ratio(DRAMA) 返 `N/A`**。

## 待办 / 未决线索
- **低频欠采隐患**：`PROF_PCIE_*_BYTES` 在 1Hz 下每样本≈一个 ~100ms 窗（~2.9MB），不是整秒累积 → **低频积分字节可能少算**。
  建议专门测：已知总传输量下，在 1/10Hz 各积分 PCITX，比对实际搬运 GB，确认是否需要"按内部率而非采样率"折算。
- **20/100Hz 过采样样本**已在 `results/idle_pcie_raw.csv` 里（`used_in_task1_baseline=False`），尚未单独导出对照 CSV。
- **NVML GPM 直调**（`nvmlGpmSampleGet`）能否突破 100ms 只据官方文档判定，未在本机实测证伪。
