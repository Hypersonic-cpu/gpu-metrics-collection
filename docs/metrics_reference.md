# 驱动侧 Metric 监控参考手册（Driver-side Metric Monitor）

> 采集 GPU 接口 metric 的**驱动侧**（非侵入式旁路）手册，**分两部分**：
> - **第一部分 · NVML 与 DCGM** —— GPU 侧 metric（HBM/SM/PCIe/**GPU 侧 NVLink**/显存…）的字段语义与用法。
>   ⚠️ 其中 **NVSwitch 侧吞吐字段在本机 H100 是坏的**（负载下恒 0，NVIDIA 已知
>   [issue #236](https://github.com/NVIDIA/DCGM/issues/236)）——交换机侧交给第二部分。
> - **第二部分 · nvswitch_traffic（自研工具）** —— 绕过 DCGM 直连 **NSCQ** 读 NVSwitch 每端口流量，
>   **填补 DCGM/NVML 在 NVSwitch 侧的空缺**（`tool/nvswitch_traffic/`）。
>
> 本手册只讲**工具本身 + 字段语义/用法**；具体 workload 的"点亮"实验（cudaMemcpy 三类、L2 影响、
> 采样频率/开销、NVSwitch 负载实测…）留在 `experiments/*/README.md`，本手册只**引用其结论**、不重复细节。
>
> 最终依据：① NVIDIA 官方文档（每条结论附 reference URL）；② 本机实测（`docs/host_probe/` dump + `experiments/` 日志）。
> 机器 `zkrh-58`：8× H100 80GB HBM3（HBM3 峰值≈3350 GB/s），Driver 590.48.01 / CUDA 13.1，DCGM 4.2.3；
> **NVLink-switched 节点**：**4 颗物理 NVSwitch**（= 4 个 PCIe 设备 05/06/07/08；`dcgmi discovery -l` 报"12"是逻辑口径）
> + Fabric Manager active，每 GPU 18 条 NVLink 全接到 NVSwitch。网上第三方说法一律回官方/本机实测二次确认后才写入。

---

# 第一部分 · NVML 与 DCGM（GPU 侧 metric）

> 覆盖 HBM / SM / PCIe / **GPU 侧 NVLink** / 显存 等。**NVSwitch 侧（交换机端口视角）DCGM 在本机不可用**
> （见 §5.2 与 [issue #236](https://github.com/NVIDIA/DCGM/issues/236)），要看过交换机的流量用**第二部分**的自研工具。

## 0. NVML 与 DCGM：是什么、什么关系、怎么采

### 0.1 DCGM 的数据来自 NVML
DCGM **自己不读硬件**，角色是"**调用 NVML → 缓存 → 分发（+改名/换量纲）**"。真正读 GPU 硬件计数器的实现在
闭源的 `libnvidia-ml.so.1`（随驱动发布，DCGM 运行时 `dlopen` 加载）里，不在 DCGM 开源树内。

- **NVML**（NVIDIA Management Library，`libnvidia-ml.so.1`）：随驱动附带的底层 C 库，是**真正**读硬件
  遥测/计数器的那一层。它有**两套**接口：
  - **经典设备查询**（`nvmlDeviceGetUtilizationRates` 等，即 `nvidia-smi` 那套）：只给粗粒度的
    `gpu%` / `memory%`，分不清各功能单元；这套里没有 `dram_active`/`sm_active`。
  - **GPM（GPU Performance Monitoring）**：Hopper（H100）起 NVML 新增的一组 API
    （`nvmlGpmSampleGet` / `nvmlGpmMetricsGet` / `nvmlGpmQueryDeviceSupport`），把 GPU 拆到**功能单元级**
    ——SM/Tensor/FP 各管线活跃度、**DRAM 带宽利用率**、PCIe/NVLink 吞吐、NVDEC/NVJPG…（映射见 §0.3）。
- **DCGM**（Data Center GPU Manager）：构建在 NVML 之上的更高层套件。带常驻守护进程 `nv-hostengine`，
  按每个字段的 watch 间隔周期性调用上面的 NVML 函数、把结果缓存；客户端（`dcgmi` CLI、`pydcgm`、
  `dcgm-exporter`）向它取数。DCGM 做的是"调 NVML + 缓存 + 分发 + 改名/换量纲"，profiling 数值不由它计算。
- 能力关系：在 H100 上 **DCGM profiling ⊆ NVML GPM**（DCGM 不产生新指标，只是把 GPM 指标包成易用字段 +
  缓存 + 多卡/多节点管理）；相对**经典 NVML**（`nvidia-smi`）则 **GPM/DCGM ⊃ 经典 NVML**。
  → NVLink 带宽两边都能给（NVML 作交叉校验）；`dram_active`/`sm_active` 经典 NVML 没有、NVML GPM 有。

### 0.2 profiling 字段在不同 GPU 代次的两条采集路径
DCGM profiling 字段族（`dram_active` 1005、`sm_active` 1002、`pcie/nvlink_*_bytes` 1009-1012…）的采集路径由
DCGM 源码分支决定（`DcgmCacheManager.cpp:14044`：
`if (!entityKeySupportsGpm) { /* Assume that DCP is updating this field. */ break; }`）：

| GPU 代次 | 采集路径 | 说明 |
|---|---|---|
| **Hopper / H100（本机）及更新** | **NVML GPM** | `EntityKeySupportsGpm()==true`：DCGM 调 `nvmlGpmSampleGet`×2 + `nvmlGpmMetricsGet`，字段值来自 NVML。 |
| Volta / Turing / Ampere 等旧卡 | 闭源 DCP 模块 | `==false`：走 PerfWorks 硬件计数器（`modules/profiling/` 只有头文件、实现闭源）。这类卡上"NVML 没有、DCGM 独有"成立。 |

- 字段→GPM 枚举的映射在源码 `dcgm_fields.cpp:6524` 的 `DcgmFieldIdToNvmlGpmMetricId()`（表见 §0.3）。
- 比值类字段：NVML GPM 给 0–100 百分比，DCGM 侧 `value /= 100.0`（`DcgmGpmManager.cpp:325`）存成 0–1 ratio。
  同一底层数：NVML GPM 里是 `87.0`，`dcgmi` 里显示 `0.87`。

▎ H100 上 `dram_active`/`sm_active` 这批 = **NVML GPM 指标改名并 ÷100**，底层数据源是 **NVML GPM**，DCGM 只包一层。
▎ "NVML 没有、DCGM 独有" 仅在 Volta/Turing/Ampere 那些走闭源 DCP 的旧卡上成立。

### 0.3 NVML GPM 指标 ↔ DCGM profiling 字段映射（H100 生效）
源码 `dcgm_fields.cpp:6524` `DcgmFieldIdToNvmlGpmMetricId()`；NVML 枚举 `sdk/nvidia/nvml/dcgm_nvml.h`
（如 `NVML_GPM_METRIC_SM_UTIL=2` "Percentage of SMs that were busy"、`NVML_GPM_METRIC_DRAM_BW_UTIL=10`
"Percentage of DRAM bw used vs theoretical maximum"）：

| DCGM 字段（别名） | ID | NVML GPM 枚举 |
|---|---|---|
| gr_engine_active | 1001 | `NVML_GPM_METRIC_GRAPHICS_UTIL` |
| sm_active | 1002 | `NVML_GPM_METRIC_SM_UTIL` |
| sm_occupancy | 1003 | `NVML_GPM_METRIC_SM_OCCUPANCY` |
| tensor_active | 1004 | `NVML_GPM_METRIC_ANY_TENSOR_UTIL` |
| **dram_active** | **1005** | **`NVML_GPM_METRIC_DRAM_BW_UTIL`** |
| fp64 / fp32 / fp16_active | 1006 / 1007 / 1008 | `NVML_GPM_METRIC_FP64/FP32/FP16_UTIL` |
| pcie_tx / rx_bytes | 1009 / 1010 | `NVML_GPM_METRIC_PCIE_TX/RX_PER_SEC` |
| nvlink_tx / rx_bytes | 1011 / 1012 | `NVML_GPM_METRIC_NVLINK_TOTAL_TX/RX_PER_SEC` |
| hmma / imma / dfma / int_util | 1013-1016 | `NVML_GPM_METRIC_HMMA/IMMA/DFMA_TENSOR_UTIL`、`INTEGER_UTIL` |
| nvlink_lN_tx / rx_bytes | 1040+ | 每链路 `NVML_GPM_METRIC_NVLINK_Lx_TX/RX_PER_SEC`（`NvLinkLinkIndexToGpmMetricId()`）|

> 本机 `dram_active`(1005) = NVML GPM 的 `DRAM_BW_UTIL` ÷100；`sm_active`(1002) = GPM `SM_UTIL` ÷100。

### 0.4 GPM 工作机制（决定了 §3 的 ~100ms 下限）
GPU 内部是一批**只增不减的硬件计数器**；GPM 用**两张快照相减**取值：
`nvmlGpmSampleGet`(A) → 隔一小段 → `nvmlGpmSampleGet`(B) → `nvmlGpmMetricsGet(A,B)` 把计数增量除以理论最大值
算出利用率。DCGM 的 `DcgmGpmManager` 维护 sample 队列照此模式封装（源码注释：*"nvmlGpmMetricsGet
needs two samples spanning an update interval"*，`DcgmGpmManager.cpp:30`）。
→ 取值**需要一个采样窗口**：窗口太短（<~100ms）计数器还没在内部刷新、增量≈0，结果为 0/N/A。这是 §3
"实际最小 ~100ms" 的物理根源，也是高频返回 0/空行的来由；此下限来自 NVML/硬件的内部刷新率，非 DCGM 施加。

- 参考：
  - DCGM Feature Overview <https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html>
  - NVML GPM Functions（GPM API 定义 + 100ms 说明 + "Hopper or newer"）
    <https://docs.nvidia.com/deploy/nvml-api/group__nvmlGpmFunctions.html>
  - DCGM 开源源码（本机 clone `~/Repos/DCGM`，函数/分支名为稳定标识，行号对应本次核对的版本）：
    `dcgmlib/src/dcgm_fields.cpp:6524`（字段→GPM 映射）、`dcgmlib/src/DcgmCacheManager.cpp:14044`（GPM/DCP 分支）、
    `dcgmlib/src/DcgmGpmManager.cpp:184/301/325`（`nvmlGpmSampleGet`/`nvmlGpmMetricsGet`/÷100）、
    `sdk/nvidia/nvml/dcgm_nvml.h:12013,12021`（NVML GPM 枚举定义）。

**两者都是旁路（out-of-band）采样**：GPU 驱动维护这些计数器，采集端只是去读——**不改被测程序、
不用把库链接进被测程序、不需要它配合**。所以采集器可作为独立进程/容器：被测程序在别的 repo/容器里
跑，采集器旁路采样，二者解耦——这正是本工具的架构。DCGM 更是天然 daemon 模型（`nv-hostengine`
常驻、`dcgmi` 只是客户端）。

**多卡要显式对齐目标卡**：DCGM/NVML 天然支持多卡与 GPU group，单卡就是选一张。但采集时**必须知道
"这次 job 用了哪几张卡"**，只采 job 占用卡（靠物理 id / UUID / `CUDA_VISIBLE_DEVICES` 对齐），
否则空闲卡的 0 值会稀释/污染分析。本机 MIG Disabled，选卡简单。
⚠️ **应用容器内部会把 `device=1,2` 重编号为 0/1，采集器必须用物理编号/UUID 指定**
（对齐表见 `docs/host_probe/dcgmi_discovery.txt`），否则采错卡。

**"trace" 一词先说清**：本工具能给的 trace = **固定间隔的时间序列采样**（DCGM 最细 100 ms，
NVML 由轮询间隔定），**不是** kernel 级、微秒级的执行轨迹。
- "某段程序运行期间 HBM/NVLink 带宽随时间的曲线" → 能做（就是时间序列）。
- "第 N 个 kernel 用了多少带宽" → NVML/DCGM 做不到，需 Nsight/CUPTI（后续阶段，见 §6）。

---

## 1. 目标 metric → 该用哪个字段（速查表）

> 本节只做"目标→选哪个字段"的路由；**每个字段完整的语义、单位、权限、坑，在 §5 对应小节**。

| 目标 | 推荐字段 (id) | 选择理由（一句话）| 详见 |
|---|---|---|---|
| **HBM 带宽利用率/trace** | `dram_active` (1005) | 唯一的 HBM 带宽利用率信号；profiling，需 SYS_ADMIN | §5.1 |
| **NVLink 带宽（聚合，一个数）** | `nvlink_bandwidth_total` (449) | device 字段、直读无需差分、大概率免 SYS_ADMIN；**单位 MiB/s**（§5.2.1）| §5.2 |
| **NVLink 带宽（tx/rx 拆分）** | `nvlink_tx_bytes`(1011)/`nvlink_rx_bytes`(1012) | 要方向拆分用它（profiling，需 SYS_ADMIN）；NVML `NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX/RX` 可交叉校验 | §5.2 |
| **NVLink 带宽（每链路）** | `nvlink_bandwidth_lN`(440–449) 或 `nvlink_lN_tx/rx_bytes`(1040+) | 看单链路热点；device 版(440–449) 免权限直读，**单位未单独标定**（大概率同 449 的 MiB/s，§5.2.1）| §5.2 |
| **SM 利用率（用了多少 SM）** | `sm_active` (1002) | ≠ nvidia-smi 的 GPU-Util | §5.4 |
| **SM 占用深度** | `sm_occupancy` (1003) | 驻留 warp 数/硬件上限，看"忙到多满" | §5.4 |
| Tensor Core 活跃 | `tensor_active`(1004)、`*_hmma/imma/dfma`(1013-1015) | Tensor 管线活跃占比 | §5.4 |
| 计算管线占比 | `fp16/fp32/fp64_active`(1006-1008)、`integer_active`(1016) | 各数值管线活跃占比 | §5.4 |
| PCIe 带宽 | `pcie_tx_bytes`(1009)/`pcie_rx_bytes`(1010) | bytes/秒速率；`pcie_*_throughput`(200/201) 已废弃、勿用 | §5.3 |
| 显存**容量**占用 | `fb_used`(252)/`fb_total`(250)/`fb_free`(251) | 是容量不是带宽；用于判 job 是否真跑起来 | §5.5 |
| GPU 整体活跃 | `gr_engine_active`(1001)；NVML `nvmlDeviceGetUtilizationRates().gpu` | nvidia-smi 的 "GPU-Util" = NVML gpu util | §5.4 |

> 权限速记：id≥1000 的 profiling 字段容器内需 `--cap-add SYS_ADMIN`；`449`/`440-449`/`252` 等 device 字段免额外权限（详见 §4）。
> 本机确认：以上 1001–1075 全部字段在 `docs/host_probe/dcgmi_fields_list.txt` 中存在，
> 即 H100 + DCGM 4.2.3 这套组合原生支持这些 profiling 字段。字段可用性避坑清单见 §7。
> 底层数据源（H100 上 profiling = NVML GPM）见 §0。

- 字段权威定义：DCGM Field Identifiers
  <https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-field-ids.html>
- Profiling API：
  <https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-profiling.html>

---

## 2. 采集命令：`dcgmi dmon` 怎么用（参数逐个说明）

核心命令形态：
```
dcgmi dmon -e <field_ids> -i <entities> -d <interval_ms> [-c <count>]
```

| 参数 | 含义 | 取值 / 说明 |
|---|---|---|
| `-e` | 要采的**字段 id 列表**（逗号分隔）| 如 `-e 1005,1011,1012,1009,1010,449`；id 见 §1 表与 §7 |
| `-i` | 采**哪些实体** | GPU：`-i 0,1`（物理 id，非容器内重编号）；NVSwitch：`-i nvswitch:*`（见 §5.2）。**实体类型用错取不到值，不是字段坏** |
| `-d` | **采样间隔（毫秒）** | 默认 1000；有效地板 ~100ms（NVML GPM，非 DCGM 策略）；再快无效（见 §3.1）|
| `-c` | 采**几拍后退出** | 不给则持续采到 Ctrl-C。⚠️ 短测 `-c 3` 会被 warmup 坑（见下）|

**输出格式**：一行一个"实体 × 拍"，表头是**字段短名**（`DRAMA / SMACT / NVLTX / NVLRX / PCITX /
PCIRX / NBWLT / MCUTL / FBUSD`…）——parser 需按 **短名 ↔ id ↔ 全名** 映射。列头的单位标注可能被截断
（如 PCIe 列头显示 `MB/`，但实际数值是 **bytes/秒**，别按字面读）。

★ **warmup：byte/profiling 字段头 ~2 拍是 N/A。** `dcgmi dmon` 里 profiling / `*_bytes` 类字段
开头约 2 个采样返回 `N/A`，第 3 拍起才出真实值（样例 `docs/host_probe/dcgmi_dmon_idle_example.txt`
实证：`dram_active`/`sm_active`/`pcie_tx/rx_bytes`/`nvlink_tx/rx_bytes` 头 ~2 拍 N/A，之后 DRAMA/SMACT=`0.000`、PCITX/RX 出 idle 值）。含义：
- 采集**至少跑 5 拍**再信；短测 `-c 3` 容易把 warmup 误判成"字段坏了"。
- 采集器 / parser / validate **必须容忍并丢弃开头的 N/A**，不得据此判失败或空跑；
  运行中偶发单点 N/A 属采样抖动，按缺失值处理即可。

★ **byte 字段已经是速率，不用差分。** `dcgmi dmon` 输出的 `pcie_*_bytes` / `nvlink_*_bytes` 就是
**bytes/秒**，与采样间隔无关——实证：同一负载下 `nvlink_tx_bytes` 原始值在 **1Hz 与 10Hz 都 ≈ 4.07e11**
（若是"每间隔字节"，10Hz 应约为 1/10），PCIe 也都 ≈ 5.9e10，与官方定义 "rate … in bytes per second"
一致。→ parser 直接 `GB/s = 值 / 1e9`。**真正需要差分的是 NVML 的累计计数器**
（`NVML_FI_DEV_NVLINK_THROUGHPUT_*`，那是另一条路，别混）。

**NVML 侧怎么读**：`nvmlDeviceGetFieldValues` 读字段（如 NVLink `NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX/RX`，
是**累计字节**，要按轮询间隔差分）；`nvmlDeviceGetUtilizationRates`（gpu/memory util）、显存/功耗/温度
用各自的 `nvmlDeviceGet*`。无固定周期，采样率由你的轮询循环决定（见 §3）。

---

## 3. 采样频率：可用范围与如何选

### 3.1 采样频率的真实下限：~100ms（NVML GPM 的地板，非 DCGM 策略；官方 + 源码 + 本机实测）
需分清**两个互相独立的频率**：

| 频率 | 含义 | 谁控制 | 真实下限 |
|---|---|---|---|
| `dcgmi dmon -d` delay | 客户端多久**读一次缓存并打印** | dcgmi CLI | **1 ms**（源码 `CommandLineParser.cpp:2747` 帮助文本 "Minimum value = 1 msec"；`:2818` `delay<1` 才报错）——但只是重复读缓存 |
| host engine 采样 updateFreq | 后台多久**真正采一次样进缓存** | watch API / NVML GPM | **~100 ms**（见下） |

- **`-d` 能填 1ms、不报错，但那是假象**：若底层 updateFreq 还是 100ms，你只是把同一个缓存值**重复打印很多遍**，
  数据并不会真的以 1ms 刷新。**所以"最小 1ms"不是有效采样率。**
- **真实地板 ~100ms 来自 NVML GPM，不是 DCGM 加的**：DCGM 的 GPM 代码里**没有**硬编码 100ms；那个下限是
  `dcgmProfMetricGroupInfo_t.minUpdateFreqUsec`，由**驱动/NVML 按 metric group 报上来**
  （源码 `modules/profiling/dcgm_profiling_structs.h:65`）。NVIDIA 官方 NVML GPM 文档原话：
  ▎ *"The interval between two nvmlGpmSampleGet() calls should be greater than 100ms due to the internal sample refresh rate."*
  （<https://docs.nvidia.com/deploy/nvml-api/group__nvmlGpmFunctions.html>）
  机制见 §0.4：GPM 靠两张快照相减，GPU 内部计数器 ~100ms 才刷新一次，窗口比它短就没有新信息可算 → 返回 0/N/A。
- **绕过 DCGM 直接调 NVML GPM 也突破不了 100ms**——面对的是同一个底层机制、同一个地板。DCGM 只是这个机制的
  一个调用者，不是限制的来源。想要 <100ms 只能换赛道（Nsight Compute / CUPTI PM sampling，侵入式，见 §6）。
- **本机实测把"有效上限"也钉死在 10Hz**（`experiments/dcgmi_pure`，纯 idle 统计 byte 字段空读比例）：
  `-d` 虽能设到 1ms，但 profiling/byte 字段**内部只有 ~100ms(10Hz) 才刷新**，超过就在真数据间插空行——

  | 频率 | 间隔 | PCITX 有效样本占比 |
  |---|---|---|
  | 5Hz | 200ms | 95% |
  | **10Hz** | **100ms** | **97%（=内部率，刚好全有效）** |
  | 20Hz | 50ms | **55%**（~一半空行）|
  | 100Hz | 10ms | **13%**（只剩 ~1/10 真数据）|

  空读的 signature：**byte 计数器→`0`（与"真没流量"二义，最坑）、ratio(DRAMA)→`N/A`**。
  → **profiling/byte 字段别设超过 10Hz(100ms)**：更快不增分辨率、只产空行、还污染 byte 曲线。
- **性能开销不是限制频率的原因**：`experiments/dcgmi_overhead` 实测，即便 `-d 1`(1000Hz)+16 字段，
  对 compute/HBM/NVLink 三类 workload 的拖慢都 **<0.2%（噪声内）**。
  → **限制频率的是数据有效性，不是开销**；默认 **100ms 既开销安全又数据有效，是最优点**。
- **NVML**：无固定周期，**由轮询循环决定**；底层利用率类历史上有 ~1s 平滑窗，NVLink 是累计计数器
  所以"差分粒度 = 你的轮询间隔"。
- **两者都拿不到 kernel 级（微秒级）trace**（见 §0 "trace" 与 §6）。

### 3.2 Profiling 并发 / 多路复用限制
- 部分 profiling 指标需多趟采集，硬件无法一次全采；DCGM 会**自动 multiplexing**（统计采样 + 分组），
  代价是每字段每 100ms 未必都有值。**每遍少放几个字段 → 每字段都能干净地采到**（fidelity ↔ 重跑次数 权衡）。
- DCGM profiling 与 **Nsight Systems/Compute、CUPTI 互相冲突**（同一 GPU 不要同时开）。该限制在
  **A100 及更早**更严格，**Hopper（本机 H100）** 对常用字段组有放宽 → 对我们（只用 DCGM）无影响。
  参考：DCGM Feature Overview（同 §0 链接）。

---

## 4. 权限：要不要 sudo / SYS_ADMIN

| 字段类别 | 要额外权限吗 | 说明 / 例子 |
|---|---|---|
| **NVML 基础字段** | 否 | 容器 `--gpus all` 能看到设备即可 |
| **DCGM 非 profiling（device）字段** | 否 | id<1000：利用率、显存、功耗、温度、NVLink 错误计数、`449 nvlink_bandwidth_total`、`204 mem_copy_util`、`200/201 pcie_*_throughput` 等 |
| **DCGM profiling 字段**（H100 上实为 **NVML GPM**，见 §0.2）| **需 root 或 `--cap-add SYS_ADMIN`** | `dram_active`/`sm_active`/`pcie_*_bytes`(1009-1012)/`nvlink_*_bytes`/`tensor_active`… 权限门控在**底层 GPM/profiling 计数器访问**（NVML GPM 与旧卡的闭源 DCP 都需要），DCGM 只是转发这条要求，不是 exporter 特有的 |

- **本机现实**：宿主机已装 DCGM 4.2.3（`dcgmi`/`nv-hostengine` 在 `/usr/bin`），`nvidia` runtime 就绪。
  profiling 字段在本宿主机**无需额外 sudo** 即可采（实测 idle 返回合法 `0` 而非权限错误，见 §7）。
  → **最省事：采集器直接跑宿主机（root，profiling 全开）**，被测程序留在它自己的容器里；
  容器化采集器（`docker run --gpus all --cap-add SYS_ADMIN ...`）作为可选方案，数据与宿主机一致。
- 参考：dcgm-exporter README <https://github.com/NVIDIA/dcgm-exporter>；
  DCGM-Exporter 文档 <https://docs.nvidia.com/datacenter/dcgm/latest/gpu-telemetry/dcgm-exporter.html>

---

## 5. 各接口指标详解（理解 + 本机更新）

> §1 速查表按目标选定字段后，这里是每个字段的完整语义、单位、权限、坑与本机实测。

### 5.1 HBM 带宽：用 `dram_active`，别用 Memory-Util（`mem_copy_util` 204）

**要衡量 HBM 带宽利用率，用 DCGM `dram_active`(1005)**：官方定义 "the ratio of cycles the device
memory interface is active sending or receiving data"（0–1），随带宽压力线性变化。
（本机 H100 上其底层数据源是 NVML GPM 的 `DRAM_BW_UTIL`÷100，见 §0.2。）近似换算
`achieved_BW ≈ dram_active × peak_HBM_BW`（H100 SXM HBM3 峰值约 3.35 TB/s）——注意这是**近似**：
`dram_active` 是**活跃周期占比**，不是精确 GB/s。

**不要用 NVML `nvmlUtilization_t.memory`**（即 nvidia-smi 的 "Memory-Util"，DCGM
`mem_copy_utilization`(204)）当 HBM 带宽利用率：
- 它是"显存被读写的**时间占比**"，不是"用了峰值带宽的百分之几"——是把**所有** HBM 读写活动
  （含**计算 kernel 的 load/store，不只是 memcpy**）混成的**一个粗粒度 HBM 总带宽利用率标量、
  无接口拆分**。名字里的 "copy" 有误导，别被带偏。
- 本机实测 ≈ `dram_active × 100`（粗粒度整数百分比，随本卡 HBM 带宽占用单调）。
- **本工具为何不采它**：接口性能 profile 要的是**按接口拆分的带宽**——HBM 用 `dram_active`、
  PCIe 用 `pcie_bytes`(§5.3)、NVLink 用 `nvlink_bytes`/449(§5.2)。204 把这些全糊成一个 HBM 标量、
  又比 `dram_active` 更粗，**任何一路都替代不了、也拆不出接口**，故不纳入。

**为什么 `dram_active×100` ≠ `mem_copy_util`（结构性差异，不是噪声）**：
**`MCUTL ≥ DRAMA×100` 恒成立**，且负载越高差得越多——因为"内存控制器在忙"（含 latency/行激活/刷新/
半宽事务等**不传数据**的周期）比"数据总线真在传 beat"覆盖更长时间，且 MCUTL 是整数 %、会**早早饱和到 100**。

| 维度 | `dram_active`(1005, DRAMA) | `mem_copy_util`(204, MCUTL) |
|---|---|---|
| 来源 | **NVML GPM `DRAM_BW_UTIL`÷100**（H100；旧卡才走闭源 DCP，见 §0.2）| NVML `nvmlUtilization_t.memory`（= nvidia-smi Memory-Util）|
| 官方定义 | "ratio of **cycles** the device memory interface is active…" | "percent of **time** during which global memory was being read/written"（采样窗 1/6–1 s）|
| 物理含义 | **周期级、带宽正比**：≈ 实测带宽/峰值 | **时间占空比**：内存有没有在被读写，**不按用了多少带宽加权** |
| 分辨率 | 浮点、细 | 整数 %、粗，且会**饱和到 100** |
| 权限 | profiling（需 SYS_ADMIN）| device 字段（免额外权限）|

NVIDIA 自己也把 DRAM_ACTIVE 描述为 "similar to DCGM_FI_DEV_MEM_COPY_UTIL and **could be more precise**"
——正是此现象。→ **量 HBM 带宽用 DRAMA，别用 204。** 本机三场景（H2D/D2H、卡间 D2D、单卡 D2D）
DRAMA×100 vs MCUTL 的逐项数值与"差几个百分点"实测见 `experiments/cumemcpy/README.md`。

⚠️ **L2 会挡住 HBM 指标——别拿 `dram_active` 判断"有没有在搬数据"**：
`dram_active`/`mem_copy_util` 是 **HBM** 指标。若拷贝/kernel 的 working set 整块装进 L2（H100 L2=50MB），
HBM 流量被 L2 挡掉，`dram_active` 会大幅下降甚至趋 0，**即便实际带宽更高**。所以它只回答"HBM 忙不忙"；
一个 cache-resident 的操作会让 `dram_active≈0` 却在猛搬数据。→ 判"有没有在搬"要认**接口 byte 计数器**
（`pcie_bytes`/`nvlink_bytes`/449，在接口处计量、**与 L2 无关**，见 §5.2/§5.3）。
L2-fit 对照实验（2GB busts L2 vs 16MB fits L2、读写不对称现象）见 `experiments/cumemcpy/README.md`。
- L2 不可关（NVIDIA 工程师："ALL accesses to DRAM take place through the L2 cache. And the L2
  cache cannot be disabled"），但装进 L2 就会挡掉 HBM——这与"能不能绕过 L2"是两码事。

**精确到 GB/s / per-kernel 带宽**：NVML/DCGM 都做不到，需 Nsight Compute / CUPTI（见 §6）。

参考：NVML `nvmlUtilization_t` <https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html>
（PDF 版 <https://docs.nvidia.com/deploy/pdf/NVML_API_Reference_Guide.pdf>）；
DCGM Field Ids（同 §1 链接）。

### 5.2 NVLink 带宽：三条路的区别

- **DCGM device 路**（★负载实测可用，最省事、大概率免权限）：
  - 字段：`nvlink_bandwidth_total`(449) 聚合，或 `nvlink_bandwidth_lN`(440–449) 每链路。
  - 语义：**已是速率、无需差分**，聚合为**双向合计**，无 tx/rx 拆分；数的是 **payload（不含协议开销）**。
  - ⚠️ **单位是 MiB/s，不是 MB/s**（dmon 表头印 `MB/`）：`GB/s = 印出值 × 1048576 ÷ 1e9`。
    照 `÷1000` 算会**系统性低 4.63%**。实测标定见 §5.2.1。
  - 属 **device 字段（id<1000，非 profiling）** → **大概率免 `--cap-add SYS_ADMIN`**（容器内待最终确认）。
  - 负载实测：单向 peer-copy **~405 GB/s**（`experiments/nvlink_bench/logs/dcgmi_dmon_449_steady_67.txt`）。
- **NVML 路**（轻量、无需额外权限；**有 DCGM 拿不到的东西**）：
  - 字段：`NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX/RX`(138/139)、`..._RAW_TX/RX`(140/141)，
    经 `nvmlDeviceGetFieldValues` 读取（旧的 `nvmlDeviceGetNvLinkUtilizationCounter` 已废弃）。
    `scopeId` = link id，`0xFFFFFFFF` = 整卡所有链路。
  - 语义：**累计字节计数器**，要自己按轮询间隔**差分**得区间带宽。
    **一格 = 1024 bytes（实测精确，见 §5.2.1）**，与 `nvml.h` 注释的 KiB 一致。
  - ★ **`RAW` 是 DCGM 完全没暴露的一对**：`nvml.h` 注 *"Data + protocol overhead"*，
    而 `DATA` 只有 payload。⇒ **`RAW ÷ DATA` = 协议开销比例，绝对字节口径，不依赖任何峰值假设**。
    本机 DMA 实测 **1.0635（开销 6.35%）** —— 这是目前唯一能直接量协议开销的路子。
  - 参考：NVML NvLink 组 <https://docs.nvidia.com/deploy/archive/R510/nvml-api/group__NvLink.html>。
- **DCGM profiling 路**（tx/rx 拆分，与其它 profiling 字段统一采集）：
  - 字段：`nvlink_tx_bytes`(1011)/`nvlink_rx_bytes`(1012) 聚合，或 `nvlink_lN_*`(1040+) 每链路。
  - 语义：DCGM 已按采样区间给出 **bytes/秒 速率**（无需差分）；单链路带宽 = RX+TX。
  - ⚠️ **数的是 payload、不含协议开销**（与 449 同口径）。判据：SM 远程读时 `NVLTX=0`
    （读请求包走 TX 但零数据字节）；同一负载程序自测 408 / 读到 405.2，含协议应为 434。
  - 属 profiling 字段 → 容器内采集需 `--cap-add SYS_ADMIN`（§4）。

选择建议：
- **主看总带宽 + 想少权限** → `nvlink_bandwidth_total`(449)（device，直读，大概率免 SYS_ADMIN；单位 MiB/s，§5.2.1）。
- **要 tx/rx 方向拆分** → profiling `nvlink_tx/rx_bytes`(1011/1012)（需 SYS_ADMIN）。反正 HBM/SM 也要 profiling，一把梭。
- NVML 的 NVLink 计数器留作**交叉校验**，且 `RAW` 那对是 DCGM 拿不到的（协议开销）。
  三者可并采，单位换对之后**实测自洽到 0.05%**（449×1048576/1e9 = 405.4 ⟷ 1011+1012 = 405.2）。
- NVLink 属**片间**互联，不是"片上网络"（片上 NoC 见 §5.4）。

#### 5.2.1 三条路的单位标定（实测，换机器要重标）

标定脚本 `experiments/nvlink_bench/workloads/nvml_counter_units.py`（容器内跑，两张空卡）：
`calib` 搬定量字节前后各读一次计数器 → 定"一格几字节"；`sustain` 自差分 → 和宿主机 `dcgmi dmon` 对照。

| 事实 | 数值 | 怎么得到的 |
|---|---|---|
| NVML `DATA_TX` 一格 | **1024 bytes**（精确）| 搬 42,949,672,960 B，计数器涨 41,943,040 格 → `1024.000` |
| NVML `DATA_TX` 差分 ÷ 程序自测 | **1.0000** | `sustain` 模式，计数器本身无偏 |
| DCGM 449 印出值的单位 | **MiB/s** | 真值 405.2 GB/s → 应印 386,400，实测印 **386,586**（+0.05%）；若按 `KiB/s÷1000` 该印 395,674（−2.3%）|
| NVML `RAW_TX ÷ DATA_TX`（DMA）| **1.0635** | calib 与 sustain 两种模式一致到 5 位 |

⚠️ **DCGM 源码与本机 binary 在这里不一致，如实记录**：`dcgmlib/src/DcgmCacheManager.cpp` 的
`ReadAndCacheNvLinkBandwidth()` 里那行是 `valueDbl /= 1000.0;  /* Convert to KiB/sec -> MiB/sec */`
（v4.2.3 与 v4.6.0 两个 tag 都写 1000）—— **注释是对的**（KiB→MiB 该除 1024），**代码写的 1000
与本机 4.2.3 binary 的实际行为不符**。所以上表是**实测标定**、不是源码推导。
`dcgm_fields.cpp` 注册的 `DCGM_FIELD_UNIT_BW_MBPS` 只是显示标签，链路上再无其它缩放。

复现性：GPU 6↔7（2026-07-31）与 GPU 2↔3（2026-07-23）同一负载**逐位一致**
（449 原始 386586 vs 386587；1011 405.17 vs 405.20）。

#### 5.2.2 链路速率（**不是吞吐**）：NVML `NVLINK_GET_SPEED`(164) = 26562 MBps/link ★

前面三条路数的都是"实际搬了多少"；这一条是"这条链路每秒**能**搬多少"，即
`nvidia-smi nvlink --status` 印出来的那个数。它是**唯一一个能从本机拿到链路速率的字段**。

| 字段 | id | 本机结果 |
|---|---|---|
| `NVML_FI_DEV_NVLINK_SPEED_MBPS_L0..L11` | 84–89 / 132–137 | ❌ **全部 `rc=3` NOT_SUPPORTED** |
| `NVML_FI_DEV_NVLINK_SPEED_MBPS_COMMON` | 90 | ❌ **`rc=3` NOT_SUPPORTED** |
| **`NVML_FI_DEV_NVLINK_GET_SPEED`** | **164** | ✅ `scopeId` = link id，18 条**全部 raw = 26562** |

⇒ 想读链路速率**必须用 164 并逐链路给 scopeId**；那组名字更直白的 `SPEED_MBPS_L*` 在 H100 上是死的。
`nvidia-smi nvlink --status -i 0` 印的 18 行 `26.562 GB/s` 就是它 ÷1000。
**整卡单方向 = 18 × 26562 MBps = 478.116 GB/s**。

⚠️ **单位陷阱：164 是十进制 MBps，和前面那些吞吐计数器不是一套单位** ——
同一个 NVML/DCGM 生态里，`THROUGHPUT_*` 计数器是**二进制**（一格 1024 B，449 印出来是 MiB/s，§5.2.1），
而 164 是**十进制**（÷1000 得 GB/s）。依据：若按二进制读，18 × 26562 MiB/s = **501.3 GB/s**，
与 `experiments/nvlink_bench` 用已知带宽负载反推的 476.9–478.1（11 行，散度 0.25%）差 4.8%；
按十进制读得 478.116，落在反推区间里。**别看见 "MBps" 就统一按一种进制换算。**

补两条实测事实：

- raw 是**整数** 26562，表达不了小数位；`nvidia-smi` 印 `26.562`。
- `nvmlDeviceGetNvLinkVersion()` 在本机 H100 上返回 **6**，**不是**市场名的 "NVLink 4.0"。
  这是 NVML 自己的协议版本号，**别拿它当代际号用**。

复现（只读，不占显存，任意一张卡）：

```bash
cd experiments/nvlink_bench
docker run --rm --gpus '"device=6"' -v "$PWD/workloads":/w -w /w \
  nvcr.io/nvidia/pytorch:25.12-py3 python /w/nvml_link_speed.py
```

> 这个 478.116 与 NVIDIA 官方规格 450 GB/s/方向（18×25）之间差 6.25%，以及"哪个数该当哪个计数器的
> 百分比分母"，是单独一件事 —— 论证、四组合拟合表和实测证据在
> `experiments/nvlink_bench/BANDWIDTH_CEILING.md`，待拍板项在 `PLAN.md §3`。
> **本节只记录"字段报了什么、单位怎么读"这个 ground truth，不在这里定分母。**

**NVLink 画绝对带宽，不画利用率%**：本仓库**不给 NVLink 的峰值分母定数**，所以 `tool/metrics.py`
的面板是 `nvlink bw (GB/s bidir)`、`parse` 速览也只报 GB/s。换算只有单位换算这一步：
- 方向拆分 `nvlink_tx_bytes`/`nvlink_rx_bytes`(1011/1012)：**bytes/s** ÷ 1e9 = GB/s。
- 总量 `nvlink_bandwidth_total`(449)：**MiB/s**（dmon 表头印 `MB/`）→ **× 1048576 ÷ 1e9 = GB/s**，
  **它本身就是双向合计**。⚠️ 按表头当 MB/s 除 1000 会低 **4.63%**，标定见 §5.2.1。
- ⚠️ 单向负载下"双向合计"天然只有一个方向在跑（NVLink 全双工）—— 比饱和度要按方向看 tx/rx，
  别拿双向合计的数去和单向的数比。

**NVSwitch（本机 NVLink-switched 节点，4 颗物理 NVSwitch + Fabric Manager active；`dcgmi discovery -l` 报"12"是逻辑口径）**，每 GPU 18 条 NVLink 全接到 NVSwitch。路由：
- **测"我的程序用了多少 NVLink 带宽"→ 仍看 GPU 侧 `nvlink_tx/rx_bytes`(1011/1012)+`449`**（该 GPU 到 switch 的收发流量，应用视角，主用路径不变）。
- ⚠️ **DCGM 的 NVSwitch 侧吞吐字段(780/781/861/862/843/897)在本机不可用**（负载下全 0、连 NVLS/multimem 都不点亮，issue #236），别拿来当带宽 trace。
- ✅ **要 fabric/per-switch-port 视角的"过交换机流量"→ 用第二部分的 `tool/nvswitch_traffic`（绕过 DCGM 直连 NSCQ）**；为什么不可用/数据源/口径/采样地板见**第二部分 §8–§10**。
- ⚠️ `dcgmi discovery -l` 尾部 `Cannot get devices/ConnectX list from remote node` 是多节点/IB(ConnectX) 探测报错，单节点**无害忽略**。

### 5.3 PCIe 带宽

- **用 `pcie_tx_bytes`(1009) / `pcie_rx_bytes`(1010)**（profiling，需 SYS_ADMIN）。方向：**device 发=tx、
  device 收=rx**（H2D 点亮 rx、D2H 点亮 tx）。值是 **bytes/秒**速率，无需差分（§2）；换算 `GB/s = 值/1e9`。
- ⚠️ **`pcie_tx/rx_throughput`(200/201) 已 Deprecated**，本机 15s 全程 N/A，**弃用**（虽免权限也没数据）。
- ⚠️ **idle 也非 0——dcgmi/驱动自身轮询就走 PCIe**：本机纯 idle（无 workload）实测 `PCITX/PCIRX`
  仍有 **~2–4 MB/s 底噪**（`experiments/dcgmi_pure/idle_pcie_check_*.txt`），来自 host↔GPU 的管理/
  遥测轮询流量本身要过 PCIe。→ 判"程序有没有在过 PCIe 搬数据"要**看是否远高于这个 idle 底噪**，
  几 MB/s 级别的 PCIe 值**不代表** workload 在传（真传输是几十 GB/s 量级，见 §5.6 引用的实验）。
- PCIe 与 NVLink 的取舍坑：卡间 `cudaMemcpyPeer` 不启 P2P 会**偷偷走 host 中转(PCIe)** 而非 NVLink——
  分析多卡通信路径时务必用 NVLink metric 核对（详见 `experiments/cumemcpy/README.md`）。

### 5.4 SM 与计算引擎

- `sm_active`(1002) = 至少有 1 个 warp 驻留的 SM 周期占比（"SM 有没有在忙"）。**"用了多少个 SM"
  最接近它**。与 nvidia-smi 的 "GPU-Util" **不是**一回事（后者是 NVML gpu util，见下）。
- `sm_occupancy`(1003) = 驻留 warp 数 / 硬件上限（"忙到多满 / 占用深度"）。
- ⚠️ **别用 `sm_active` 判断"GPU 是否在搬数据"**：跨边界拷贝（H2D/D2H/卡间 D2D）走 Copy Engine/DMA，
  `sm_active=0` 却在满速搬数据（详见 §5.6）。
- `tensor_active`(1004) 及 `*_hmma/imma/dfma`(1013-1015)；`fp16/fp32/fp64_active`(1006-1008)、
  `integer_active`(1016)：各计算 / 数值管线活跃占比。
- `gr_engine_active`(1001) = 图形/计算引擎活跃占比；NVML `nvmlDeviceGetUtilizationRates().gpu`
  = nvidia-smi 的 "GPU-Util"。
- **"GPU 片上网络（NoC/crossbar）utilization"**：NVML/DCGM **没有**直接字段。最接近的旁路信号是
  `dram_active`（到 HBM 的接口）、L2/显存相关计数（部分需 CUPTI）→ 本阶段覆盖不了，标记为后续（§6）。

### 5.5 显存容量占用

- `fb_used`(252) / `fb_total`(250) / `fb_free`(251)（MB）= **占了多少显存空间，不是带宽**。
  用途：`validate` 判断 job 是否真的加载 / 跑起来（配合 `sm_active`/`dram_active` 剔除空跑）。

### 5.6 引擎 vs 接口：接口 metric 与执行引擎无关（copy engine 不漏）

**背景**：kernel 内对 global memory 的搬运由 **SM** 驱动，`cudaMemcpy` 类 API 由 **Copy Engine/DMA**
驱动——那接口 metric 会不会漏掉 copy-engine 的流量？

**结论（本机 cudaMemcpy 三类实验实测）**：
- **接口 metric（`dram_active`/`pcie_bytes`/`nvlink_bytes`/449）按"接口"计量**，不管 SM 还是
  Copy Engine 驱动都照收——跨边界拷贝 `sm_active=0`（走 DMA），但流量**完整**落在 PCIe/NVLink 字节上，
  命中 HBM 的读写也落在 `dram_active`。**copy-engine 流量不漏。**
- 唯一对跨边界 copy-engine "瞎"的是 `sm_active`（那时=0）→ **别用它判"是否在搬数据"**（§5.4）。
- 反直觉点：**单卡内 D2D 反而是 SM(kernel) 搬**（`sm_active≈1`），只有跨边界（H2D/D2H/卡间）才用 Copy Engine。

→ 采集"第一遍" `dram_active + nvlink_tx/rx_bytes(+449) + pcie_bytes` 覆盖全部接口带宽，
对 Copy Engine 与 SM-kernel 驱动都不漏。三类拷贝的完整**点亮矩阵与实测数值**见
`experiments/cumemcpy/README.md`（本文档只留上面这条通用结论）。

---

## 6. 本阶段边界：NVML/DCGM 覆盖不了的（→ Phase 2）

若将来需要 **kernel 级 / 微秒级 / 精确 GB/s / 片上网络利用率** 的数据，NVML/DCGM 不够（DCGM 与 NVML GPM
是同一条采集路、同卡共用同一个 ~100ms 地板，绕过 DCGM 直调 NVML 也没用，见 §0.4/§3.1），应转向：
- **Nsight Systems**（时间线 trace）、**Nsight Compute**（单 kernel 深度 profiling）、
- **CUPTI**（编程式采集 SASS/counter，可做 per-kernel 带宽）。

这些是"侵入式/独占式" profiling，与本工具的"旁路定时采样"是不同层次，且与 DCGM profiling 冲突
（§3.2），故列为后续阶段。
> 注意：**"过 NVSwitch 交换机的 NVLink 流量"不属于这块 Phase 2 缺口** —— 它是**驱动侧**能拿到的（走 NSCQ，
> 非侵入式），只是 DCGM 没接好；已由**第二部分**的自研工具解决。这里说的"片上网络"指 GPU 内部 NoC，另一回事。

---

## 7. 本机字段可用性清单 + ground-truth 索引（避坑，选字段前必查）

> 本机 = `zkrh-58`, 8×H100, DCGM 4.2.3, driver 590.48.01。
> 依据 = 本机字段/拓扑 dump（`docs/host_probe/`）+ 各 `experiments/` 负载/频率/开销日志 + 官方 Deprecated 标注。
> idle 下的 warmup 与 200/201=N/A 现象见样例 `docs/host_probe/dcgmi_dmon_idle_example.txt`；负载下的字段行为见 `experiments/`。

### 7.0 ground-truth 文件索引
- **静态事实**（本机 dump，稳定）：
  - `docs/host_probe/dcgmi_fields_list.txt` —— 本机 DCGM 全部可见字段（含 1001–1075 profiling 组）。
  - `docs/host_probe/dcgmi_discovery.txt` —— 8× H100 的 GPU id / PCI / UUID（选卡对齐用 UUID）。
- **示例日志**（`dcgmi dmon` 长啥样，非静态事实）：
  - `docs/host_probe/dcgmi_dmon_idle_example.txt` —— 一份 idle 样例（`-e 1005,1002,200,201,1009,1010,1011,1012,449,252 -c 15 -d 1000`），
    展示表格格式 + warmup（头 ~2 拍 N/A）+ 200/201 全程 N/A + idle 稳态值（DRAMA/SMACT=`0.000`、pcie idle 底噪、nvlink/449=0）。
- **实验日志**（负载/频率/开销 ground-truth）：`experiments/cumemcpy/`、`experiments/dcgmi_pure/`、`experiments/dcgmi_overhead/`。

### 7.1 确认真不可用（warmup 再久也不出）
| 字段 (id) | 短名 | 现象（idle 样例实测） | 原因 | ✅ 替代字段 |
|---|---|---|---|---|
| `pcie_tx_throughput` (200) | TXTPT | **全程 N/A** | 官方标 **Deprecated** | `pcie_tx_bytes` (1009) |
| `pcie_rx_throughput` (201) | RXTPT | **全程 N/A** | 官方标 **Deprecated** | `pcie_rx_bytes` (1010) |

### 7.2 ✅ 已在 NVLink 负载下验证可用（1Hz + 10Hz 双频复核）
卡间 D2D 负载实测（`cudaMemcpyPeer`+P2P，单向 GPU1→GPU2，~399 GB/s；样例
`experiments/cumemcpy/logs/freq1/dcgmi_dmon_copy_d2d_inter.txt` 与 `logs/freq10/…`）：

| 字段 (id) | 短名 | idle | **负载下实测** | 说明 |
|---|---|---|---|---|
| `nvlink_bandwidth_total` (449) | NBWLT | 0 | **~405 GB/s** | **device 字段(非 profiling)**；聚合双向带宽，已是速率，无 tx/rx 之分。**单位 MiB/s**，`× 1048576 ÷ 1e9` 才是 GB/s（§5.2.1）|
| `nvlink_bandwidth_l0..` (440–448) | NBWLx | 0 | per-link | per-link 版，同为 device 字段、速率（单位未单独标定，大概率同 449）|
| `nvlink_tx_bytes` (1011) | NVLTX | 0 | **源卡 ~406 GB/s**（目的卡 0）| **profiling(需 SYS_ADMIN)**；tx 单向 |
| `nvlink_rx_bytes` (1012) | NVLRX | 0 | **目的卡 ~406 GB/s**（源卡 0）| profiling；rx 单向 |

自洽：源卡 NVLTX≈406、RX≈0；同卡 449(双向聚合，换算后)≈405 ≈ NVLTX，**吻合 0.05%**。
⚠️ 若照 dmon 表头把 449 当 MB/s 除 1000，会读到 ~388 而以为它"比 profiling 低 5%"——那是单位，不是精度（§5.2.1）。
> ⚠️ 教训：3s idle 测试曾把 449/440 误判"不可用"。**device 计数器 idle=0 是正常的，必须负载实测才能定性**（已修正）。

### 7.3 能采到、但易被误用
- **`mem_copy_utilization`(204) / nvidia-smi "Memory-Util"**：混合所有流量的粗粒度 HBM 总带宽利用率、
  无接口拆分，本工具**不采**（见 §5.1）。测 HBM 带宽用 `dram_active`(1005)。
- **`pcie_*_bytes` idle 非 0**：约 ~2–4 MB/s 是 dcgmi/驱动轮询自身的 PCIe 底噪，不是 workload（见 §5.3）。
- **NVSwitch 侧吞吐字段（780/781/861/862/843/897）**：本机**采得到但不可信为带宽**——负载下全 0 或不随流量变
  （连 NVLS/multimem 都不点亮），别拿来当"过交换机的带宽 trace"。须按 `nvswitch` 实体采（wildcard 不支持，
  须显式 `-i nvswitch:0,...,11`），用 GPU 的 `-i 0,1` 采不到。完整依据见**第二部分 §8** 与 `experiments/nvswitch/`。

### 7.4 本机 idle 实测要点
- profiling 字段 idle 返回合法 `0`（非权限错误）→ **本宿主机 profiling 可采，无需额外 sudo**（§4）。
- `nvlink_bandwidth_total`(449)/`_l0`(440) idle 全程返回**合法 `0`（非 N/A）**——idle 使然，负载下才现真值（§7.2）。
- **idle 底噪只有 PCIe 一个（全字段实测）**：空卡上只有 `pcie_tx/rx_bytes`(1009/1010) 有 ~2MB/样本 ≈ **20MB/s @10Hz**
  底噪（host↔GPU 遥测/管理轮询走 PCIe，§5.3/§7.3）；**HBM(`dram_active`)、NVLink(`nvlink_tx/rx_bytes`+449)、
  SM/计算(`sm_active`/`sm_occupancy`/`gr_engine_active`/`tensor_active`)、`mem_copy_util`(204)、`fb_used`(252)
  idle 全 0/NA，无底噪**。→ 采集器**只需给 PCIe 扣底噪/设阈值**，其余接口 idle 即 0、任何非 0 即真活动。
  依据：GPU7+GPU3 双卡全字段扫描 `experiments/dcgmi_pure/idle_allmetrics_100ms_gpu{7,3}.txt`
  （判定脚本 `analysis/analyze_idle_noise_allmetrics.py` → `idle_noise_allmetrics_summary.csv`）。
  NVLink 特别确认：本机 NVSwitch + Fabric Manager active，但 FM 只维持链路状态（控制面），**不在 `nvlink_*_bytes` 留数据字节**。

> 维护约定：**换机器/驱动/DCGM 版本发现新的采不到/不可信字段，追加到本表并注明机器与版本**；
> 对"idle=0"类字段，**尽量补一次负载实测**再定性，别只凭 idle 判死。

---

# 第二部分 · nvswitch_traffic —— NSCQ 直读 NVSwitch 流量（补 DCGM/NVML 的空缺）

> DCGM/NVML 在本机拿不到"**过 NVSwitch 交换机的 NVLink 流量**"（见第一部分 §5.2 与
> [issue #236](https://github.com/NVIDIA/DCGM/issues/236)）。自研工具 `tool/nvswitch_traffic/` 绕过 DCGM、
> **直连 `libnvidia-nscq` 读每端口累计计数器再差分成速率**，填补这块空缺。
> 本章是**参考/依据**；构建与参数细节见 `tool/nvswitch_traffic/README.md`，负载实测见 `experiments/nvswitch/`。

## 8. 为什么需要它：DCGM/NVML 的 NVSwitch 空缺
三层数据源都试过（源码 + 本机负载实测，见 `experiments/nvswitch/`）：
- **GPM/NVML 没有 NVSwitch**：GPM 的 `nvmlGpmSampleGet` 只吃 GPU device 句柄、只测 GPU 内部单元；
  经典 NVML 只有拓扑计数（`NVML_FI_DEV_NVSWITCH_CONNECTED_LINK_COUNT` 147），**没有交换机吞吐 API**。
- **DCGM 有 NVSwitch 字段但在本机是坏的**：`nvswitch_(link_)throughput_tx/rx`(780/781/861/862)、
  `nvlink_*_bandwidth_total`(843/897) 在 24TB GPU↔GPU 负载下**全 0 或不随流量**（连 NCCL NVLS/multimem 走交换机
  计算引擎的流量都不点亮）。已知 NVIDIA 问题 <https://github.com/NVIDIA/DCGM/issues/236>。根因：本机 DCGM 走 **NVSDM**
  后端（IB 式静态 port 计数器 `NVSDM_PORT_TELEM_CTR_EXT_XMIT/RCV_DATA`），不覆盖 intra-node 数据面。
- **但底层 NSCQ 有一条 DCGM 没接的路**：`libnvidia-nscq` 的 `/{nvswitch}/nvlink/{port}/throughput_counters`
  路径直读能**精确拿到每端口流量**（本机实测：idle=0；P2P 8TB→测得 7.4TB；NVLS 15.5TB；点亮端口数=参与卡×18）。
  → 这就是本工具做的事。**免 sudo**（FM 在跑也能并发只读）。

## 9. 数据源与口径
- **源** = NSCQ per-port `throughput_counters`（`struct nscq_link_throughput_t{ uint64_t rx, tx; }`，**单位 Mibits，累计计数器**）。
- **实体** = per (物理 switch, port)：本机 **4 颗 NVSwitch × 64 端口 = 256**（switch 按 UUID 排序编号 `0..3`）。
- **速率靠差分**：`GB/s(十进制) = ΔMibits · 2²⁰ / 8 / 1e9 / Δt`。（对照第一部分：DCGM GPU 侧 `nvlink_*_bytes`
  已是 bytes/s、无需差分——两条路口径不同，别混。）
- **视角 = fabric/端口**：一条跨卡流量在源侧 switch 计 rx、目的侧计 tx → **整机总带宽 ≈ 2× 单向应用带宽**。
  量"我的程序用了多少 NVLink"仍首选 GPU 侧 `nvlink_tx/rx_bytes`(§5.2)；**看交换机端口热点/每 switch 分布**才用这个。

## 10. 采样地板 ~100ms（机制与第一部分的 GPM 不同）
- **计数器本身细粒度**（50ms 间隔下每拍增量平滑一致、无 0-洞）；真正的地板来自**读延迟**——`nscq_session_path_observe`
  读全部 256 端口约需 **~100ms**（`-d 1` 空跑两拍间隔 min/median=100ms 实测）。
- 工具用**后台 reader 线程**背靠背读（~100ms/次 = 一次采集）、主线程按 `-d` 取最新快照差分打印，**读写解耦** →
  `-d 100` 负载下实测稳定 ~100ms；`-d<100` 不会更快，`-d≥100` 按 `-d` 出拍、dt 按真实墙钟算、速率正确。
- 对照第一部分 §3.1：**GPM 的地板是"计数器刷新率"，这条是"读延迟"**，结论都 ~100ms、机制不同。无官方文档规定 NSCQ 刷新率
  （未公开的 stable API），以上为本机实测。

## 11. 坑：掉卡 / down-link 会污染端口计数器 → 间歇打印巨大假尖峰（实测）
**现象**：当某 GPU 掉卡（`nvidia-smi` 少一张）、其一条 NVLink down 后，`nvswitch_traffic` 会**隔几拍在该 down-link
所在的 (switch, port) 打印一个极大值**（本机实测 ~4.2e11 GB/s 量级），其余拍为 0。这是**掉卡产生的假值，不是真实流量**，
也不是工具正常逻辑的 bug。

**机理**（本机实测证据链）：
- down-link 的端口**仍被 NSCQ 枚举**（记录数 256 不变 → 工具"端口数变化才重建"的守卫不触发）。
- 对这条口，`nscq_session_path_observe` **返回 `rc=NSCQ_RC_SUCCESS`（rc=0），但 `throughput_counters` 无效**：
  其绝对值在"正常低值 ↔ 一个**固定垃圾常量**"之间来回跳（本机测得该常量 `1643747071612666` = `0x0005d6fab04dd6fa`，
  且 **rx 与 tx 完全相等**——真实流量几乎不可能逐比特相等，可作脏数据信号）。
- 工具按 `delta = cur − prev` 差分：`cur=垃圾, prev=正常` 那拍 → delta ≈ 1.6e15 Mibits → 巨大尖峰；下一拍
  `cur=正常, prev=垃圾`（cur<prev）→ `delta()` 的"计数器倒退归 0"守卫生效 → 打印 0。**一冒一停**即"隔一段时间冒一次"。

**当前工具无防护**：`tput_cb` 信任 `rc=SUCCESS`、不对单次读到的累计计数器做量级 sanity check；`delta()` 只挡"倒退"、
挡不住"暴涨"。识别信号：单口速率**大到不可能**（比同一负载下 GPU 侧 NVLink 计数器高一个量级以上）、且 **rx==tx**。

**复现 / 定位**（用已发布工具，raw 模式看每端口 Δ）：
```bash
tool/nvswitch_traffic/nvswitch_traffic -l port -i <sw> -r -e total -d 500 -c 12   # 异常端口那列会周期性出现固定大 Δ
```
本机观测实例（2026-07-23，GPU-7 掉卡）：假尖峰落在 **switch 2 / port 58**（该 down-link 对应端口），
Δtotal 恒为 `3287494143225332` Mibits（= 2×上述常量）。rc=SUCCESS + 绝对计数器的确认，用一段只读 NSCQ 探针
（打印 per-port `rc` 与绝对 `rx/tx`）验证；工具原码未改。
