# experiments/nvswitch —— 背景与交接（scoping / 数据源结论 / 原始 prompt）

> 这个文件保存**跑实验之前**的调查背景：三层数据源结论、字段清单、DCGM 源码依据、
> 以及上个 session 的起手 prompt。**实验介绍 / log 索引 / 结果总结在 `README.md`。**
> 分工：本文件 = "为什么这么设计实验 + 上游事实"；README = "怎么测、测了什么、结论"。

本机：`zkrh-58`，8× H100 80GB HBM3，**NVLink-switched：12× NVSwitch + `nvidia-fabricmanager` active**（单节点），
DCGM 4.2.3，driver 590.48.01。

---

## 0. 这个实验要回答什么
DCGM 的 **NVSwitch 侧**字段（每链路/聚合吞吐、延迟直方图）在本机能不能当"**应用打过交换机的 NVLink 带宽**"
的时间序列 trace 用？具体：
- `780/781`（per-link tx/rx）、`861/862`（switch 聚合 tx/rx）——**哪些字段跟随真实流量抬升**、
- 单位是 **bytes/s（速率，直读）** 还是 **累计计数器（需差分）**、
- idle 基线是多少、warmup 行为如何。

→ **结论见 README §「结果」**（一句话：本机这些 switch 侧字段都不跟随流量，不可用）。

---

## 1. 已核实的背景结论（DCGM 源码 + 本机实测 + 官方；勿再推翻）

### 1.1 三层数据源：只有 DCGM 有 NVSwitch，GPM/NVML 都没有
- **GPM 没有 NVSwitch。** GPM 的 `nvmlGpmSampleGet(nvmlDevice_t device, ...)` 只吃 **GPU device 句柄**、
  只测 GPU 内部功能单元（SM/DRAM/PCIe/**GPU 侧** NVLink）。DCGM 源码里 `DcgmFieldIdToNvmlGpmMetricId()`
  **没有任何 nvswitch 字段映射到 GPM metric**。→ 主线文档里那套"profiling=NVML GPM、~100ms 地板"完全**不延伸**到交换机。
- **经典 NVML 基本没有。** NVML 里 NVSwitch 只有 `NVML_FI_DEV_NVSWITCH_CONNECTED_LINK_COUNT`(147) 这种拓扑计数；
  带宽类的 `nvmlDeviceGetNvLinkUtilizationCounter` 是 **GPU 侧**、且已废弃。**没有交换机吞吐 API。**
- **DCGM 有，而且很全。** DCGM 有专门的 NVSwitch 模块，数据源 = **NSCQ（`libnvidia-nscq`）/ NVSDM**——
  **Fabric Manager 那套栈，不是 NVML/GPM**。源码：`modules/nvswitch/DcgmNscqManager.cpp`、
  `modules/nvswitch/DcgmNvsdmManager.cpp`（NVSDM 注释："NVSWITCH throughput is obtained by summation of its links throughput"）。
  - **本机实测（README §2）确认 DCGM 走的是 NVSDM 后端**：780/781/861/862 都映射到 NVSDM 的 IB 式 port 遥测
    计数器 `NVSDM_PORT_TELEM_CTR_EXT_XMIT/RCV_DATA`（`DcgmNvsdmManager.cpp:130-137`）；DCGM 的 NSCQ manager 无 throughput 代码
    → 这些字段不跟随本机流量（已知 NVIDIA 问题 DCGM issue #236）。
  - **⚠️ 但 NSCQ 库本身有 DCGM 没接的 throughput 路径**：`libnvidia-nscq` 的 `/{nvswitch}/nvlink/{port}/throughput_counters`
    （`nscq_link_throughput_t{rx,tx}` Mibits 累计），**本机直连读能精确拿到过交换机流量**（README §3）。即"只有 DCGM 有
    NVSwitch"要修正为"**DCGM 的字段在本机不灵，但底层 NSCQ 有一条能用的路**"。

### 1.2 采集方式
- 用 **`-i nvswitch:N` 实体**采（不是 GPU 的 `-i 0,1`）。
- ⚠️ **wildcard `nvswitch:*` 在 `dcgmi dmon` 不支持**（报 `Entity type: [3] does not support wildcard`）——
  必须显式列：`-i nvswitch:0,nvswitch:1,...,nvswitch:11`。
- 本机 `dcgmi discovery -l` → **12 个 NvSwitch（ID 0–11）**。
  （`discovery -l` 尾部 `Cannot get devices/ConnectX list from remote node` 是多节点/IB(ConnectX) 探测报错，
  单节点无害忽略；也侧面印证 NVSDM 在找多节点/IB 侧端口。）

### 1.3 对主线工具的定位（别搞混）
测"**我的程序用了多少 NVLink 带宽**"，**GPU 侧 `nvlink_tx/rx_bytes`(1011/1012，走 NVML GPM) 仍是对的**
——活的、per-GPU、应用视角（主线工具已选它，见 `docs/metrics_reference.md` §5.2）。
NVSwitch 侧字段是 **fabric/端口视角**（per-switch、per-port），看交换机热点/健康有用；**单节点**下对"应用带宽"
与 GPU 侧口径重叠。本调查的目的是**搞清 switch 侧字段是否可用**，不是要替换 GPU 侧主路径。

---

## 2. NVSwitch 字段清单（本机 `docs/host_probe/dcgmi_fields_list.txt` 均存在）

| 类别 | 字段 (id, 短名) | 说明 |
|---|---|---|
| **每链路吞吐（主候选）** | `nvswitch_link_throughput_tx/rx` (780/781, SWLNKTX/SWLNKRX) | 交换机端口的 tx/rx；实体其实是 FE_LINK。→ README §结果：负载下恒 0 |
| **交换机聚合吞吐（主候选）** | `nvswitch_throughput_tx/rx` (861/862, SWTX/SWRX) | 整台 switch tx/rx；idle=大常量。→ README §结果：NVSDM 累计计数器但不跟随流量 |
| 每链路带宽（另一组） | `nvlink_tx_bandwidth_link0..17`+total (825–843, NVLTXB*/NTXBWLT)、`nvlink_rx_bandwidth_link0..17`+total (879–897, NVLRXB*/NRXBWLT) | MB/s；switch 和 GPU 实体都能查。→ README §结果：H100 上恒 0（死字段） |
| 每链路延迟直方图 | 789–808（low/medium/high/panic × VC0–3, SWVCL*） | 按 VC 分桶的延迟分布；本次未逐一验证（switch 吞吐字段已全灭，延迟同后端） |
| 错误计数 | 782–788、809–824（fatal/nonfatal/replay/recovery/crc/ecc） | 健康监控用，非本次重点 |
| 遥测 | 701–707、858–862（电压/电流/功耗/温度） | 非本次重点 |

---

## 3. scoping 阶段跑过的命令（复现用）
```bash
# 交换机拓扑（12 台，ID 0-11）
dcgmi discovery -l
# idle 探测（无 workload）——观察到 780/781=0、861/862=大常量、tx≈rx 几拍不变
dcgmi dmon -e 861,862,780,781 -i nvswitch:0,nvswitch:1,nvswitch:2 -c 4 -d 1000
```
（idle 观测当时的疑点"780/781=0 像活的、861/862=大常量像累计或静态"，已被 README §结果的负载实测定性。）

---

## 4. 源码 / 文档参考
- DCGM 源码（本机 clone `~/Repos/DCGM`）：
  - `modules/nvswitch/DcgmNvsdmManager.cpp`（NVSwitch 吞吐的**实际后端**；`getNvsdmPortFieldId()` @130-137
    把 780/781/861/862 映射到 `NVSDM_PORT_TELEM_CTR_EXT_XMIT/RCV_DATA`；`isCompositeFieldId()` @217 注释
    "throughput = summation of its links throughput"）、`modules/nvswitch/DcgmNscqManager.cpp`（无 throughput 代码）。
  - `dcgmlib/src/dcgm_fields.cpp:638/3617`：780/781/861/862 的 field 定义（**unit 串为空 = raw INT64 累计计数器**）。
  - `dcgmlib/dcgm_fields.h`：`DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX/RX`(780/781)、
    `DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX/RX`(861/862)、延迟直方图(789-808)。
  - `dcgmlib/src/dcgm_fields.cpp:6524` `DcgmFieldIdToNvmlGpmMetricId()`（确认**无** nvswitch→GPM 映射）。
- 本仓库：`docs/metrics_reference.md` §0（NVML/DCGM/GPM 关系）、§5.2（NVLink/NVSwitch 段）；
  `docs/host_probe/dcgmi_fields_list.txt`（本机全字段）；`experiments/cumemcpy/`（cross-card NVLink workload 可复用）。
- 官方：DCGM Feature Overview（NVSwitch 监控）
  <https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html>。

---

## 5. 上个 session 的起手 prompt（历史留存）
```
读 experiments/nvswitch/README.md —— 那是上个 session 对 NVSwitch metric 的调查交接文档，
含已核实结论（GPM/NVML 都没有 NVSwitch，只有 DCGM 走 NSCQ/NVSDM）+ 本机 idle 探测结果 + 待跑的负载实测设计。

任务：按该 README 的 §4「待跑：负载实测设计」执行，验证 DCGM 的 NVSwitch 侧字段
（780/781 每链路吞吐、861/862 聚合吞吐；可选 789-808 延迟、825-843/879-897 per-link 带宽）
在本机（8×H100 + 12×NVSwitch、Fabric Manager active）能不能当"应用打过交换机的 NVLink 带宽"时间序列 trace 用。
重点搞清：哪些字段随真实流量抬升、单位是 bytes/s（速率直读）还是累计计数器（需差分）、idle 基线与采样地板。

做法：复用 experiments/cumemcpy 的 cross-card cudaMemcpyPeer+P2P workload 打满 NVLink（务必确认 P2P 真启用、
流量真走 NVLink 而非 PCIe），宿主机同时用 dcgmi dmon 抓 nvswitch 实体（-i nvswitch:0,...,11，注意 wildcard 不支持）
的这些字段，并同时采 GPU 侧 nvlink_tx/rx_bytes(1011/1012) 做口径基准。

遵守 CLAUDE.md 的实验约定与工作原则。跑完把结果矩阵写进 README，并把定论写回 docs/metrics_reference.md §5.2。
```
（本 session 已完成该任务，并额外补了 NCCL/NVLS(multimem) 集合通信对照——见 README §结果。）
