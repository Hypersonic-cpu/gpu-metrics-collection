# experiments/nvswitch —— 能否拿到"过交换机的 NVLink 流量" trace（DCGM ❌ / NSCQ 直读 ✅）

> **状态：✅ 已完成。** 两句话结论：
> 1. **DCGM 的 NVSwitch 侧吞吐字段（780/781/861/862/843/897）在本机不可用**——真实 NVLink 负载下（P2P 单播 /
>    NCCL Ring / NCCL NVLS-multimem 都试过）全部读 0 或不随流量变（已知 NVIDIA issue，见 §4）。
> 2. **但 NVSwitch 芯片的流量是拿得到的——绕过 DCGM，直连 `libnvidia-nscq` 读 per-port `throughput_counters`**：
>    idle=0、负载下逐端口精确抬升（P2P 8TB→测得 7.4TB、NVLS 15.5TB），**本机免 sudo**。DCGM 只是没接这条路。
>
> 量"我的程序用了多少 NVLink 带宽"仍首选 GPU 侧 `nvlink_tx/rx_bytes`(1011/1012)+`449`（最省事）；
> **要 fabric/per-switch-port 视角的过交换机流量 → 用 §5 的 NSCQ 直读**。
> 背景/数据源/原始 prompt 见 **[`BACKGROUND.md`](BACKGROUND.md)**；定论已写回 `docs/metrics_reference.md §5.2`。

本机：`zkrh-58`，8× H100 80GB HBM3，**12× NVSwitch + Fabric Manager active**（单节点），DCGM 4.2.3，driver 590.48.01。
所有 GPU 两两 = **NV18**（18 条 NVLink 全接 NVSwitch，无 GPU 直连；`nvidia-smi topo -m` 证）→ 任何跨卡 NVLink 流量**必过交换机**。

---

## 1. 测试介绍

### 1.1 两类 workload（都在容器 `nvcr.io/nvidia/pytorch:25.12-py3` 里跑，用空卡）
| workload | 通信方式 | 走交换机的什么 | 打出的带宽 |
|---|---|---|---|
| `workloads/nvlink_p2p_saturate.py` | 单卡→单卡 `cudaMemcpyPeerAsync`（**P2P**）| 交换机 **crossbar 单播 DMA** | ~398 GB/s（单向，GPU1→GPU2）|
| `workloads/nccl_allreduce.py` | 多卡 NCCL `all_reduce` | Ring=单播环；**NVLS=multimem/SHARP，在交换机里做 reduce** | busbw ~350–364 GB/s（4 卡）|

- **P2P 真启用已核对**：`cudaDeviceCanAccessPeer=1` **且**实测 ~398 GB/s（若退化到 PCIe host 中转只 ~37 GB/s、nvlink=0，见 cumemcpy README 的 P2P 坑）。
- **NVLS/multimem 真使用已核对**（双证据，见 §2.3）：运行时 NCCL_DEBUG 打 `Algo NVLS proto SIMPLE`；
  静态 cuobjdump 见 libnccl sm_90 有 7687 条 `LDGMC` multimem 指令。

### 1.2 采集（宿主机，dcgmi 是 host 二进制）
两条 `dcgmi dmon` 并采、夹住同一时间窗（idle 头 → workload → idle 尾）：
- **switch 流**：`-e 861,862,780,781[,843,897] -i nvswitch:0..11`（全 12 台，**wildcard 不支持须显式列**）。
- **GPU 基准流**：`-e 1011,1012,449 -i <占用卡物理 id>`（已知可信，做口径基准）。
- 1Hz smoke + 10Hz(`DMON_MS=100`) 正式双频复核。
- **861/862 专测**（累计计数器判定）：单次快照法 `dcgmi dmon -c 1`，比 60s idle 漂移 vs 60s 负载（~398GB/s×60s≈**24TB**）的 Δ
  （因 861/862 在单次 dmon invocation 内被冻结，只能跨 invocation 比较，见 §2.2）。

### 1.3 复现
```bash
# 单卡 P2P（1Hz smoke / 10Hz 正式）
experiments/nvswitch/run_probe.sh smoke 1 2 2048 10
DMON_MS=100 experiments/nvswitch/run_probe.sh p2p_10hz 1 2 2048 30
# NCCL 集合通信（auto 会选 Ring；强制 NVLS 走 multimem）
experiments/nvswitch/run_nccl_probe.sh nccl_auto_4gpu 1,2,3,4 512 20
NCCL_ALGO=NVLS experiments/nvswitch/run_nccl_probe.sh nccl_nvls_4gpu 1,2,3,4 512 20
```

---

## 2. 结果

### 2.1 结果矩阵（字段 × idle × 各类负载 × 判定）
"负载下"= 上述任一 workload 打满时，全 12 台 switch 同采所见：

| 字段 (id, 短名) | 实体 | 单位 | idle | P2P 单播 | NCCL Ring | **NCCL NVLS/multimem** | 判定 |
|---|---|---|---|---|---|---|---|
| `nvswitch_link_throughput_tx/rx` (780/781, SWLNKTX/RX) | FE_LINK | 无(raw) | `0` | `0` | `0` | **`0`** | ❌ 死字段，全程 0 |
| `nvswitch_throughput_tx/rx` (861/862, SWTX/SWRX) | switch | 无(raw) | 大常量、tx≈rx | Δ=0 | Δ=0 | **Δ=0** | ❌ 累计计数器**但不跟随** GPU↔GPU 流量 |
| `nvlink_tx/rx_bandwidth_total` (843/897, N*BWLT) | switch & GPU | MB/s | `0` | `0` | `0` | **`0`** | ❌ H100 上死字段（旧 NVML per-link bw，同 200/201 已废） |
| `nvlink_tx/rx_bytes` (1011/1012)〔GPU 基准〕 | GPU | bytes/s | 0(头~2拍N/A) | **~400 GB/s，方向分明** | ~340×2 GB/s | ~330 GB/s | ✅ 可用（主线已选）|
| `nvlink_bandwidth_total` (449)〔GPU 基准〕 | GPU | MB/s | 0 | ~388 GB/s | ~682 GB/s(双向) | ~592 GB/s | ✅ 可用（MB/s 直读免差分）|

> **DCGM 的 switch 侧 5 个候选字段没有一个随流量抬升**——连"真的在交换机里做 reduce 的" NVLS/multimem 流量都不点亮。
> 这是**已知 NVIDIA 问题**（[DCGM issue #236](https://github.com/NVIDIA/DCGM/issues/236)：H100 HGX 上
> `DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_{TX,RX}` 恒 0，官方未修）。**→ DCGM 这条路在本机拿不到过交换机流量；
> 但换 NSCQ 直读能拿到，见 §3。**

### 2.2 为什么不可用 / 为什么 idle 下"每台 switch 都有个差不多的大数"（源码 + 实测）
1. **那个"大常量"不是你的流量，是 NVSDM 的 IB 式累计 port 计数器。** 780/781/861/862 在本机走 **NVSDM 后端**
   （不是 NSCQ——NSCQ manager 没有 throughput 代码），映射到 `NVSDM_PORT_TELEM_CTR_EXT_XMIT/RCV_DATA`
   （`DcgmNvsdmManager.cpp:130-137`）——就是 InfiniBand 的 `PortXmitData/PortRcvData` 那种**累计计数器**。
   字段定义 unit 串为空（`dcgm_fields.cpp:638/3617`）= raw INT64 计数器，本就不是速率。各台 switch 的数值
   （47G/51G/42G…）是**开机至今的静态累计基线**，彼此接近只是因为都是同型号 switch 的相近常量，**跟你有没有跑程序无关**。
2. **它不计量本机 intra-node GPU↔GPU 数据面流量（实测决定性）。** 24TB 的 GPU1↔GPU2 负载前后单次快照对比：
   **Sw0–10 计数器 Δ=0（一位没动）**，只有 Sw11 有 ~10⁵ 量级微漂——且这漂移量**与 60s idle 的漂移量一样大**
   （疑似 GPU0 上别的租户 / FM 心跳），**不是**我们的 24TB。若它真按字节累计，24TB 应让某些 switch Δ≈10¹²~10¹³。
   （推测：NVSDM 面向 NVLink-network/多节点 fabric 端口，单节点 HGX 的 GPU-facing 数据口没接到这些计数器；
   `discovery -l` 那两条 ConnectX/remote-node 报错也印证 NVSDM 在找多节点/IB 侧。）
3. **即便它跟随，也给不出"运行中"的时间序列。** 10Hz 跑 30s（每台 550 拍）里 861 `min==max`——
   **在单次 dmon invocation 内被冻结**，只有跨 invocation 才见漂移。无 <invocation 级分辨率。
4. **关于"P2P 是不是每台 switch 都均摊流量"**：物理上，GPU1→GPU2 的单播确实会**把 18 条 NVLink 铺开、经多台
   switch 转发**（所以理论上每台都该有份）——但上面的计数器**根本没记录**这份流量，所以你在 dmon 里看到的
   "每台都差不多"是**静态基线**而非均摊的实时流量。别把它当成"流量均匀分布"的证据。

### 2.3 multimem/NVLS 确实被使用（回应"确认有 multimem 指令"）
- **运行时**：`multicast_supported=1`；NCCL 建 NVLS multicast group；强制 `NCCL_ALGO=NVLS` 后 **9054 次 all_reduce
  全部 `Algo NVLS proto SIMPLE`**（`logs/nccl_nvls/nccl_debug_nccl_nvls_4gpu.txt`）。NVLS SIMPLE 就是用 multimem 实现的。
- **静态**：`multimem.ld_reduce/st` PTX 在 **sm_90 SASS 里是 `LDGMC/STGMC`**（LoaD/STore Global MultiCast；
  用 `workloads/multimem_opcode_probe.cu` 编译反汇编确认）。libnccl(2.28.9) sm_90 含 **7687 条 `LDGMC`** multimem 指令
  （ADD/MIN/MAX × F32/F64/BF16/F16；`logs/nccl_nvls/multimem_sass_proof.txt`）。
- 即便如此，DCGM switch 侧字段在这类流量下仍全 0 → **DCGM 侧结论对 multimem 同样成立**。

---

## 3. ✅ 拿得到过交换机流量的路：绕过 DCGM，直连 NSCQ 读 per-port `throughput_counters`

DCGM 拿不到，是因为它这版把 NVSwitch 吞吐**接到了 NVSDM 后端**（IB 静态计数器）；而 **NSCQ 库（`libnvidia-nscq`，
Fabric Manager 那套）本身有一条 DCGM 没用的路径** `/{nvswitch}/nvlink/{port}/throughput_counters`
（返回 `struct nscq_link_throughput_t{ uint64_t rx, tx; }`，**单位 Mibits，累计计数器**）。**直连读它就能拿到真流量。**

### 3.1 实测：NSCQ per-port 计数器精确跟随过交换机的流量
`workloads/nscq_throughput_probe.cpp`（host 侧链接 `-lnvidia-nscq`，read#1 → workload → read#2 差分）：

| workload | 打了多少（GPU 侧口径）| NSCQ 测得 ΣΔrx | NSCQ 测得 ΣΔtx | 点亮端口数 |
|---|---|---|---|---|
| **idle** | 0 | **0 GB** | **0 GB** | 0 |
| **P2P GPU1→GPU2**（单向 ~398GB/s×20s≈8TB）| ~8 TB 单向 | **7440 GB** | **7440 GB** | 18 tx + 18 rx（=GPU1 的 18 条 NVLink）|
| **NVLS/multimem**（4 卡 all_reduce）| busbw ~363 GB/s | **15516 GB** | **15517 GB** | 72 tx + 72 rx（=4 卡 ×18）|

- **idle=0、负载下逐端口精确抬升**，量级与 GPU 侧口径吻合（8TB→7.4TB；点亮端口数 = 参与卡数×18 条 NVLink，物理自洽）。
- **单播 P2P 和 multicast NVLS/multimem 都被计到** —— 这正是 DCGM 那 5 个字段做不到的。
- **本机免 sudo** 即可读（FM 在跑也能并发读，只读 observer）。

### 3.2 用它当 trace 的要点
- **是累计计数器（Mibits），要差分**：`速率 = (读2−读1) / Δt`；`GB ≈ Mibits·2²⁰/8/1e9`。（对照 DCGM GPU 侧
  `nvlink_*_bytes` 是已经算好的 bytes/s、无需差分——两条路口径不同。）
- **实体是 per (switch, port)**：**本机 4 颗物理 NVSwitch（=4 个 PCIe 设备 05/06/07/08），每颗 64 端口 = 256 端口回调**
  （注意：`dcgmi discovery -l` 报"12 NvSwitch"是逻辑口径，NSCQ/sysfs 的物理芯片数是 **4**）。要"整卡/整机聚合"自己按端口求和。
- **采样地板 ~100ms，机制≠GPM**：计数器本身**细粒度**（50ms 间隔下每拍增量平滑一致、无 0-洞）；地板来自
  **读延迟**——`observe` 读全 256 端口约需 **~100ms**（`-d 1` 空跑两拍间隔 min/median=100ms 实测）。故 `-d<100ms` 不会更快、
  单拍速率有量化抖动（时间平均仍准）。GPM 的地板是"计数器刷新率"，这条是"读延迟"——结论相近、机制不同。无官方文档规定 NSCQ 刷新率。
- **口径提醒（单节点）**：per-port 是 **fabric/端口视角**（一条跨卡流量源侧 switch 计 rx、目的侧计 tx，故整机总带宽≈2×单向应用带宽），
  和 GPU 侧 `nvlink_tx/rx_bytes` 口径重叠；量"我的程序用了多少 NVLink"用 GPU 侧最省事，**要看交换机端口热点/每 switch 分布**才用 NSCQ 这条。
- **已产品化为工具** → `tool/nvswitch_traffic/`（`make` 即用，`dcgmi dmon` 风格 CLI，可实时打印/可脚本采 CSV）。
  本目录的 `run_nscq_probe.sh` + `workloads/nscq_throughput_probe.cpp` 是原型/ground-truth（2-读差分）；日常用 `tool/` 那个。
- 复现：`experiments/nvswitch/run_nscq_probe.sh {p2p,nvls,idle} <workload> <devices> [size] [sec]`。

---

## 4. 日志索引（ground truth）
> **原始日志已打包为 `logs.tar.gz`**（解压后即 `logs/` 树，下文 `logs/...` 路径均指压缩包内）。
- **采集器**：`run_probe.sh`（单卡 P2P + DCGM 双流）、`run_nccl_probe.sh`（NCCL + DCGM 双流）、
  **`run_nscq_probe.sh`（✅ NSCQ 直读）**。
  **workload**：`workloads/nvlink_p2p_saturate.py`、`nccl_allreduce.py`、`multimem_opcode_probe.cu`、
  **`nscq_throughput_probe.cpp`（NSCQ 客户端）**。
- **DCGM 侧（结论：不可用）**：
  - `logs/freq1|freq10/dcgmi_{switch,gpu}_*.txt` —— P2P 1Hz/10Hz，switch 全 0 / GPU 基准点亮。
  - `logs/counter_diff/{counter_before_after,deltas_summary}.txt` —— 861/862 的 24TB before/after（证不跟随）。
  - `logs/nccl_nvls/dcgmi_*_nccl_{auto,nvls}_4gpu.txt` + `nccl_debug_*`（NVLS 选择证据）+ `multimem_sass_proof.txt`
    （libnccl 含 7687 条 LDGMC multimem 指令）。
- **NSCQ 直读（结论：✅ 可用）**：`logs/nscq_direct/nscq_{idle,p2p_load_raw,nvls}.txt`
  —— idle=0 / P2P 7.4TB / NVLS 15.5TB，逐端口 rx/tx 差分。

> **dmon 日志怎么读**：`dcgmi dmon` 每个采样拍**都会把 `-i` 里列的每一个实体各打一行**（switch 流 = 每拍 12 行，
> 12 台 switch 各一行；GPU 流 = 每拍每张卡一行）。所以 switch 日志行数 = 拍数 × 12。列头是字段短名
> （SWTX/SWRX/SWLNKTX/SWLNKRX = 861/862/780/781；NVLTX/NVLRX/NBWLT = 1011/1012/449），单位标注可能被截断（`MB/`）。
