# CLAUDE.md — tool/nsys（给自己看：机制 / 口径 / 实测澄清）

**分工**：操作怎么调 → `README.md`（给用户）；某次实验的数值和解读 → 本目录 `EXP_*.md`；
**本文件 = 为什么这么设计、哪些是实测钉死的口径、哪些坑别再踩**。动手改本目录代码前先读这里。

---

## 一、采集侧机制

### 1.1 四类轨道 / scope（决定能不能 attach）★

| 轨道 | 看到什么 | nsys 开关 | scope | 要 sudo | 能对已在跑的进程事后开采？ |
|---|---|---|---|---|---|
| GPU Metrics | SM/DRAM/NVLink/PCIe 连续曲线 | `--gpu-metrics-devices` | System | ✅ 恒需 | ✅ **能** |
| CPU 采样 / 线程调度 | host 线程、调用栈、OS runtime | `-s`, `--cpuctxsw` | System | process-tree ❌<br>system-wide ✅ | ✅ 能 |
| **CUDA HW 轨** | **kernel/memcpy 甘特图** | `-t/--trace` | **Application** | ❌ | ❌ **不能** |
| CUDA API 轨 | host 侧 `cudaLaunchKernel` + 连线 | `-t/--trace` | Application | ❌ | ❌ 不能 |

**根因**：`--trace` 是 Application scope，只存在于 `nsys launch`/`profile`，`nsys start` 根本没这个开关 ——
进程起来之后再想 trace 它不可能。所以 attach + CUDA 组时 `nsys-tool` **fail fast**，
不让你采完才发现报告里没 kernel。（容器里 nsys 2026.3.1 多一个 `--cuda-trace-scope=system-wide`，
但官方原文限定 "only CUDA processes launched by the same user **after the collection start** will be traced"
—— 已在跑的仍然不行。）

两把**互相独立**的权限锁：GPU Metrics 卡在驱动 `RmProfilingAdminOnly`（本机 =1，**恒需 root**）；
CPU 采样卡在内核 `kernel.perf_event_paranoid`（本机 =2，process-tree 免 root）。CUDA trace 两把都不管。

### 1.2 窗口固定开销 = 组决定的，不是 `-t` 决定的 ★

报告时间跨度 = `-t` + 一段**固定开销**，这段由 metric 组决定。组越重开销越大，`-t` 再小也压不下去。

实测（容器内 `cuda_dram` 组 = 4 序列 / 无 osrt / 无 CPU 采样 @200 kHz）：

| 请求 `-t` | 报告跨度 | 固定开销 | 实测采样率 |
|---|---|---|---|
| 0.05 s | **1.01 s** | 0.96 s | 200.0 kHz |
| 0.3 s | 1.20 s | 0.90 s | 200.0 kHz |
| 2 s | 2.86 s | 0.86 s | **133.6 kHz（掉速）** |

对照 `cuda` 组（30 序列 + osrt + CPU 采样）跑 sglang：同样 `-t 2` 跨度 **8.84 s**，开销 6.8 s（7 倍）。

- **地板约 1 秒**，`-t` 砍到 0.05 也没用 → 想要"短窗口+高分辨率"就**换轻量组**。
- **200 kHz 掉不掉速是「组」的属性，不是通用规律**：`cuda_dram` 组 `-t 2` 掉到 133 kHz（上表），
  但 **`nvlink` 组不掉** —— 8 卡 @200 kHz、窗口 1 / 3 / 10 / 20 s 四档，实测采样率**全是 200.0 kHz、
  覆盖 100%、diagnose 全 PASS**（`experiments/nvlink_bench/logs/multimem_winprobe_8gpu_mon8_*.csv`，
  `run_nsys_multimem.sh winprobe` 出的）。**换组要重新量，别把 `cuda_dram` 那条套上去。**
- ★ **GPU Metrics 采集对被测负载的扰动 ≈ 0**（这条以前只有"看起来没影响"，现在量过了）：
  同一个稳态负载底下连开 4 个窗口、窗口之间留 12 s 空档当「没采」对照段，
  采集中 vs 没采的吞吐差 **−0.01% ~ −0.05%**（8 卡 @200 kHz，全在噪声里）。
  所以**不必为了"怕影响性能"去压频率或缩窗口**；该压的理由是采样缓冲和后处理成本，不是扰动。
- 固定开销两截（start 的 ramp-up + stop 的 flush）都在 nsys 内部；拆到不同 shell 发 start/stop 也不会变短（已验证）。
- ★ `gen` 从调用到窗口真正打开有 **5–9 秒**（容器模式还要 `docker inspect`+`docker exec` 探 GPU 映射）。
  **想卡住一个短阶段必须提前这么多发命令**——实测想采 burst+1s 只有 5 秒的 prefill 墙，
  命令按点发出去窗口 burst+10s 才开，整个墙错过。
  ⚠️ **这个 5–9 秒不是常数，它随「频率 × 卡数 × 序列数」涨**：`nvlink` 组（9 序列）
  **200 kHz × 双卡**下实测 `start` 单独就要 **24–33 秒**（`experiments/nvlink_bench/` 六个相位，
  `marks.txt` 的 `start_cmd_begin → window_start`）。所以**高频窗口必须按实测量重新估**，
  别套 5–9 秒去规划稳态时长——稳态给短了窗口会整个滑到负载结束之后。
- ★ `stop` 之后写报告要 **10–22 秒**，`gen` 阻塞到写完 → **窗口之间至少留 ~30 秒**。
  实际落点永远以 `marks.txt` 的 `window_start` 为准，别信请求的偏移量。
  ⚠️ **`stop` 的耗时随窗口长度线性涨**（`nvlink` 组 8 卡 @200 kHz 实测）：
  窗口 1 / 3 / 10 / 20 s → `stop` 21 / 33 / 55 / 82 秒（`start` 反而基本恒定，16–33 秒）。
  换算成「每秒有效数据要付多少固定开销」：1 s 窗口 13.6×、3 s 10.8×、10 s 7.0×、20 s 5.1× ——
  **窗口越长越划算**，但报告体积和 `check_report`/`post_export` 的时间也跟着线性涨
  （8 卡 × 200 kHz × 20 s ≈ 3400 万点，导出要好几分钟）。3 s 是够用的默认；要长时间序列再上 10 s。

### 1.3 `cuda_graph_trace` 决定 CUDA graph 里的 kernel 看不看得见 ★

nsys 原生默认 `graph`：整张图在甘特图上折成**一条**，图里 kernel 一个都看不到。
被测程序用 CUDA graph 时（sglang decode 100% 在图里）**必须 `node`**。
实测：3-kernel 的图 replay 3 秒，`node` 下 `cuda_gpu_kern_sum` 是三行各 12324 次；`graph` 下只有一条 graph 记录。

### 1.4 分遍采是为了什么（实测澄清）

**不是**因为"一遍采太多会把 NVLink 丢掉"—— 实测同一 p2p 负载下，`full` 组（30 序列/卡）里
NVLink 数值与专用 `nvlink` 组**逐位一致**（user 41.3% / protocol 2.49%）。真实好处只有两条：

- 只留要看的曲线（`full` 在纯拷贝负载下 34/60 条恒 0，GUI 里得翻）；
- 报告小、导出快（5 秒 1 卡：`pcie` 1.3M / `dram` 1.4M / `pass1_core` 1.5M / `full` 2.2M）。

`sets/*.config` 的 `schedulingRule: optional` 是 NVIDIA 的降级机制，实测触发条件是**硬件压根没这个计数器**：
本机 `nvidia-smi -q` 报 `GPU C2C Mode: Disabled`（Intel Xeon 主机，非 Grace-Hopper），
gh100 里的 CTC 两项永远采不到 → `sets/pcie.config` 已去掉。

### 1.5 比 dcgmi 细在哪（选 nsys 的理由）

| dcgmi 字段 | nsys 拆成 | 多出来的信息 |
|---|---|---|
| 1005 `dram_active` | `dram__{read,write}_throughput` | **读写分开** |
| 1011/1012 `nvlink_{tx,rx}_bytes` | `nvl{tx,rx}__bytes_packet_{request,response}_data_{user,protocol}` | 请求/响应 × 用户/协议 8 路 → 算得出链路有效率（实测 p2p 41.3%:2.49% ≈ **94% 有效载荷**）|
| 1009/1010 `pcie_{tx,rx}_bytes` | `PCI.TriageCompute.pcie__{read,write}_bytes` | 含协议开销 |
| — | 全部 | **10 kHz** 而非 10 Hz |

### 1.6 GPU Metrics 与 `EXCLUSIVE_PROCESS` 不冲突

采样器**不占 CUDA context**，卡设成 `nvidia-smi -c 3` 照样能采。实测：卡 3 设 Exclusive_Process
后跑单进程 CUDA 负载 + `cuda_dram` 组 → 189,338 个 DRAM 采样点 + 9,712 个 kernel、零 Error；
全程 `nvidia-smi` 在该卡上**始终只看得到被测程序一个 compute 进程**。

（`tool/sglang_bench` 的 nsys yaml 仍写 `exclusive_process: false`，理由与 nsys 无关：
连续跑多个实验时上一个容器 CUDA context 没释放干净，下一个 server `set_device` 会撞 `cudaErrorDevicesUnavailable`。）

### 1.7 溢出是**间歇性**的，不是必现 ★

按「出问题窗口数 / 总窗口数」统计（每窗口 0.3 s，`cuda_dram` 4 序列）：

| 负载 | 采 metrics 卡数 | 频率 | 卡 | 溢出窗口 |
|---|---|---|---|---|
| sglang | 2 | 200 kHz | 3,4 | **4/4** |
| sglang | 2 | 200 kHz | 1,3 | **1/4** |
| sglang | **1** | 200 kHz | 4 | **0/4** |
| sglang | **1** | 200 kHz | 1 | **0/4** |
| sglang | 2 | **100 kHz** | 3,4 | **0/4** |
| `graph_busy`（轻负载）| 2 | 200 kHz | 5,6 | 0/3 |

合计：**双卡 200 kHz 溢出 5/8；单卡 200 kHz 0/8；双卡 100 kHz 0/4**。三条结论：

**10 序列（`cuda_iface`：DRAM+NVLink+PCIe）下重测同样的 2×2**（`EXP_iface_freq_card.md`，每格 4 窗口）：

| | 200 kHz | 100 kHz |
|---|---|---|
| 采 2 张卡 | **FAIL 2 / PARTIAL 2（4/4 全溢出）** | PASS 4/4 |
| 采 1 张卡 | PASS 4/4 | PASS 4/4 |

**14 序列（`cuda_iface_full`：NVLink 补全 8 路）下再测一遍同样的 2×2，四格判定逐格一致** ——
仍然只有「200 kHz × 双卡」坏。

★ **瓶颈是采集设备数，不是序列数**：序列 **4 → 10 → 14**（+250%）没有让任何一个能过的格子翻掉
（`200k-1card` / `100k-2card` / `100k-1card` 三格在 10 和 14 序列下都是 PASS 4/4）；
而只把采集卡从 2 张减到 1 张，同频同序列数立刻从"4/4 全坏"变"4/4 全过"。
序列数只改变**坏得多稳定**（4 序列时间歇 5/8 且只砸 `[0]`；10/14 序列时 4/4 必现、能砸 `[0,1]`），
不改变**哪一格**会坏。
→ 选型顺序：**先砍卡数、再砍频率，最后才砍序列**（砍序列换来的余量最小，且砍掉的是信息）。

⚠️ **溢出后的数据会打印出像模像样的假速率**：那次 `[0]` 中间断了 1208.6 ms，
"总样本/总跨度"算出 106.4 kHz；`[1]` 停在整 199,999 样本 / 1.0000 s（采满即停，
干净时同 workload 能连续撑 0.65–2.2 s）。**只看速率数字看不出坏**，必须看断块数和 `diagnose`。

1. **不是某张卡坏**。报错永远是 `GPU Metrics [0]`（第一个采集设备），但这个 `[0]` 两次分别是
   物理卡 3 和物理卡 1 —— **跟着索引走，不跟着卡走**。
2. **不是单纯的设备数**。轻负载双卡 200 kHz 一次没溢出过；是 sglang 那种多进程 + 海量 CUDA trace
   下，profiler 要同时排空 trace 缓冲和两路 200 kHz metrics 才顶不住。
3. **溢出只砸中 `[0]` 那一路**：那卡样本被切成两块、全部落到窗口之外（覆盖率 0%），另一卡照样干净
   的一整块精确 200.0 kHz。**报告不是全废**，但双卡对齐分析没法做。

**「双卡 200 kHz」本身不是判据，带不带 CUDA trace 才是**：`nvlink` 组（9 序列、`trace=none`）
在 **200 kHz × 双卡**下跑满载 p2p 拷贝，六个窗口**全 PASS**、两路都是精确 200.0 kHz 的单块、
覆盖 100%（`experiments/nvlink_bench/`，每窗口 170 万点/卡）。对照 `cuda_iface`（10 序列
+ `trace=cuda,nvtx`）同样双卡 200 kHz **4/4 全坏**。序列数几乎一样，差别在于 profiler 要不要
同时排空 CUDA trace 缓冲 —— 和 §1.7 第 2 条（"不是单纯的设备数"）是同一个结论的正面例证。
**所以别把"双卡就得降频"写成规则**，纯 GPU Metrics 的组该拉满就拉满，判据仍然是采完 `diagnose`。

**两条可用的退路**（溢出了再选，别预先写成死规则——具体配置 case by case 定）：
少采一张卡（TP 对称时一张足够代表），或降到 100 kHz。
kernel 甘特图不受 `-i` 影响、**始终覆盖全部卡**（Application scope；实测单卡采 metrics 时两个
deviceId 各 15,453 个 kernel），所以少采一张卡不影响 kernel 时间线。

⚠️ 样本量小且是**间歇**发生 —— **跑一轮没报错 ≠ 这个配置安全**。
唯一可靠做法是**每次采完都 `nsys-tool diagnose`**，别靠"上次没事"来预判。

**200 kHz 的收益边界**：decode 窗口 34,620 个 kernel 的实测分布 —— 吃掉 **83% GPU 时间**的大 kernel
（≥30 µs）在 100 kHz 下已有 3–10 个采样点；中位 3.8 µs 的小 kernel 在 200 kHz（5 µs/点）下**同样解析不了**。
即 200 kHz **既不解锁小 kernel，对大 kernel 也只是锦上添花**。
`groups/cuda_dram.conf` 仍保留 200 kHz 作为高分辨率档（要看 kernel 级 HBM 起伏时用），
代价是溢出风险 —— 所以那一组的纪律是「采完必 diagnose」。日常用 10 kHz 的其余 8 组。

### 1.8 判定口径（`check_report.py`）

nsys 把采集期 Error **只写进报告内部**，stdout 一个字不打、`rc` 仍是 0 → 不主动查就会拿废数据分析。
"某一路可用" = 该设备**最大连续采样块**（相邻间隔 >1 ms 算断）覆盖 kernel 时间段 ≥80%；
没有 kernel 轨时改看"最大连续块占全部样本的比例"。
PASS / PARTIAL（至少一路可用）/ FAIL（没有一路可用）/ ERROR（overflow 以外的 Error）。

⚠️ **它判的是每张卡自己好不好，不判两张卡之间对不对得上**。实测双卡窗口里两张卡的 GPU Metrics
起止差 0.5–1.3 s（`EXP_iface_freq_card.md` §4：2.1 s 跨度里真正双卡同时有数据的只有 0.8–1.2 s），
四个窗口全是 PASS。**跨卡对齐分析必须自己按 CSV 的 `epoch_s` 裁到两卡交集**。

⚠️ **读报告的 nsys 必须 ≥ 产出版本**。本机 PATH 里是 `/usr/local/cuda-12.8/bin/nsys`（2024.6.2，**比报告旧**），
拿它读会报 `Please update your Nsight Systems…`；`/usr/local/cuda/bin/nsys` 才是 2025.5.2。
所以 `host_nsys()` **枚举所有候选按版本号取最新**，不看 PATH 顺序。想固定：`NSYS_HOST_BIN=<路径>`。

---

## 二、后处理侧：四步流水线与实现口径

```
① post_export.py          report.nsys-rep ─> post_metrics.csv + post_meta.json   纯标准库/宿主机
② post_plot.py overview   ①              ─> <iface>_overview.png                matplotlib/容器
③ dram_analyze.py         ①              ─> dram_stats.txt + dram_analysis.json numpy/容器
④ post_plot.py detail     ①③             ─> <iface>_trace_*.png                 matplotlib/容器
```

**命名规则**：`post_*` = profile 之后的**通用**后处理（导出的不只 DRAM，组里采到的每条序列都在
CSV 里，所以不叫 `dram_*`）；`dram_analyze.py` 保留 `dram_` 前缀是因为它**是** adhoc 的那一步 ——
主周期以 `dram_total` 为尺，那是当前负载（LLM 推理）的性质。**名字本身就是"哪步能通用、哪步得改"的标记。**

### 2.0 依赖关系（动手改之前先看这张图）★

```
                        post_io.py                       ← 契约层
        （文件名 / CSV 读写 / 接口口径 IFACE / 分母 / PX_PER_POINT）
              ↑              ↑                ↑            纯标准库；不 import 任何同目录脚本
    ┌─────────┘              │                └─────────┐
post_export.py         dram_analyze.py              post_plot.py
  ① 导出                 ③ 分析（adhoc）              ②④ 画图
    ↑                          └── dram_analysis.json ──→ ④（**文件**契约，不是 import）
check_report.py
 (host_nsys / export_sqlite)
```

- **三个 CLI 都只 import `post_io`，彼此互不 import。** ④ 要的主周期是**读 `dram_analysis.json`**
  拿的，不是 `from dram_analyze import detect_period` —— 否则通用的画图脚本会挂在 adhoc 的分析
  脚本上，"哪步能通用"这件事立刻糊掉。代价是 `detail --periods` 必须先跑 ③（脚本会直接报出这句话）。
- `post_export.py` 额外 import `check_report.py`：复用 `host_nsys()`（按版本挑能读报告的 nsys）
  和 `export_sqlite()`。方向是 ① → check_report，反向没有。
- **加一个接口家族**：`post_io.IFACE` 加一条（②③④ 同时生效）+ `post_export.METRIC_COLS` 加白名单列。
- **加一种分析**：只动 `dram_analyze.py`。**加一种画法**：只动 `post_plot.py`。
- **谁在容器里跑**：③④ 要 numpy/matplotlib → 容器；① 纯标准库 + 要调宿主机 `nsys export` → 宿主机。
- `post_io` 里那几个吃 `np` 参数的函数（`as_arrays`/`derive_family` 的调用方传 numpy）是为了
  让契约层自己**不 import numpy** —— ① 在宿主机跑，宿主机装不了 numpy（`pip install` 会 OOM）。

**为什么是四步**（原来揉在一个 `gpu_metrics.py` 里，是 ad-hoc 的来源）：
`detect_period()` 是**分析**原语却埋在画图代码里；画图为了拿副标题去 **grep `dram_stats.txt` 的
注释行**（硬耦合）。拆开后上下文靠 `post_meta.json` / `dram_analysis.json` 显式传递。
**②④ 都是画图但分成两步**：一步回答"整段哪里在忙"（压桶、一张）、一步回答"循环内部长什么样"
（逐点、劈段、要周期）—— 参数默认值正好相反（`--bins 1200` vs `0`），混在一个命令里必然要么
画出混叠的锯齿当周期读，要么闷头输出几百张图。

**CSV 自描述**：值列一律 `*_pct` 结尾，`post_io.load_csv()` 按表头动态发现列 →
① 换 metric 组（加 NVLink/PCIe 曲线）时 ②③④ 不用改。

**每次都重导 sqlite**（① 默认行为）：目录里现成的 `report.sqlite` 可能是上一版报告留下的，
`.nsys-rep` 换了它不会跟着变，而 sqlite 里读不出这件事 —— 静默拿旧数据出结论比慢十几秒糟得多。
导到临时文件、用完删（`--reuse-sqlite` 可关掉）。实测 200 kHz×2 卡×3 s 报告：重导+写 CSV 共 ~17 s。

### 2.1 数据口径（查表 + 实测钉死，别臆测）

- **`total = read + write`（仅 DRAM）**：两条都是 `pct_of_peak_sustained_elapsed`，
  **共用同一个 HBM 峰值分母**（NVIDIA 自己的 `sets/dram.config` 把这俩标成 `type: stacked`）。
  实测 91261 个采样点里 `read+write` 最大 99、**一个 >100 的都没有**，佐证分母确实同一个。
  GB/s 用 `TARGET_INFO_GPU.memoryBandwidth`（本机 3352.32 GB/s）换算。
- ★ **NVLink 一个方向 = 四路相加**：`(request + response) × (user + protocol)`。
  同方向那 4 路共用一个分母（`sets/nvlink.config` 把 RX 4 路 / TX 4 路各标成 `type: stacked`），
  protocol 是链路上**真实占掉**的字节（包头/ACK/credit）→ 算"这条链路被占了多少"必须带上它。
  实测 `nvl_write_fp8`（`runs/nsys_nvlink_nvl_write_fp8_200k__20260729-230253`）：
  发送卡 tx = 69.57(user) + 15.58(proto) = **85.15%**，接收卡 rx 同为 85.15%；
  **只算 user 会少报 18%**。（`user/(user+protocol)` = 链路有效率，那是
  `experiments/nvlink_bench/nvlink_user_vs_proto.py` 的活，不混进这条曲线。）
- ★ **双向 total = `(rx + tx) / 2`，不是相加**（NVLink 和 PCIe 都是）：收发是**物理分离的
  双向链路**、各有各的分母，相加会出 >100% 这种没有物理意义的数。同一份实测数据：
  tx 85.15% / rx 9.84%，相加 95% 会被读成"链路快满了"，其实是**一个方向**快满 ——
  取均值（47.5%）才是"这条链路两个方向的平均占用"。**逐个采样点算**（CSV 一行一算），
  不是先取均值再合：后者会把"两个方向不同时忙"抹平。
  PCIe 那两条计数器**各自归一化**这件事有实测支撑：`pcie_calib` 单向 H2D 55.5 GB/s payload
  读到 RX 91.9% —— 若共用一个双向分母（126 GB/s）只该读到 44%。
  → 口径写在 `post_io.IFACE`（`dirs` = 同方向相加哪几路，`total` = `sum` 还是 `mean`），
  ②③④ 全走 `derive_family()`，所以统计表里的数和图上的线**永远是同一个定义**。
- **每条列用自己的分母换 GB/s**（`post_io.peak_for()` 按列名查，查不到就**不换算**）：
  `dram_*` 用报告里的 `memoryBandwidth`（本机 3352 GB/s）；`pcie_*` 用 **63.02 GB/s/方向**
  （Gen5×16 规格值，见下）。标定值写在 `post_io.IFACE_PEAK_GBPS`，**本机专属，换机器必须重标**。
- ★ **NVLink 那组百分比的分母本仓库不定数 → 不换算 GB/s，只报 `% of peak`**（`peak_for()` 返回 0）。
  分子的定义是清楚的（NVIDIA 本机 UserGuide 对 `nvlrx__bytes`：*"the ratio of bytes received on the
  NVLink interface to **the maximum number of bytes receivable in the sample period**… This value
  **includes protocol overhead**."*），分母那个"最多能收多少"是多少 GB/s 不往代码/文档里写死。
  **与分母无关的量照常可信**：`user/(user+protocol)`（链路有效率）、rx vs tx 的相对高低、
  同一负载不同相位的对比 —— 要绝对带宽用 workload 自测或 DCGM 字节计数器。
- **PCIe 分母 = 63.02 GB/s/方向**（Gen5 × 16 规格：32 GT/s × 16 / 8 × 128/130；
  本机 `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current` = 5, 16 确认）。
  已用 `experiments/cumemcpy/workloads/pcie_calib.py` 双向反推核对：H2D payload 55.50 ÷ RX 91.936%
  = 60.37（占 63.02 的 95.8%）；D2H 39.52 ÷ TX 66.540% = 59.39（94.3%）。
  ⚠️ **反推是一个方程两个未知量**（PCIe 没有 user/protocol 拆分），这里取规格峰值，
  残差 4.4–6.1% 就是协议开销 —— 大 TLP 下 PCIe 头部开销正是这个量级，自洽。
  ⇒ 换算出来的是**线上字节（含协议）**，正是这个计数器的定义；真实 payload 比它低约 5%。
- **值是整数百分比 0–100**（官方 UserGuide 明写；实测 `nvlink_tx_req_proto` 99.9% 的样本恒等于 `5`）。
  → **小数值的相对精度很差**：protocol 读到 5 时，绝对粒度 1 个百分点 ≈ 4.8 GB/s，即 ±10%。
  有效率这种比值受影响小（±0.5pp），但**别拿 protocol 的绝对值去反推包格式的精确字节数**。
- ⚠️ 官方 UserGuide 的 Limitations 原文：*"If metric sets with NVLink are used but the links are not
  active, they may appear as fully utilized."* —— **链路没起来时可能显示成满载**，
  看到"空闲却 100%"先确认链路状态，别当成真流量。
- **`METRIC_COLS`（`post_export.py`）是白名单**：不在表里的 metric 采到了也不进 CSV。
  现覆盖 `sets/` 下所有组用得到的序列（DRAM/SM/GR + NVLink 8 路 + PCIe 2 路）。
  下游 `post_io.load_csv()` 按 `*_pct` 后缀动态发现列，**加列不用改 ②③④**。
- **卡号**：`GPU_METRICS.typeId` 的**低 32 位** = nsys 侧 GPU 号（高位是 vmId，**别拿整个值当卡号**），
  join `TARGET_INFO_GPU.id`。物理卡号取 `run_meta.json` 的 `gpus`+`gpus_nsys` 配对 ——
  那是 nsys-tool 在**宿主机**写的；**不能用容器里的 `nvidia-smi` 反查**（容器内会重编号）。
  实测 2 卡报告出 `0x…00000000`/`0x…00000001` 两个 typeId，对上 id 0/1，按 uuid 查正好是物理卡 1 和 3。
- **时间**：`epoch_s = TARGET_INFO_SESSION_START_TIME.utcEpochNs/1e9 + timestamp/1e9`。
  CSV 相对时间和绝对 epoch 都给，方便和 `marks.txt` / bench 日志对齐。
- `nsys stats` **没有** GPU Metrics 的内置 report（`--help-reports` 一条都没有）→ 只能直接读 sqlite 表。
  表结构在本地官方文档 `/usr/local/cuda/nsight-systems-*/documentation/` 里核对过。
- **① 的产物 2026-07 前叫 `dram_metrics.csv` / `dram_meta.json`**：`post_io` 读的时候两个名字都认
  （`LEGACY_*_NAME`），写一律新名 → 已归档的 `runs/` 和 `experiments/*/logs.tar.gz` 不用重新导。

### 2.2 mean vs median：必须两个都报 ★

| 读法 | decode-hi 实测 | 什么意思 |
|---|---|---|
| p50 / 色块顶边 | **92%** = 3084 GB/s | 在搬数据时的**瞬时典型值**（nsys-ui 里肉眼读到的就是这个）|
| mean | **71.3%** = 2389 GB/s | **平均带宽**（总字节/总时间），把 kernel 间空隙也算进去 |

差距来源：**纯平台段（不含 step 间凹陷）里仍有 13% 的采样点 <10%** —— 单 kernel 中位 3.4 µs，
5 µs 采样正好逮得到 kernel 之间的空隙。**只报一个必然被误读**成"只跑到 71%"或"一直满载"，
所以子图标题和 `dram_stats.txt` 两个都打，`analysis.json` 还额外给 `frac_below_10pct`。

### 2.3 每像素画什么：这是显示口径，不是数据问题 ★

91261 个点塞进 ~1300 像素 = 每列 70 个点，**必须**有规则决定这列画什么，逃不掉。
- nsys-ui 画**填充面积**（`type: stacked`）→ 眼睛读**色块顶边** ≈ 92–98%，就是用户记忆里的"90%多"。
- 早期版本画**每列均值线** ≈ 73% → 看起来像"只跑到 73%"。**同一份数据，不同的每像素规则。**

**所以 ④ `detail` 默认逐点原样、不聚合**（`--bins 0`），糊不糊靠加像素解决，不靠聚合。
② `overview` 反过来：默认 `--bins 1200` 压成桶均值 —— 它回答的是"整段哪里在忙"，
那个尺度上桶均值才是想要的。**代价是箱宽（1200 箱 ≈ 2 ms）比 decode 循环（~530 µs）还长，
总览图上的锯齿是混叠、不是周期**；这也是 ②④ 必须分成两个子命令的直接原因。

**不要画 min/max 包络**：实测 0.3 ms 桶内 `dram_total` 的 max−min 中位数就有 **96**
（kernel 边界上是 0↔100 方波），包络等于把整张图重新涂满，等于没画。

### 2.4 `PX_PER_POINT = 4` 的标定（`post_io.py` 里的常量）★

```
轴区像素 = 图宽(in) × dpi × 0.85(AXES_FRAC，扣页边+右侧 GB/s 轴) × --rows
一张图放得下的点数 = 轴区像素 / PX_PER_POINT
```

| 每点像素 | 观感 | 出处 |
|---|---|---|
| 52 px/点 | 一坨色块，只能读包络顶边 | 91261 点画一张 16in 图 |
| 2.05 px/点 | 能看出形状，但挤 | `--split 100`，4.56 ms/图 ≈ 914 点 |
| **3.6–4.0 px/点** | **清楚，单 kernel 的读/写平台分得开** | ← **定这个** |

⇒ 20in@110dpi 每图约 **470 点**；200 kHz 下就是每图约 **2.3 ms**。
两条路互相印证过：`detail --split auto` → 466 点/4.0 px；`detail --periods 5` → 519 点/3.6 px。**收敛，标定成立。**

### 2.5 周期检测（`dram_analyze.detect_period`）

自相关在基频的**整数倍上都会出峰**，所以取「相关最强的那批峰（≥best×0.9）里最小的那个」当基频，
否则会把 2 倍频当周期。实测 decode-hi 基频 519.2 µs (r=0.841)，1043/1567/2087/2611/3130 µs 全是谐波。

**尺（`--period-ref`）是负载给的，不是通用事实** —— 这也是为什么只有这一步叫 `dram_*`：
LLM 推理里每个 decode step 把权重过一遍 HBM，所以 `dram_total` 是唯一稳定的尺（r=0.85–0.91）；
NVLink/PCIe 太突发（94% 的采样点 <10%），拿它们自相关测不出 decode 循环。换成 collective 压测
之类的负载，DRAM 可能一直平而 NVLink 才有节奏 → `--period-ref nvlink_total`（实测 nvlink_bench
的相位循环用 `nvlink_total` 测出 3299 µs、r=0.874）。**没有周期结构的负载**（prefill、一次性拷贝）
本来就测不出 —— 那时 ④ 用 `--split auto` 按 4 px/点切，别硬套 `--periods`。
脚本里 `pick_period_ref()` 的退路顺序：`--period-ref` → 各家族的 `*_total` → 随便一条采到的；
**实际用了哪一列会写进 `dram_stats.txt` 和 `dram_analysis.json` 的 `period_ref`**，读数前先看它。

### 2.6 其它渲染约定

- 采样**断块**（>1 ms，即 overflow 造成的空洞）在图上**断开不连线**，免得直线横跨空洞看着像"一直在跑"。
- 多卡窗口里两卡采样区间**本来就可能不齐**（各自采样器起停时刻不同，diagnose 仍判 PASS）→
  图上两格子图的线起止位置不同，**不是 bug**。
- 劈段后某卡在某段可能一个点都没有 → `axes[0]` 空，取图例要**遍历找第一个有内容的格子**
  （否则 `fig.legend(ncol=0)` 直接崩）。断块落在 `--overlap` 交集**里面**时同样会画出空格子，
  那是真没数据（`dram_stats.txt` 的「采样断块」一节会列出位置）。
- 容器无中文字体 → 图内文字一律 `ascii_only()`（含图例：NVLink 图例里写的是实际相加的那几路）；
  中文留控制台/`dram_stats.txt`/README。
- 容器默认 root → 产出 PNG 是 root 属主、宿主机改不动。**docker run 必须带 `-u $(id -u):$(id -g)`**，
  配套 `MPLCONFIGDIR=/tmp/mpl`（非 root 进去 HOME 不可写，matplotlib 字体缓存要落地）。

### 2.7 `detail --combined`：三接口叠一张图的几个决定 ★

同一格子图里同时画 DRAM/NVLink/PCIe 时，**信息维度靠两条正交通道编码，别混**：

- **颜色 = 接口，线型 = 方向**：`--lines dir` 下每个接口画三条 —— **total 实线·接口原生色**、
  **进 GPU 虚线**、**出 GPU 点线**；方向线的颜色往白里调一档（`_tint`，`DIR_TINT=0.55`），
  **只有 total 是原生色**，一眼先看到总量、进/出是挂在它上面的浅色拆分。
  颜色取 **dataviz 技能验证过的分类色前 3 槽**（`IFACE_COLOR`，白底 light 列：蓝 `#2a78d6` /
  橙 `#eb6834` / 青 `#1baf7a`）——这三色两两 CVD ΔE≥9.2，色盲下也分得开；**按 `post_io.IFACE`
  固定顺序配色、不轮换**（加接口/换机器顺延，不重排，免得"同一接口换了次跑就换色"）。
  DRAM 没有真正的"进/出 GPU"（HBM 在片内），这里把 **read 当进、write 当出**，纯粹是复用同一套线型约定。
- **图例拆两条正交通道、只占 1 行**：彩色实线块 = 接口（+dual 时的 `[L]/[R]`），灰色线型键 =
  方向（solid=total / dashed=进 / dotted=出）。9 条线全列会挤成 3 行糊住 x 轴标签 —— 拆通道后
  读者自己组合"蓝+虚线=DRAM read"。total 一直画，所以也**一直参与 dual 的 auto-scale 定尺**
  （DRAM 的 total=读+写会超过任一方向，不带上会被顶到轴外）。
- **口径不在 `post_plot` 里另立**：total/rx/tx/read/write 全走 `post_io.derive_*`，与逐接口图、③ 统计表
  **逐点同源**（`--lines total` 画的 `nvlink_total` 就是 `(rx+tx)/2`，不是这里现拼的）。

**两根纵轴（`--yaxis dual`，默认）是有意违反"一张图别放两个 y 轴"这条通用制图铁律的**：
DRAM 常年满载、NVLink/PCIe 多数时刻 <2% 只偶发尖峰（实测 iface 窗口 DRAM p50 89% vs NVLink mean 1.7%），
同一根 0–100 轴上互联全贴地板、看不出形状。所以给了 dual（左 DRAM / 右 NVLink+PCIe，**各自 auto-scale**）
把互联撑开——代价就是那条铁律的坑（两轴容易被误读成同尺度）。**安全档 `--yaxis aligned` 始终在**：
同一根 0–100 轴、直接比"离各自峰值多远"。**都只用 % util，从不换 GB/s**（NVLink 分母本仓不定数，
且三家分母差上百倍，混一根 GB/s 轴没意义）。

- **dual 的尺取全窗口 `[X0,X1]` 峰值、各段共用**（不是每段各自撑满）→ 段与段之间可比：某段 NVLink 静默时
  右轴不会把它那点噪声放大成满屏。实现见 `plot_one` 的 `fam_max()`。
- **缺一侧就退单轴**：纯 NVLink 组（无 DRAM）或只有一种接口时没有第二根轴；此时若用户点的是 dual，
  单轴**仍按数据 auto-scale**（`scale["single_auto"]`），别被 `--ylim peak` 压回 0–100 让 18% 的线贴地板。
- **`--combined` 忽略 `--style`**（area 是 DRAM 专属堆叠，合并视图自带线型）；`--metric` 仍生效，
  但想"只叠 DRAM+NVLink 不要 PCIe"目前靠采集组本身不含 PCIe（`nvlink` 组），没单开 metric 子集开关。
- 每段只出**一个** `interface_trace_*.png`（不再逐接口分文件）；`MAX_AUTO_FILES` 的段数上限照旧。

---

## 三、坑速记（均为本机实测）

- **别自己直接 `nsys profile -d N`**：`--kill` 默认 `sigterm`，窗口一到就把主任务杀了。
  本工具统一走 `launch/start/stop` 绕开（也因此 `-t` 才能是浮点秒）。
- **`--keep N` 是"至少 N 秒"不是"正好 N 秒"**：12 秒窗口 `--keep 3` 留下 4.85 秒（按缓冲块粒度截断）。
  ⚠️ **子秒级 `--keep` 会把数据全丢光**：2 秒窗口 `--keep 0.2` → **根本不写报告**（只剩 `marks.txt`，rc=0）。
  想要短窗口用 `-t`，别用 `--keep`。工具在 `--keep < 2` 时会先 WARN。
- **一条数据都没采到时 nsys 静默不写报告**（rc 仍是 0）。典型：`cuda_lite` 组配一个不跑 CUDA 的程序。工具会 WARN。
- **nsys ↔ DCGM 抢硬件计数器**：别对同一张卡同时跑 `nsys-tool` 和 `profile.sh --backend dcgm`。→ metrics_reference §3.2
- **报告不小**：10 kHz × 5 秒 × 2 卡 ≈ 1.5 MB；加 CUDA trace 涨到 3.7 MB（`cuda_lite` 只 300 KB）。
  长任务用三段式 + `--keep`。
- **`perf_event_paranoid` 重启回到 4**（没做持久化）。回到 4 后 `cuda_lite` 自动开始要 sudo ——
  工具每次重读该值，不会坏，只是不再免密。固定住：
  `echo 'kernel.perf_event_paranoid = 2' | sudo tee /etc/sysctl.d/99-nsys.conf`（**共享机，改前知会同机的人**）。
- **容器 nsys 版本 2026.3.1 产出的报告打不开在宿主机 2025.5.2 GUI 里**。解法是把宿主机
  nsight-systems 挂进容器 + `NSYS_BIN` 指过去（`tool/sglang_bench` 的 `profiler: nsys` 已自动做）。
