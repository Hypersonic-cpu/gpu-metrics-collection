# tool/nsys —— Nsight Systems 采时间线 + 出图（操作手册）

采 **GPU 硬件轨（µs 级）/ CPU 轨 / CUDA kernel 甘特图**，再把 GPU Metrics 轨拉成 CSV、统计、画图。
和 `profile.sh` 的两个后端是不同赛道：那两个是旁路定时采样、10 Hz 地板、逐行 CSV；这里产二进制 `.nsys-rep`。

```
tool/nsys/
├── nsys-tool             # 采集入口
├── check_report.py       # 判报告能不能用（nsys-tool diagnose 转发到它）
├── post_export.py        # ① 导出：report.nsys-rep -> post_metrics.csv + post_meta.json
├── post_plot.py overview # ② 全局图：CSV -> <iface>_overview.png
├── dram_analyze.py       # ③ 分析(adhoc)：CSV -> dram_stats.txt + dram_analysis.json
├── post_plot.py detail   # ④ 细节图：CSV + ③ 测的周期 -> <iface>_trace_*.png（--combined -> interface_*.png：三接口叠一张）
├── post_io.py            # ①②③④ 共用的契约：文件名 / CSV 读写 / 接口口径（不是 CLI）
├── groups/<name>.conf    # 组文件：这一遍四条轨各采不采、用哪个 metric set、多少 Hz
└── sets/<name>.config    # nsys 原生 metric set：GPU 计数器里挑哪几条曲线
```

> - **为什么这么设计 / 口径与实测澄清 → [`CLAUDE.md`](CLAUDE.md)**（scope 限制、溢出统计、画图口径、标定依据）
> - 某次实验的数值和解读 → 本目录 `EXP_*.md`
> - 什么时候用 nsys 而不是 dcgm → `docs/metrics_reference.md` §6
> - ⚠️ 别对同一张卡同时跑本工具和 `profile.sh --backend dcgm`（抢硬件计数器）

## 依赖与权限

| | 位置 | 版本 |
|---|---|---|
| 宿主机 | `/usr/local/cuda/bin/nsys`（**不在 PATH**，脚本自动找最新的） | 2025.5.2 |
| `lmsysorg/sglang:*` | `/usr/local/bin/nsys` | 2026.3.1 |
| `nvcr.io/nvidia/pytorch:25.12-py3` | `/usr/local/cuda/bin/nsys` | 2025.5.2 |

| 要采的 | 本机现状 | 结果 |
|---|---|---|
| GPU Metrics | 驱动 `RmProfilingAdminOnly=1` | **恒需 root**（脚本自动加 sudo，本账号免密）|
| CPU 采样 `process-tree` | `kernel.perf_event_paranoid=2` | ✅ 免 root |
| CPU 采样 `system-wide` | 同上（要 ≤0）| 需 root |
| CUDA trace（kernel 甘特图）| — | ✅ 免 root |

---

# 一、采集：`nsys-tool`

```bash
export PATH="$PWD/tool/nsys:$PATH"

# ① attach：不碰目标进程，对已经在跑的东西开个窗口（最常用）
nsys-tool -i 6,7 -t 10                       # 采 10 秒
nsys-tool -i 6,7 -t 10 -y 60                 # 等 60 秒再采 10 秒
nsys-tool -g nvlink -i 6,7 -t 10             # 换组，只看 NVLink

# ② wrap：把命令跑在 nsys 下，能拿 kernel 甘特图；采完主任务继续跑到自然结束
nsys-tool -g cuda -i 6,7 -t 6 -y 8 -- python train.py

# ③ 三段式：窗口起止由外部条件决定，报告最小
nsys-tool launch -g cuda -i 6,7 -- python server.py   # app 起来，暂不采
nsys-tool gen -t 5 --name decode-hi                   # 到点了才采 5 秒
nsys-tool start; nsys-tool stop --keep 30             # 或手动卡起止

nsys-tool sessions | status | cancel | shutdown       # 透传 nsys 同名子命令
nsys-tool diagnose runs/<run 目录>                     # ⚠️ 采完必跑，见下
```

`gen` = `start` → `sleep t` → `stop`。**三段式里 `-g`/`-i`/`--in-container` 只在 `launch` 写一次**，
后面 `gen`/`start`/`stop` 自动继承（显式再给一次就覆盖）。

⚠️ 时间尺度（详见 [CLAUDE.md §1.2](CLAUDE.md)）：窗口**地板约 1 秒**；`gen` 从发命令到窗口真正打开
有 **5–9 秒**延迟；`stop` 写报告要 **10–22 秒** → **窗口之间至少留 ~30 秒**。
实际落点以 `marks.txt` 的 `window_start` 为准。

## 参数

| 参数 | 含义 | 默认 |
|---|---|---|
| `-i, --gpus 6,7` | **物理** GPU 编号（= `nvidia-smi` index）；也可 `all`/`none`/`cuda-visible` | `all` |
| `-t, --time N` | 采集窗口**秒数，浮点**。`gen`/wrap 必填 | — |
| `-y, --delay N` | 窗口开始前的延迟秒数，浮点 | `0` |
| `-g, --group NAME` | metric 组 → `groups/<NAME>.conf` | `pass1_core` |
| `--freq HZ` | GPU Metrics 采样频率（10–200000），覆盖组文件 | 组文件（10000）|
| `--trace LIST` | 覆盖组文件的 trace 列表 | 组文件 |
| `--name TAG` / `--note "…"` | run 名 / 备注 | — |
| `-o, --out DIR` | 输出根目录 | `<repo>/runs` |
| `--in-container N` | 在指定容器里跑 nsys | 空 = 宿主机 |
| `--keep N` | 仅 `stop`：只保留 stop 前 N 秒（治报告过大）| 全留 |
| `--stats` | 采完导出 `report.sqlite`（+ `stats/*.csv`）| off |
| `--session NAME` | session 名 | `nsystool` |

⚠️ **`-t` 在本工具是「时长」，不是 nsys 原生的 `--trace`**。
环境变量 **`NSYS_BIN=<路径>`** 指定用哪个 nsys 二进制；**`NSYS_HOST_BIN`** 指定读报告用哪个。

## metric 组

组文件 `groups/<name>.conf` 是 `key = value` + `#` 注释，只写「**采什么**」；
「采多久/何时起停」由 `-t`/`-y` 控制。key 按 scope 分两段：

```ini
# ── Application scope（拼到 nsys launch）──
trace            = cuda,nvtx,osrt
cuda_graph_trace = node                          # 用 CUDA graph 时必须 node，否则看不到图里的 kernel
# ── System scope（拼到 nsys start）──
sample           = none
cpuctxsw         = none
gpu_metrics_set  = file:sets/pass1_core.config   # 也可填内置 gh100 / gh100-ct / none
gpu_metrics_freq = 10000
```

`file:` 后的相对路径**相对 `tool/nsys/`**（不是相对 `groups/`）。加一组 = 丢一个 `.conf`，不用改脚本。

| 组 | GPU Metrics | trace | CPU 采样 | 要 sudo | attach | 序列/卡 | ≈ dcgmi |
|---|---|---|---|---|---|---|---|
| **`pass1_core`**（默认）| GR/SMs + DRAM 读写 + NVLink RX/TX + PCIe RX/TX | — | — | ✅ | ✅ | 10 | `pass1_core.txt` |
| `dram` | GR/SMs + DRAM **读/写分开** | — | — | ✅ | ✅ | 4 | 1005 |
| `nvlink` | GR + NVLink RX/TX 全 8 路 | — | — | ✅ | ✅ | 9 | 1011/1012+449 |
| `pcie` | GR + PCIe RX/TX | — | — | ✅ | ✅ | 3 | 1009/1010 |
| `full` | 内置 `gh100` 全集 | — | — | ✅ | ✅ | 30 | — |
| `timeline` | 内置 `gh100` | — | system-wide + ctxsw | ✅ | ✅ | 30 | — |
| `cuda` | 内置 `gh100` | `cuda,nvtx,osrt` | process-tree | ✅ | ❌ 报错 | 30 | — |
| `cuda_lite` | 无 | `cuda,nvtx` | process-tree | ❌ **免 sudo** | ❌ 报错 | 0 | — |
| **`cuda_dram`** | DRAM 读写 **@200 kHz** | `cuda,nvtx` | — | ✅ | ❌ 报错 | 4 | 1005 |
| **`cuda_iface`** | DRAM + NVLink(user) + PCIe **@200 kHz** | `cuda,nvtx` | — | ✅ | ❌ 报错 | 10 | `pass1_core.txt` |
| **`cuda_iface_full`** | DRAM + NVLink **全 8 路** + PCIe **@200 kHz** | `cuda,nvtx` | — | ✅ | ❌ 报错 | 14 | `pass1_core.txt`+ |

- `cuda_iface_full` = **三个接口全采，首选**（nsys 一次 session 只能给一个 `--gpu-metrics-set`，
  三个接口没法靠多组拼，只能合成一个 set）。NVLink 全 8 路 = request/response × user/protocol × RX/TX，
  **只有带 protocol 才算得出链路有效率** `user/(user+protocol)`（实测 sglang TP2 decode ≈ 85%）。
  PCIe 侧就 `pcie__{read,write}_bytes` 两条，协议开销已混在里面、拆不开，没有可补的。
- `cuda_iface` = 同上但 NVLink 只有 user 4 路（10 序列）。两者**溢出地图完全一样**，
  `full` 多 4 路没有额外代价 → 默认用 `full`，只在想让报告更小时才用它。
- ⚠️ **两者的「200 kHz × 双卡」都必坏**（4/4 窗口，且坏得不明显）；要双卡就 `--freq 100000`，
  要 200 kHz 就 `nsys_gpus` 只给一张卡。**瓶颈是采集设备数不是序列数**：
  序列 4→10→14 都没翻掉能过的格子，减一张卡立刻从全坏变全过。
  两轮 2×2 实测 → [`EXP_iface_freq_card.md`](EXP_iface_freq_card.md)。
- `cuda_dram` = **刻意保留的 200 kHz 高分辨率档**：kernel 甘特图 + HBM 读写，5 µs/点，
  窗口固定开销从 6.8 s 压到 0.95 s。采的是 `cuda,nvtx` trace（HW 轨 + API 轨一起，甩不掉）
  + 4 条 GPU Metrics（GR/SMs Active + DRAM 读/写），没有 CPU 采样。
  ⚠️ **200 kHz 在重负载下可能间歇溢出**（多进程 + 海量 CUDA trace 时，profiler 排空不过来），
  所以**采完一定要 `nsys-tool diagnose`**——溢出只写进报告内部，不查看不见。
  真溢出了再按当次情况调（降到 `--freq 100000`、或少采一张卡），依据见 [CLAUDE.md §1.7](CLAUDE.md)。
- `cuda_lite` = **唯一免 sudo 的组**（kernel 甘特图 + CPU 轨，不采 GPU Metrics）。

## 在容器里用

```bash
# 容器必须带 --cap-add SYS_ADMIN，否则采不了 GPU Metrics / CPU
# 想让报告能被宿主机 GUI 打开，把宿主机 nsight-systems 挂进去 + NSYS_BIN 指过去
docker run -d --name myctr --gpus '"device=6,7"' --cap-add SYS_ADMIN \
    -v /usr/local/cuda-13.1/nsight-systems-2025.5.2:/opt/nsys-host:ro <image> sleep infinity

export NSYS_BIN=/opt/nsys-host/target-linux-x64/nsys
nsys-tool gen --in-container myctr -i 7 -t 5      # -i 仍然给【物理】编号，工具按 PCI bus id 自动换算
```

`tool/sglang_bench` 的 `profiler: nsys` 模式已自动做了挂载 + `NSYS_BIN`。

## ⚠️ 采完必须判一次

nsys 把采集期 Error **只写进报告内部**，stdout 一个字不打、`rc` 仍是 0 → 不查就会拿废数据分析。

```bash
nsys-tool diagnose runs/<run 目录>        # 递归判目录下所有 report.nsys-rep
tool/nsys/check_report.py <目录> --json   # 机器可读（diagnose 就是转发到它）
```

| 判定 | 含义 | 退出码 |
|---|---|---|
| **PASS** | 没有任何 Error | 0 |
| **PARTIAL** | 报了 overflow，但**至少一路**仍完整可用（该路照常分析，作废那路别用）| 1 |
| **FAIL** | 没有任何一路可用；或没有一路覆盖到 kernel 段（窗口开早/开晚）| 2 |
| **ERROR** | overflow 以外的 Error | 2 |

每个报告目录会落一份 `diagnostics.txt`。`tool/sglang_bench` 的 nsys 模式跑完自动判一次。

## 采集产出什么

`runs/nsys_<组>[_<name>]__<时间戳>/`（多窗口时在 `windows/<窗口名>/`）：

| 文件 | 是什么 |
|---|---|
| `report.nsys-rep` | 主产物，`nsys-ui` 打开 |
| `report.sqlite` | `--stats` 时；也是出图流水线的输入 |
| `stats/*.csv` | `--stats` + 有 trace 时，`nsys stats` 摘要（`s_cuda_gpu_kern_sum.csv` 最常用）|
| `marks.txt` | `epoch\t标签`：`app_launch`/`start_cmd_begin`/`window_start`/`window_end`/`stop_cmd_done`/`session_stop` |
| `run_meta.json` | 组/卡/时长/是否 sudo/容器/nsys 版本/`workload_rc` |
| `diagnostics.txt` | `diagnose` 的判定 + 每路体检 |
| `README.md` | note + meta + 怎么打开 |

---

# 二、后处理：四步流水线

`post_*` = **profile 之后的通用后处理**（导出的不只 DRAM —— 组里采到的每条序列都在 CSV 里）；
`dram_analyze.py` 是**当前负载专属**那一步：它的主周期是拿 `dram_total` 当尺量出来的。

```
① post_export.py          report.nsys-rep ─> post_metrics.csv + post_meta.json   纯标准库 → 宿主机
② post_plot.py overview   ①              ─> <iface>_overview.png                matplotlib → 容器
③ dram_analyze.py         ①              ─> dram_stats.txt + dram_analysis.json numpy      → 容器
④ post_plot.py detail     ①③             ─> <iface>_trace_*.png                 matplotlib → 容器
                          ↑ 四步共用 post_io.py（文件名 / CSV 读写 / 接口口径，不是 CLI）
```

②③ 都只依赖 ①，先跑哪个都行；**④ 的 `--periods` 要 ③ 的 `dram_analysis.json`**
（"负载有没有周期"是分析结论，画图脚本不自己重测一遍）。

四步都吃同样的 `<路径>`：**window 目录 / run 目录（递归找每个 window）/ 文件本身**，
`-o OUT` 把产物写到别处（落 `<OUT>/<窗口名>/`），不给就写回原目录。

```bash
# ① 宿主机（纯标准库）
python3 tool/nsys/post_export.py runs/nsys_xxx

# ②③④ 进容器。-u 别省，否则产出 PNG 是 root 属主、宿主机改不动
DK='docker run --rm -u '"$(id -u):$(id -g)"' -e MPLCONFIGDIR=/tmp/mpl -v '"$PWD"':/work -w /work nvcr.io/nvidia/pytorch:25.12-py3'
$DK python3 tool/nsys/post_plot.py overview runs/nsys_xxx      # ② 全局：一个接口一张
$DK python3 tool/nsys/dram_analyze.py       runs/nsys_xxx      # ③ 统计 / 主周期 / 建议切分
$DK python3 tool/nsys/post_plot.py detail   runs/nsys_xxx --periods 5 --parts random:3   # ④ 细节
```

⚠️ **`-o` 指到仓库外面 = 产物丢在容器里**（`--rm` 一删就没，脚本照样打印 `saved`）。
写到别处就得把那个目录也 `-v` 挂进去。默认不给 `-o`（写回 run 目录）不会踩这个。

## 产物

| 文件 | 谁写的 | 内容 |
|---|---|---|
| `post_metrics.csv` | ① | 逐点宽表：`gpu,gpu_nsys,uuid,t_ms,epoch_s` + 组里采到的每条序列一列 `*_pct`（`dram_{total,read,write}`、`sm_active`、`gr_active`、`nvlink_{rx,tx}_{req,rsp}[_proto]`、`pcie_{rx,tx}`）+ `dram_*_GBps` + `hbm_peak_GBps`。CSV 里的 GB/s 列只有 DRAM；PCIe 的 GB/s 由 ②③④ 用标定分母现算，NVLink 不换算（分母不定数）|
| `post_meta.json` | ① | 窗口名/note/组/频率/每卡口径（峰值、样本数、跨度、实测采样率）—— ②③④ 的上下文来源 |
| `<iface>_overview.png` | ② | 全局图，一个接口一张（`dram_overview.png` / `nvlink_overview.png` / `pcie_overview.png`）|
| `dram_stats.txt` | ③ | 人读：口径行 + 每卡每 metric 的 mean/p50/p95/max(+GB/s)、`<10%` 占比、主周期、建议切分数、断块 |
| `dram_analysis.json` | ③ | 同上，机器读；④ `--periods` 读的就是这里的 `period_us` |
| `<iface>_trace_*.png` | ④ | 细节图，劈段时是 `<iface>_trace_<i>of<N>.png`，`--xlim` 时带毫秒区间 |
| `interface_trace_*.png` | ④ `--combined` | 三接口叠一张（颜色=接口、线型=方向），`--yaxis dual\|aligned` × `--lines dir\|total` |

> 旧 run 里 ① 的产物叫 `dram_metrics.csv` / `dram_meta.json`（2026-07 之前的命名）。
> **读的时候两个名字都认**，所以归档的 `runs/` 和 `experiments/*/logs.tar.gz` 不用重新导。

## 图长什么样

每张 GPU 一格子图（标题 = 卡号 + 该段 mean 和 median），每格**三条线**：
**total（黑）/ 收（红）/ 发（蓝）**。左轴一律 `% of peak`（CSV 里 `*_pct` 列的原始单位），
右轴换算成 GB/s。`--metric` 默认 `all` = CSV 里有的接口各画一张，互不覆盖。

三条线的口径写在 `post_io.IFACE`，**③ 的统计表和 ④ 的线逐点同源**：

| | `dram` | `nvlink` | `pcie` |
|---|---|---|---|
| 一个方向 | read / write 各自一条 | **四路相加**：`(request+response) × (user+protocol)` | 各方向就一条计数器（协议开销已混在里面）|
| `total` | `read + write`（同一片 HBM、共用分母，相加有意义）| **`(rx + tx) / 2`** | **`(rx + tx) / 2`** |
| 右轴分母 | 报告里的 `hbm_peak_GBps`（本机 3352 GB/s）| **无** —— 分母不定数，只给 %| 63.02 GB/s/方向（Gen5×16 规格）|
| 周期尺 | 自己 | 也用 `dram_total` | 也用 `dram_total` |

- **NVLink 一个方向必须四路都算**：protocol 是链路上真实占掉的字节（包头/ACK/credit）。
  实测 `nvl_write_fp8` 发送卡 tx = 69.57%(user) + 15.58%(proto) = **85.15%**，
  只算 user 会少报 18%。图例里会写清楚相加了哪几路；组里没采 protocol（`cuda_iface`/`pass1_core`）
  时标 `(user only!)`，别把那种图当成含协议读。
- **`(rx+tx)/2` 而不是相加**：收发是**物理分离的双向链路**、各有各的分母，相加会出 >100%
  这种没物理意义的数（实测 nvl_write：tx 85% / rx 10%，相加 95% 会被读成"链路快满了"）。
- 想看 `user/(user+protocol)` = 链路有效率：那是
  `experiments/nvlink_bench/nvlink_user_vs_proto.py` 的活，不在这三条线里。

⚠️ **mean 和 median 差很多，别混着读**：decode-hi 实测 median 92%（搬数据时的瞬时典型值）
vs mean 71.3%（含 kernel 间空隙的平均带宽）。图标题两个都打。原因见 CLAUDE.md §2.2。

## ① `post_export.py`

| 开关 | 默认 | 作用 |
|---|---|---|
| `--resample-us N` | `0` | 写 CSV 前按 N µs 取均值给 CSV 瘦身（**会损失原始分辨率**）|
| `--clip-window` | 关 | 只留 `marks.txt` 的 `window_start..window_end` |
| `--reuse-sqlite` | 关 | 用目录里现成的 `report.sqlite` |

**每次都从 `.nsys-rep` 重新 `nsys export` 一份临时 sqlite**（用完删、不留在 run 目录），
即使目录里已经有 `report.sqlite` 也不用 —— 那份可能是上一版报告留下的，`.nsys-rep` 换了它
不会跟着变，拿它导出等于拿旧数据出结论。确定是新的、只想省时间才 `--reuse-sqlite`
（实测 200 kHz × 2 卡 × 3 s 的报告重导 + 写 CSV 一共 ~17 秒）。

## ② `post_plot.py overview`

全局图：**哪一段在忙、平台多高**。默认把整段压成 `--bins 1200` 个桶画均值，一个窗口一张。

⚠️ **总览图上的锯齿不是真周期**：1200 箱把 2.5 s 压成 ~2 ms/箱，比 decode 循环（~530 µs）
还长，画出来是**混叠**。总览只能读"哪一段在忙、平台多高"；**要读循环结构必须走 ④**。

⚠️ NVLink/PCIe 在 LLM decode 下均值只有 1–2%，默认 0–100 的纵轴上线会贴在轴底
（脚本会提示）→ 加 `--ylim auto` 按实际画出来的线取上限。

## ③ `dram_analyze.py`

| 开关 | 默认 | 作用 |
|---|---|---|
| `--periods-for N` | `5` | 顺带算「一图画 N 个循环要劈几张」|
| `--period-ref COL` | `dram_total` | 拿哪一列当周期尺（没采到就退回某个 `*_total`）|

**四步里只有这一步是 adhoc 的**（所以名字还带 `dram_`）：主周期是拿 `dram_total` 当尺量的 ——
那是**当前负载**（LLM 推理，每个 decode step 把权重过一遍 HBM）的性质。换成 collective 压测
之类的负载，尺得换（`--period-ref nvlink_total`），没有周期结构的负载（prefill、一次性拷贝）
本来就测不出 —— 那时 ④ 用 `--split auto` 而不是 `--periods`。
详见该脚本 docstring 的「Notes for future usage」。

## ④ `post_plot.py detail`

**先看这条**：糊不糊只取决于**一个采样点分到几个横向像素**，标定值是 **4 px/点**
（20in@110dpi ⇒ 每图约 470 点；200 kHz 下约 2.3 ms）。脚本每次都打印实测的「每图几个点 /
几 px 每点」，不足 2 px/点会提示该怎么改。依据见 [CLAUDE.md §2.4](CLAUDE.md)。

```bash
--periods 5               # 一张图画 5 个循环（周期读 ③ 的 dram_analysis.json）
--split auto              # 或按 4 px/点自己算劈几张（看不出周期的负载，如 prefill，用这个）
--parts random:3          # ⚠️ 段数 > 24 时必给：只渲染随机 3 张
--parts 3,29,47 / 10-14   # 或指定 / 区间；--parts all = 真要全画
--overlap                 # ⚠️ 多卡窗口必加，见下
```

⚠️ **`--periods N` 会算出成千上万段**（2.1 s ÷ (5×530 µs) = 800+）→ 段数超过 **24** 时脚本
**直接停下**让你用 `--parts` 挑，不会闷头画几百张。

⚠️ **多卡窗口一定要加 `--overlap`**：nsys 给每张卡起停采样的时刻不一样（实测差 0.5–1.3 s），
窗口两端各有一段**只有一张卡有数据**——`--parts random` 会随机撞进去，画出上面一格全空的图。
那不是 bug，原始 `.nsys-rep` 里那段就没有那张卡的采样（已直接读 `GPU_METRICS` 表核对过）。
`--overlap` 把范围限死在各卡交集，并打印交集有多长（实测 2.1–3.5 s 的双卡窗口里交集只有 0.7–1.6 s）。
交集**里面**还可能有 overflow 造成的断块（那一格照样是空的，`dram_stats.txt` 会列出断块位置）。

### `detail --combined`：三个接口叠一张图（`interface_*.png`）

上面 `detail` 是**逐接口各一张**（`dram_trace_*` / `nvlink_trace_*` / `pcie_trace_*`）；加 `--combined`
把 DRAM/NVLink/PCIe 叠进**同一格子图**（仍每卡一格、共用 `--split/--periods/--parts/--overlap`），
产出 `interface_trace_*.png`。**颜色 = 接口**（DRAM 蓝 / NVLink 橙 / PCIe 青），**线型 = 方向**。
画哪几个接口按 CSV 自动判：单卡纯 HBM=只 DRAM；`nvlink` 组=DRAM+NVLink；`iface` 组=三种全有。

```bash
$DK python3 tool/nsys/post_plot.py detail runs/nsys_xxx --combined --overlap --split auto --parts random:2
# 默认 = --yaxis dual --lines dir（下面两个开关都取第二档）
```

| 开关 | 选项 | 含义 |
|---|---|---|
| `--yaxis` | `dual`（默认）| 左轴 DRAM、右轴 NVLink+PCIe，**各自 auto-scale**（都是 % util，不换 GB/s）；DRAM 满载而互联只 1–2% 时靠这个把互联撑开看形状 |
| | `aligned` | 全部同一根 0–100% 轴（`--ylim` 控上限）；直接比"离各自峰值多远" |
| `--lines` | `dir`（默认）| 每接口三条：**总**（实线·接口原生色）+ **进 GPU**（虚线·浅）+ **出 GPU**（点线·浅）。图例拆两通道：彩块=接口、灰键=方向 |
| | `total` | 每接口只画**总利用率**（实线），三色三条，最干净 |

- dual 各段共用**全窗口**峰值当尺 → 段与段之间可比（不是每张各自撑满）。缺 DRAM（如纯 NVLink 组）
  或只有一种接口时 dual 退成单轴，仍按数据 auto-scale。
- `--metric` 仍生效（默认 `all`）：想只叠 DRAM+NVLink、不要 PCIe，目前靠采集组本身不含 PCIe（`nvlink` 组）。
- `--style` 对 `--combined` 无效（合并视图自带线型口径）。

## ②④ 公共开关

| 开关 | 默认 | 作用 |
|---|---|---|
| `--metric dram\|nvlink\|pcie\|all` | `all` | 画哪个接口；产出 `<metric>_*.png`，互不覆盖 |
| `--ylim peak\|auto\|MAX` | `peak`(0–100) | `auto` = 按**实际画出来的线**取上限（NVLink/PCIe 只有 1–2% 时用）|
| `--overlap` | 关 | **多卡窗口必加**：只画各卡采样区间的交集 |
| `--rows N` | `1` | 单个文件内再折成 N 条时间带 |
| `--width` / `--row-height` / `--dpi` | `20` / `2.4` / `110` | 图宽、每带高（英寸）、dpi |
| `--bins N` / `--agg` | ② `1200` / ④ `0` / `mean` | 压点画趋势线（`0`=逐点原样）；`mean\|median\|p95\|max` |
| `--xlim A,B` | — | 只画 A–B 毫秒（另存文件名，不覆盖总览）|
| `--style line\|area` | `line` | line=三条线；area=read/write 堆叠填充（nsys-ui 画法，顶边=total，只对 dram 有效）|
