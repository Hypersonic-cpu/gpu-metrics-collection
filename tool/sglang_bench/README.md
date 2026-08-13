# tool/sglang_bench —— SGLang dummy-weight 性能实验一键编排（操作手册）

把「开容器 → 起 server → 盯 ready → 开采集 → 发 bench → 停采集 → parse/画图 → 改目录名」这套
**多终端手工流程**，收敛成**宿主机上一条命令**，按配置清单挨个自动跑完。

```
tool/sglang_bench/
├── run.sh             # 一键入口（在【宿主机】跑）
├── experiments/       # 配置清单：<短名>.yaml，一份一组实验；archive/ 放历史备份
├── parse_yaml.py      # 配置 -> run.sh 消费的行（run.sh 内部调用，不用手跑）
├── wait_ready.sh      # 轮询 server 日志判就绪/崩溃
├── wait_burst.sh      # 轮询 server 日志等真正的压测流量（nsys 模式的 burst 锚点）
├── align_windows.py   # nsys 窗口 × server.log 对齐成表（自动写进 run 的 README）
└── analysis/          # 跑出来的结果分析笔记
```

> - **为什么这么编排 / 口径与实测澄清 → [`CLAUDE.md`](CLAUDE.md)**（改代码前先读）
> - 采集字段/指标语义 → 仓库根 `README.md` 与 `docs/metrics_reference.md`
> - nsys 本身怎么用 → `tool/nsys/README.md`

---

## 前置（都在 zkrh-58 宿主机上）

- **必须在宿主机跑**，不要在 sglang 容器里跑：它要用**宿主机的 `dcgmi`** 采集 + **宿主机 `docker`** 起被测容器。
- 宿主机需有：`dcgmi`、`docker`（+ nvidia runtime）、`python3` 带 `pyyaml`（`python3 -c 'import yaml'` 能过）。
- 两个镜像（本机已 local，无需拉）：被测 `lmsysorg/sglang:v0.5.15.post1-cu129`、画图 `nvcr.io/nvidia/pytorch:25.12-py3`。
- dummy 权重模型骨架在 `/ssd/yjdiao/models/*-config`（只有 config，`--load-format dummy` 随机初始化，性能真、输出乱码）。

---

## 快速上手

```bash
cd ~/Repos/metric_collection

# 有哪些配置、每份里有哪些实验（不带参数跑也是列这个，不会误跑）
tool/sglang_bench/run.sh --list

# 跑一份配置（短名 = experiments/<短名>.yaml），全自动，人可以走开
tool/sglang_bench/run.sh 2gpu

# 只跑其中某几个实验
tool/sglang_bench/run.sh 2gpu --only qwen3_32b_tp2,qwen3_30b_a3b_tp2_ep2

# 先看每个实验会执行哪些命令（不碰 docker/GPU）
tool/sglang_bench/run.sh llama405b.dramsweep --dry-run

# 半自动：脚本只开好容器 + 打印 server/bench/profile 命令给你手动粘，容器留着
tool/sglang_bench/run.sh deepseek.8gpu --interactive --only dsflash_fp8_tp8
```

| 选项 | 作用 |
|---|---|
| `<短名>` 或 yaml 路径 | 位置参数选配置，**必给**；短名去 `experiments/<短名>.yaml` 找，找不到当路径。**没有隐式默认**——不给就列配置后退出 |
| `--list` | 列出 `experiments/` 下的配置 + 各自的实验名，不跑 |
| `--only a,b` | 只跑指定 `name` 的实验（逗号分隔）|
| `--interactive` | 半自动：开好容器 + 打印命令，容器留给你（**自己 `docker rm -f`**）|
| `--keep-container` | 实验结束不删容器 |
| `--dry-run` | 只打印命令，不碰 docker/GPU |

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `GPU_FREE_RETRIES` / `GPU_FREE_WAIT` / `GPU_FREE_MEM_MIB` | `3` / `5`s / `1024`MiB | 卡空闲检测：重试几次 / 间隔 / 显存阈值 |
| `PLOT_IMAGE` | `nvcr.io/nvidia/pytorch:25.12-py3` | 出图用的容器镜像 |

---

## 配置：`experiments/<短名>.yaml`

`defaults` 给该文件所有实验兜底，实验自身同名键覆盖它。只写「要改的」参数——下面这些**自动注入**，不用写：

- server 自动加：`--load-format dummy` `--enable-metrics` `--tp-size <张数>` `--host 0.0.0.0 --port <port>`
- bench 自动加：`--backend sglang` `--dataset-name random-ids` `--host 127.0.0.1 --port <port>`
- docker 自动加：`--gpus '"device=<gpus>"'` `--shm-size 32g --ipc=host --cap-add=SYS_ADMIN` + 三个 `-v` 挂载

| 键 | 含义 | 默认 |
|---|---|---|
| `name` | 实验名（**必填**）——运行目录名、容器名、log 名都由它派生 | — |
| `gpus` | 物理卡号，逗号分隔（**必填**），如 `"4,5,6,7"`；张数即 tp-size | — |
| `model` | `models_dir` 下的模型目录名（**必填**），如 `qwen2.5-72b-instruct-config` | — |
| `tp` | 覆盖 tp-size（EP/特殊场景才用）| = gpus 张数 |
| `server` | 追加到 launch_server 的额外参数串 | 空 |
| `bench` | 追加到 bench_serving 的额外参数串 | `--num-prompts 1000 --random-input-len 1024 --random-output-len 512 --request-rate inf` |
| `note` | 说明，写进 run 的 README + runs 索引 | 空 |
| `fields` | dcgm 字段组（`tool/dcgmi/<name>.txt`）| `pass1_core` |
| `interval_ms` | 采样间隔（下限 100）| `100` |
| `server_timeout` | server 多久没 ready 判失败（秒）| `600` |
| `exclusive_process` | 计算模式：`true`=可以独占→`nvidia-smi -c 3`(EXCLUSIVE_PROCESS)；`false`=不能独占(需多进程共用一张卡的负载)→`nvidia-smi -c 0`(DEFAULT) | `true` |
| `model_tag` / `label` | 运行目录名里的短模型名 / 尾部自定义描述（如 `dramhigh`）| 推导 / 空 |
| `image` / `models_dir` / `sglang_repo` / `port` | 环境路径 | 见 `parse_yaml.py` 的 `DEFAULTS` |
| `extra_docker` | 额外 docker run 参数串（少用）| 空 |
| `profiler` | `dcgm` = 带宽曲线 ｜ `nsys` = kernel 时间线（见下节）| `dcgm` |
| `nsys_group` | 仅 nsys：metric 组 → `tool/nsys/groups/<name>.conf` | `cuda` |
| `nsys_windows` | 仅 nsys：采集窗口表（语法见下节）| 见 `parse_yaml.py` |
| `nsys_lead` | 仅 nsys：提前多少秒发 `gen`（补 `gen` 自身的前置开销）| `7` |
| `nsys_freq` | 仅 nsys：GPU Metrics 采样频率(Hz)，覆盖组文件；空=用组文件的 | 空 |
| `nsys_gpus` | 仅 nsys：只在这几张卡上采 GPU Metrics（空=同 `gpus`）。kernel 甘特图不受影响 | 空 |
| `nsys_host_dir` | 仅 nsys：挂进容器的宿主机 nsight-systems 目录 | `/usr/local/cuda-13.1/nsight-systems-2025.5.2` |

新增配置 = 往 `experiments/` 丢一个 `.yaml`，不用改脚本。想加**新的键**要同时改三处，见 [CLAUDE.md §2](CLAUDE.md)。

---

## 两种 profiler：`dcgm` 看带宽曲线，`nsys` 看 kernel 时间线

|  | `profiler: dcgm`（默认）| `profiler: nsys` |
|---|---|---|
| 采集器在哪 | 宿主机 `dcgmi` 旁路采样 | **容器内** Nsight Systems |
| 分辨率 | 10 Hz（NVML GPM 地板）| **10 kHz** GPU 硬件轨 + 逐 kernel 事件 |
| 覆盖 | startup + bench **两个全程窗口** | bench 期间**几个 2–3 秒的短窗口** |
| 产出 | `metrics.csv` + `trace.png` | 每窗口一份 `.nsys-rep`（`nsys-ui` 打开）|
| 回答什么 | 「HBM/NVLink 随时间怎么走、哪段是瓶颈」| 「这一刻具体是哪些 kernel、各多久、气泡在哪」|

### nsys 的窗口表语法

`nsys_windows` = 逗号分隔的 `<标签>@[bench|burst]<±偏移>:<窗口秒数>`，不写锚点前缀 = `burst`。

| 锚点 | 是什么 | 什么时候用 |
|---|---|---|
| `burst` | server 日志里第一条真正的压测 `Prefill batch`（`#new-seq ≥ 4`）| decode 各段 |
| `bench` | bench 进程启动时刻 | **prefill 墙必须用它** |

```yaml
nsys_windows: "prefill@bench+0:8,decode-hi@burst+20:2,decode-mid@burst+45:2,decode-lo@burst+70:2"
```

排窗口的三条硬约束（原因见 [CLAUDE.md §5.2](CLAUDE.md)）：

- **prefill 只能挂 `bench` 锚点**，配 `nsys_lead` 把窗口顶到 burst 附近；
- **窗口之间至少留 ~30s**（写报告 10–22s 且全程阻塞，排太密会被顺次推后）；
- **实际落点以 `windows.tsv` / run README 里那张对齐表为准**，不是 yaml 里写的偏移量。

怎么定窗口：先用 `profiler: dcgm` 跑一遍同配置，从那个 run 的 `server.log` 按
`#running-req` / `cuda graph:` 划出阶段（方法见 `analysis/qwen3-30b-a3b-tp2ep2-dram-phases.md`），
再把窗口摆到各阶段中段。**别采末期 decode**（batch 掉到个位数，带宽不稳、不代表稳态）。
现成例子：`experiments/2gpu.nsys-timeline.yaml`。

⚠️ **跑完一定看那行诊断输出**：`run.sh` 结束前会调 `nsys-tool diagnose` 把**报告内部**的 Error 捞出来
（nsys 只写进报告，stdout 一个字不打、`rc` 仍是 0）。最要命的是
`GPU Metrics [N]: Sampling buffer overflow` —— **那张卡的样本会整段错位到窗口之外、数据作废**。
高频组（`nsys_group: cuda_dram`，组文件里是 200 kHz）本机双卡实测会溢出、100 kHz 干净
→ **双卡就写 `nsys_freq: 100000` 压下来**（或用 `nsys_gpus` 只采一张卡）。

---

## 一个实验产出什么

运行目录统一命名（字段固定顺序，全部自动推导）：

```
<phase>_<ngpu>gpu_<model_tag>_tp<N>_ep<M>_in<X>_out<Y>_<cudagraph>[_<label>]__<时间戳>
```

### `profiler: dcgm` —— 两个 run 目录

- `runs/server_4gpu_qwen2.5-72b_tp4_ep0_in1024_out512_graph__<ts>/` —— **server 启动窗口**（权重初始化 + 抓 CUDA graph 的 HBM 行为）
- `runs/bench_2gpu_qwen3-30b-a3b_tp2_ep2_in1024_out512_graph__<ts>/` —— **bench 服务窗口**（负载下的接口利用率）

每个目录里：`dcgm_raw.log` · `marks.txt` · `metrics.csv`(parse 产出) · `trace.png`(容器画图) ·
`run_meta.json` · `README.md`（`profile.sh` 的 meta+note，脚本再**追加一段可复制的复现命令**）。
外加 sglang 自己的日志：server 目录有 `server.log`（启动段快照）；bench 目录有 `server.log`
（完整运行期，含 bench 窗口的 `gen throughput`）+ `bench.log`（bench_serving 客户端输出）。

### `profiler: nsys` —— 一个 run 目录，窗口是子目录

```
runs/nsys_<ngpu>gpu_<model>_tp<N>_ep<M>_in<X>_out<Y>_<cudagraph>[_<label>]__<ts>/
├── README.md        # 配置 + 诊断判定 + 「窗口真正采到了哪一段」对齐表 + 怎么看
├── diagnose.txt     # nsys-tool diagnose 的逐份判定
├── windows.tsv      # bench_start / burst_anchor / 每个窗口的实际 epoch
├── server.log       # server 完整运行期日志
├── bench.log        # bench_serving 客户端输出
└── windows/
    ├── nsys_cuda_01_prefill__<ts>/report.nsys-rep
    ├── nsys_cuda_02_decode-hi__<ts>/report.nsys-rep
    └── ...
```

> 两种模式都会自动追加一行到 `runs/README.md` 总索引（run → backend → note）。

---

## 失败怎么办（不卡死）

- **server 没起来**（`server_timeout` 到点，或日志出现 `CUDA error: out of memory` /
  `Fail when using backend` / `Traceback`）：该目录标 `_FAIL`，**跳过 bench**，打印 server.log 末尾，
  直接跑下一个实验。
- **bench 非零退出**：bench 目录标 `_FAIL`，输出仍在 `workload.log` / `bench.log`。
- **卡一直不空**：重试 `GPU_FREE_RETRIES` 次（默认 3）仍忙 → 跳过该实验。
- **Ctrl-C / 脚本退出**：`trap` 先把独占复位成 Default（`nvidia-smi -c 0`）再 `docker rm -f` 当前容器，
  不留僵尸、不留独占态（`--interactive` 除外，那个容器归你）。

就绪/失败的判定在 `wait_ready.sh`。误判（启动期的非致命 Traceback 被当成崩了）就去那里调 `FATAL_RE`；
启动特别慢的大模型把 `server_timeout` 调大。

---

## 分析笔记（`analysis/`）

跑出来的曲线怎么读、瓶颈在哪、有哪些调优旋钮，沉淀在 `analysis/`：

| 笔记 | 讲什么 |
|---|---|
| [`qwen3-30b-a3b-tp2ep2-dram-phases.md`](analysis/qwen3-30b-a3b-tp2ep2-dram-phases.md) | `trace.png` 的 `dram_active` **为什么分段**：prefill 墙 → 大 batch decode 走 eager → 小 batch 进 CUDA graph；含显存账 + 两条"让 decode 多进 graph"的调优路径；文末给了把这套对齐分析套到别的 run 的通用方法 |
| [`qwen3-30b-a3b-deepep-bf16.md`](analysis/qwen3-30b-a3b-deepep-bf16.md) | 未量化 BF16 下 DeepEP 怎么跑通（附走通/走不通的参数组合）+ 性能对照 |
| [`llama405b-dram-active-explained.md`](analysis/llama405b-dram-active-explained.md) | 405B TP8 的 `dram_active`：为什么并发降低反而升高，以及 ~0.64 的"地板"从哪来 |
| [`deepseek-v4-8gpu-configs.md`](analysis/deepseek-v4-8gpu-configs.md) · [`…-explained.md`](analysis/deepseek-v4-8gpu-explained.md) | DeepSeek-V4 Flash 8×H100 实测跑通的配置（可直接抄）+ 逐个技术名词详解 |
