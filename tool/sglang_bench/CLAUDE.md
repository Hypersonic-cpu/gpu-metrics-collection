# CLAUDE.md — tool/sglang_bench（给自己看：机制 / 口径 / 实测澄清）

**分工**：怎么调用 → `README.md`（给用户）；某次实验的数值和解读 → `analysis/`；
**本文件 = 为什么这么编排、哪些口径是实测钉死的、改代码前要知道的约束**。

> `analysis/` 目前是这个工具的实验记录层（按 CLAUDE.md 规则本该叫 `EXP_<名字>.md`），暂未收敛。

---

## 一、为什么是「宿主机脚本驱动 docker」

采集器和被测程序不在同一侧：**dcgmi / nsys-tool 在宿主机，server 和 bench 在容器**。
所以 `run.sh` 必须在宿主机跑，用 docker 去驱动容器，而不是反过来。

| 环节 | 做法 | 为什么 |
|---|---|---|
| 容器 | `docker run -d … sleep infinity` 起具名常驻容器 | 一个实验多次 `docker exec`（起 server、发 bench）都要打进同一个容器 |
| server | `docker exec -d` 后台起，输出重定向到挂载出来的日志 | 采集在宿主机，得让宿主机能读到 server 状态 |
| 就绪判定 | 宿主机轮询 `<sglang_repo>/logs/<name>.server.log` | 仓库已 `-v` 挂进容器，读宿主机文件即可，不用 `docker exec` 反复探 |
| 窗口 | `profile.sh` **wrap 模式**分别包住「轮询就绪」和「跑 bench」 | 被包的命令一返回就停采 —— 比 `--duration 600` 这种固定窗口准 |

⚠️ 起 server 前必须 `: > "$server_log_host"` **清空旧日志**，否则 `wait_ready.sh` 会读到上一次的
ready 行，秒判"就绪"，startup 窗口直接空掉。

## 二、`parse_yaml.py` ↔ `run.sh` 的契约（改配置键必读）

一行一个实验，列用 **`\x1f`(US) 分隔**：必须是**非空白**分隔符，否则空字段会被 bash 的 IFS 折叠、
整行错位。列序 = `parse_yaml.COLS`，run.sh 侧那句 `IFS="$SEP" read -r name gpus tp …` 按同一顺序接。

**加一个配置键要同时改三处，且新键只能加在末尾**：

1. `parse_yaml.py` 的 `DEFAULTS`（给默认值）+ `COLS`（追加到列尾）
2. `run.sh` 顶层 `IFS="$SEP" read -r …` 的变量表（追加同名变量）
3. `run.sh` 的 `run_one()` 位置参数（`local name="$1" … nsys_gpus="${25}"` 那一串，新键接在后面）
   —— 以及底部 `run_one "$name" … ` 那行实参

漏一处 = 所有实验的参数整体错位，且**不会报错**（都是字符串）。

取值三层覆盖：`parse_yaml.DEFAULTS` ← yaml 的 `defaults:` ← 实验自身的键。
`tp` 不填 = `gpus` 张数；布尔经 `truthy()` 归一成 `1/0`（bash 侧只认 1/0）。

## 三、运行目录名（stem）：字段恒在才能横比

```
<phase>_<ngpu>gpu_<model_tag>_tp<N>_ep<M>_in<X>_out<Y>_<cudagraph>[_<label>]__<ts>
```

设计点：**每个字段永远出现**，哪怕没启用 —— 没写 `--ep-size` 也写 `ep0`，没关 CUDA graph 也写 `graph`。
这样目录名能直接按字段排序/横比，不会因为某个实验少写一个 flag 就整体错位。

⚠️ 这些字段是 `run_one()` 用 **grep 从命令行字符串里抠的**，不是问 server 要的：

| 字段 | 来源 |
|---|---|
| `ngpu` | `gpus` 逗号切分的个数 |
| `ep` | server 里 `--ep-size N`；没有 → 0 |
| `in/out` | bench 里 `--random-input-len` / `--random-output-len`；缺任一 → 整段省略 |
| `cudagraph` | `--disable-cuda-graph`→`eager`；`--cuda-graph-backend-prefill tc_piecewise`→`tcpiece`；`… disabled`→`nopfgraph`；否则 `graph` |
| `model_tag` | yaml 的 `model_tag`，不填则 `model` 目录名去掉 `-config`/`-instruct` |

→ 换一种等价写法（比如序列长度改从别的 flag 给）目录名就退化成默认值。**新增 flag 想进目录名，得来这里加 grep。**

改名链路：`profile.sh` 先按裸时间戳 `20260722-202132` 建目录 → `latest_raw_run()` 按 mtime 找形如
`^\d{8}-\d{6}$` 的目录 → `process_run()` parse 完再 `mv` 成 stem 名 → 顺手 `sed` 修
`runs/README.md` 索引里那行的链接（不然索引指向已经不存在的时间戳目录）。

## 四、dcgm 模式的两个窗口

| 窗口 | wrap 的是什么 | 什么时候停 |
|---|---|---|
| startup | `bash wait_ready.sh <log> <timeout>`（`--lead 1 --lag 1`）| server 打出 ready 行的那一刻 |
| bench | `docker exec … bench_serving` | bench 进程退出 |

`wait_ready.sh` 在 `profile.sh` 眼里就是"被测命令"，它的返回码进 `run_meta.json` 的 `workload_rc`，
run.sh 据此决定要不要跳过 bench：**0=ready / 1=命中致命签名 / 2=超时**。

⚠️ `FATAL_RE` 里含 `Traceback` —— 启动期任何非致命 Traceback 都会被判成崩了。误判就去
`wait_ready.sh` 改 `FATAL_RE`，别去改 run.sh。

**日志收集口径**（bench 目录里三份日志不是重复）：

- `server.log` = server **完整运行期**（bench 结束那一刻的宿主机日志快照，含 bench 期的 `gen throughput`）
- `workload.log` = `profile.sh` 落的被测命令输出，**`metrics.py` 依赖这个名字，不能改**
- `bench.log` = `workload.log` 的副本，只是名字直观 —— 拷贝而不是改名就是为了不动上面那条依赖

## 五、nsys 模式（`run_nsys_phase`）

### 5.1 和 dcgm 模式的三点差别

1. **server 必须起在 nsys 下**（`nsys-tool launch`）：CUDA kernel 甘特图是 Application scope，
   对已经在跑的进程事后 attach 拿不到。→ `tool/nsys/CLAUDE.md` §1.1
2. **不采 startup 窗口**：启动期时间线不是这个模式要的，全程 trace 报告也会大到打不开。
3. **bench 放后台**，前台按窗口表调度短窗口，最后 `shutdown` 连 server 一起收。

### 5.2 两个锚点：为什么 prefill 只能挂 `bench` ★

| 锚点 | 定义 | 探测方式 |
|---|---|---|
| `burst` | server.log 里**第一条真正的压测 `Prefill batch`**（`#new-seq ≥ 4`）| `wait_burst.sh` 轮询日志 |
| `bench` | bench 进程启动时刻 | `date` 直接记 |

`bench_serving` 会先发一条 warmup 请求、等 tokenizer/连接就绪才猛发，**这段前戏时长会漂（实测
~8–10s，跟模型/镜像有关）**。decode 各段锚在 `burst` 上位置才稳。

但 prefill 墙只有 ~5s，而 `burst` 得等日志行出现才探得到，再叠上 `gen` 自身
**5–9s 的前置开销**（`docker exec` + `nsys start`）—— **实测：请求 burst+1s，实际 burst+10s 才开窗，
四个窗口全落进 decode**。所以 prefill 挂 `bench+0`，靠 `nsys_lead`（默认 7s，提前这么多发命令）
把窗口顶到 burst 附近。

正因如此，**burst 探测跑在后台**（`wait_burst.sh &`）：bench 锚点的窗口必须在探到 burst 之前就发出去，
等不了探测结果。第一个 burst 锚点的窗口才 `wait` 那个后台进程。探测超时 → 退回 bench 启动时刻当锚点。

⚠️ **窗口之间至少留 ~30s**：`stop` 写报告要 10–22s 且 `gen` 全程阻塞，排太密后面的窗口被顺次推后。

### 5.3 `windows.tsv` 记的是实际落点，不是请求偏移

每个窗口记的是 nsys 自己打的 `window_start`（从该窗口 `marks.txt` 里读），不是 `gen` 返回的时刻、
更不是 yaml 里写的偏移量。`align_windows.py` 拿它对 `server.log` 的 `Prefill/Decode batch` 行，
算出每个窗口**真正**盖住了哪一段（批次种类 / `#running-req` 区间 / cuda graph 与否），结果贴进 run README。
**判断某窗口采到 prefill 还是 decode，只能看这张表。**

### 5.4 采集参数的实测依据

- **`nsys_freq`**：sglang 负载下**双卡 200 kHz 会间歇溢出**（溢出那路样本整段错位、作废），
  单卡 200 kHz / 双卡 100 kHz 干净。统计与结论见 `tool/nsys/CLAUDE.md` §1.7 —— 结论是**用 100 kHz**。
- **`nsys_gpus`**：只在一张卡上采 GPU Metrics 能把缓冲压力减半；TP 对称负载下一张卡足够代表。
  kernel 甘特图是 Application scope，**不受 `-i` 影响、始终覆盖全部卡**。
- **`nsys_host_dir`**：容器自带 nsys 2026.3.1 产出的报告，宿主机 GUI 2025.5.2 打不开（GUI 版本必须
  ≥ 产出版本）→ 把宿主机 nsight-systems 挂进容器 + `NSYS_BIN` 指过去。
- **`exclusive_process: false`**（nsys 的几份 yaml 都这么写）：**理由与 nsys 无关** —— 连续跑多个实验时
  上一个容器的 CUDA context 没释放干净，下一个 server `set_device` 会撞 `cudaErrorDevicesUnavailable`。
  （GPU Metrics 采样器不占 CUDA context，和 `EXCLUSIVE_PROCESS` 本身不冲突，实测见 `tool/nsys/CLAUDE.md` §1.6。）

### 5.5 diagnose 为什么放在最后统一跑

nsys 把采集期 Error **只写进报告内部**，stdout 一个字不打、`rc` 仍是 0 → 不查就会拿废数据分析。
但查一次要 export sqlite（~1s/份），**塞进窗口调度路径会把后面窗口的起点推后** —— 所以放在
`shutdown` 之后一次性跑完所有窗口，退出码存进 `diagnose.txt` 并贴进 run README：
**0=全 PASS / 1=有 PARTIAL / 2=有 FAIL 或 ERROR**。

### 5.6 两处实现细节（别顺手删）

- `docker exec … > "$parent/bench.log"` 之前有一句 `mkdir -p "$wdir"` 兜底：server 启动那几分钟里
  这个目录被清掉过，重定向 ENOENT 会**直接吞掉 bench**、还不报错。
- `shutdown` 显式给 `--in-container`：平时它从状态文件继承容器名，状态文件没了就会跑去宿主机找 nsys。

## 六、卡空闲检测与计算模式

- **主判据 = 有没有 compute 进程**（`--query-compute-apps`）：宿主机 `nvidia-smi` **能看到容器内的 GPU 进程**
  （实测确认），所以在宿主机看就够，不用进容器。显存阈值只作兜底（进程已退但显存没释放/泄漏）。
- 忙 → 等 `GPU_FREE_WAIT` 秒重试，`GPU_FREE_RETRIES` 次仍忙就**跳过这个实验继续下一个**（不卡死）。
- 计算模式在**容器内**设（root + `--cap-add=SYS_ADMIN`）：容器只见被分配的那几张卡（重编号 0..n），
  所以 `nvidia-smi -c` 不需要 `-i`。跑完 / `trap` 退出前一律 `-c 0` 复位，不把卡留在独占态。

## 七、坑速记

- **别在 sglang 容器里跑 `run.sh`**：它要宿主机的 `dcgmi` + 宿主机的 `docker`。
- **加配置键要同时改三处**（§2），漏了不报错、只是参数静默错位。
- **目录名字段靠 grep 命令行**（§3），换写法就退化成默认值。
- **别对同一张卡同时跑 nsys 和 `profile.sh --backend dcgm`**：抢硬件计数器。
- `--interactive` 会把 `CUR_CONTAINER` 置空交给用户，**`trap` 不会帮你删容器**，也不会复位独占。
- nsys 模式里 bench 提前结束 → 剩下的窗口直接 `break` 不采（`kill -0 $bench_pid` 探的），
  所以窗口表排得比 bench 长不会报错，只是后面几个静默没了。
- 出图容器默认 root → 产出属主是 root、宿主机改不动。`process_run()` 里的 `docker run` 已带
  `--user $(id -u):$(id -g)` + `MPLCONFIGDIR=/tmp/mpl`，自己另起容器画图时别漏。

---

## 八、`experiments/` 待整理（🚧 已审计未执行，下次单独开 session）

2026-07-29 审计过一遍：**没有孤儿脚本**（4 个脚本全被 `run.sh` 调用），**18 份 yaml 全部真的跑过**
（每份在 `runs/` 里都能找到对应的 `server_*` / `bench_*` / `nsys_*` 目录）。
下面是当时定位到的冗余与碎片，**用户决定推迟处理，文件一个没动** —— 动手前先复核一遍下面的证据还成不成立。

### 8.1 内容重复（旧版本 / 纯备份，非独立实验）

| 文件 | 证据（当时实测） | 建议 |
|---|---|---|
| `experiments/archive/qwen-baseline.260723.yaml` | 与 `default.yaml` 去掉注释后 `diff` **只差一行**（`default.yaml` 多 `exclusive_process: true`）| 删；`archive/` 随之空掉也一并去掉 —— 文件级备份交给 git |
| `experiments/qwen.260722.yaml` | `default.yaml` 的前身。`diff` 去注释后**只有 name / model_tag / label 行不同**，gpus/model/server/bench 逐字相同（旧命名 `4gpu_qwen2.5-72b_tp4` → 新命名 `72b_baseline` + `model_tag`）| 删；那 4 个已跑的 run 目录仍在 `runs/`，复现改用 `default.yaml` |

### 8.2 一条调查线碎成多份（都是真实验，合并时 **name/label 必须原样保留**，否则对不上已有 run 目录）

| 现状 | 内容 | 建议 |
|---|---|---|
| `2gpu.nsys-dram200k` / `-dram200k-1gpu` / `-dram100k` / `2gpu.nsys-200k-ab` | 4 份 / 5 个实验，彼此只差 `nsys_freq`+`nsys_gpus`+`gpus`；全是「200 kHz 采样缓冲溢出」这一条调查线（结论已落 `tool/nsys/CLAUDE.md` §1.7）| 合成 `2gpu.nsys-dramfreq.yaml`（5 个实验，用 `--only` 挑着跑）|
| `2gpu.cudagraph`（out512×5）+ `2gpu.q30b-seqlen-deepep`（out2048×5 + out4096×5 + deepep×1）| 2 份 / 16 个实验，同一组旋钮（`--cuda-graph-max-bs-decode` / `--max-concurrency`）不同 out 长度，**无重名、互补** | 可合成 `2gpu.q30b-cudagraph-sweep.yaml`；注意两边 name 风格不同（`qwen3_30b_a3b_tp2_ep2_*` vs `q30b_ep2_out2048_*`），合并后一个文件里会有两套 |

### 8.3 `default.yaml` 的名字

隐式默认已经去掉（`run.sh` 不给配置就列清单退出，不会误跑那份含 4 卡 235B 的清单），
所以这份文件现在只是「07-22 那批 qwen baseline」，**`default` 这个名字已经名不副实** → 整理时顺手改名
（如 `qwen.baseline.yaml`）。改名要同步：本文件、`README.md` 里的示例、`analysis/` 里的引用。

### 8.4 已执行的部分

`plot_dramsweep.py`（数值硬编码的一次性画图脚本，`run.sh` 不调用）已移到仓库根 `backup/`（gitignore 目录，
不进版本库）。它产出的 `analysis/llama405b-dram-sweep.png` 和引用它的笔记都留在原处。
