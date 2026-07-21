# metric_collection —— GPU 接口 metric 旁路采集工具

旁路采样任意 GPU 程序运行期间的**接口 metric 时间序列**（带墙钟时间戳、可与"程序跑到哪"对齐）：
- **GPU 侧**：HBM / SM / PCIe / GPU 侧 NVLink / 显存占用 —— 用 DCGM。
- **交换机侧**：过 NVSwitch 的 NVLink 流量（每台 switch·每端口 rx/tx）—— 用自研 NSCQ 工具。

> 字段依据/指标语义/机制细节 → `docs/metrics_reference.md`（手册，细节查这里）。
> 设计与待办 → `PLAN.md`（当前/未来）、`PLAN_DONE.md`（已完成/已锁定）。全库导航看 `CLAUDE.md` 仓库地图。

---

## 两类工具（都在 `tool/`）

| 工具 | 采什么 | 怎么采 | 工具文档 |
|---|---|---|---|
| **① DCGM 采集**（主力）| HBM / SM / PCIe / GPU 侧 NVLink / 显存 | `dcgmi dmon` 旁路 | [`tool/dcgmi/README.md`](tool/dcgmi/README.md) |
| **② nvswitch_traffic**（补 DCGM 空缺）| 过 NVSwitch 交换机的 NVLink 流量 | 直连 `libnvidia-nscq` | [`tool/nvswitch_traffic/README.md`](tool/nvswitch_traffic/README.md) |

**一句话选路**：测 GPU 侧带宽 → 用 ①；测过交换机的流量 / 每 switch 端口热点 → 用 ②。两者可同时跑、共用同一时间窗。

`tool/` 目录结构（详见 `PLAN.md`「工具架构」）：
```
tool/
├── profile.sh  metrics.py  stamp.py          # 高层/共用脚本（DCGM 路径入口 + 解析·绘图 + 打戳）
├── workloads/                                 # 测试负载（gpu_busy.py 等"被测程序"替身）
├── dcgmi/                                     # ① DCGM 工具：README(快速上手) + 字段组 <name>.txt
└── nvswitch_traffic/                          # ② NSCQ 工具：C 源码 + Makefile + README
```

---

## 快速上手

前置（本机 zkrh-58 已满足）：宿主机装了 DCGM（`dcgmi` 在 `/usr/bin`）、Docker + `nvidia` runtime 就绪；
采集器**在宿主机**跑、`-i` 用**物理卡号**；被测程序在它自己的容器里跑。选一张空卡：
```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

当前两类工具**分别启动**（合并成一键脚本的计划见下 TODO）。各自的完整用法见对应 tool README，最小跑通如下：

### ① DCGM 采集（GPU 侧 metric）—— 入口 `tool/profile.sh`
`profile.sh` 编排整条流水线：起 `dcgmi dmon` + 打墙钟戳 + 记 marks，wrap/attach 两模式，输出统一进 `runs/<id>/`。
```bash
# wrap：工具拉起负载 → 自动起停采集（GPU 4,5；1000ms 低频 smoke）
tool/profile.sh --gpus 4,5 --fields pass1_core --interval-ms 1000 -- \
  docker run --rm --gpus '"device=4,5"' -v "$PWD":/work -w /work \
    nvcr.io/nvidia/pytorch:25.12-py3 python tool/workloads/gpu_busy.py --gpus 0,1 --seconds 20

# attach：被测程序在别处已在跑，只采一个时间窗
tool/profile.sh --gpus 4,5 --fields pass1_core --interval-ms 1000 --duration 60

# 解析 + 出图（RUN=$(ls -dt runs/*/ | head -1) 取最新）
python3 tool/metrics.py parse runs/<run_id>  # → metrics.csv + 控制台速览（宿主机, 纯 stdlib）
docker run --rm -v "$PWD":/work -w /work nvcr.io/nvidia/pytorch:25.12-py3 \
  python tool/metrics.py plot runs/<run_id>  # → plots/trace.png（容器）
```

`profile.sh` 参数：

| 参数 | 含义 | 默认 / 说明 |
|---|---|---|
| `--gpus` | 采哪几张卡（**物理卡号**）| 必填，如 `4,5` |
| `--fields` | 字段组名（读 `tool/dcgmi/<name>.txt`）| `pass1_core` |
| `--interval-ms` | 采样间隔 ms | `1000`；**下限 100**（先 1000 smoke 再 100 正式）|
| `--duration N` | attach 采 N 秒窗口 | 与 `-- <命令>`（wrap）二选一 |
| `--lead N` / `--lag N` | wrap：命令前空采基线 / 命令后续采等流量回落 | `2` / `3` 秒（`--lag 0` 关）|
| `--out DIR` | 输出根目录 | `runs/` |

输出 `runs/<id>/`：`dcgm_raw.log`(dmon 原样+每行 epoch) · `marks.txt`(关键时刻 epoch) · `workload.log` ·
`metrics.csv`(长表 `epoch,t_rel,gpu,short,metric,value`) · `plots/trace.png` · `run_meta.json`。

> `dcgmi dmon` 命令本身怎么用（`-e/-i/-d/-c` 参数、字段组文件、输出短名、warmup）→ **[`tool/dcgmi/README.md`](tool/dcgmi/README.md)**；
> 字段语义/选字段/权限 → `docs/metrics_reference.md`。

### ② nvswitch_traffic（过交换机的 NVLink 流量）
```bash
cd tool/nvswitch_traffic && make             # 一次编译，产出 ./nvswitch_traffic
./nvswitch_traffic                           # 每 1s 打印每台 switch 的 rx/tx/total GB/s
./nvswitch_traffic -d 500 --csv > sw.csv & P=$!; run_workload; kill $P   # 脚本采一段窗口
```
完整参数、CSV 采集、口径说明 → **[`tool/nvswitch_traffic/README.md`](tool/nvswitch_traffic/README.md)**。

---

## TODO：一键启动脚本（待实现）

目标：一条命令**同时**起两类采集、共用同一 `runs/<id>/` 与时间窗（epoch + marks 对齐），跑完统一解析出图。
设计意图（详见 `PLAN.md`「工具架构」§1.2）：给 `profile.sh` 加后端分派，例如
```bash
tool/profile.sh --backend dcgm,nvswitch --gpus 4,5 --interval-ms 100 -- <被测命令>
# dcgm 起 dcgmi dmon → dcgm_raw.log；nvswitch 起 nvswitch_traffic → nvswitch.csv；同一 marks 对齐
```
**当前**：`nvswitch_traffic` 已可独立/脚本调用，先手动并用即可；等后端分派做好，再把上面「①②」两段 quick start **合并成一段**。

---

## 已知坑（详见 `docs/metrics_reference.md` §7 / `CLAUDE.md`）
- **warmup**：dmon 头 ~2 拍是 `N/A`，正常；采集至少跑 5 拍再信数据。
- **采样地板 ~100ms**：profiling/byte 字段有效采样上限 10Hz（NVML GPM 内部刷新率，非 DCGM 策略）；设更快只产空行。
- **PCIe 自采底噪**：`pcie_*_bytes` 有 ~17 MB/s 空载底噪（DCGM 轮询本身走 PCIe），分析 PCIe 时扣掉；其它接口 idle 即 0。
- **P2P 坑**：卡间 `cudaMemcpyPeer` 不启 P2P 会偷偷走 host 中转(PCIe)；分析多卡通信务必用 nvlink metric 核对。
