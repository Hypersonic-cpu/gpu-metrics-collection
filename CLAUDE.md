# CLAUDE.md — metric_collection

收集 GPU 接口性能 metric 的工具（HBM / NVLink / PCIe 带宽时间序列，基于 **NVML/DCGM 旁路采样** + 交换机侧 **NSCQ 直读**）。
**本文件 = 工作原则 + 本机事实 + 仓库地图（索引）**。细节都在被索引的文档里，动手前按地图去读对应文档。

## 工作原则（重要）
- **最终依据 = NVIDIA 官方文档 + 本机实测**。网上第三方说法回到这两者二次确认后才写入；**引用联网结论必须给 reference URL**。
- 推荐某字段前，**先确认它在本机存在且可用**（`docs/host_probe/dcgmi_fields_list.txt` / metrics_reference §7）。
- **不要重复推翻已定结论**：动手前先读 `PLAN.md` / `PLAN_DONE.md` 与 `docs/metrics_reference.md`。
- 结论若来自实测，**给出脚本与日志路径**（见「实验约定」）。
- **改文档/README 只陈述当前事实**，不写"旧版/已订正/之前错了"这类修订元注释（记忆 [[docs-state-facts-no-meta]]）。
- **装依赖的分析/绘图一律进容器跑**（宿主机 `pip install` 会 OOM，exit 137）：
  `docker run --rm -v "$PWD":/work -w /work nvcr.io/nvidia/pytorch:25.12-py3 ...`。
  ⚠️ 例外：`dcgmi`/`nv-hostengine`/`nvswitch_traffic` 是**宿主机**二进制（本账号 **passwordless sudo**）；只有"读日志→算/画图/pynvml"才进容器。
  绘图若容器无中文字体：图内英文，中文留控制台/`stats.txt`/README。

## 本机 ground truth（`zkrh-58`，详见 `docs/host_probe/`）
8× H100 80GB HBM3（HBM3 峰值≈3350 GB/s），driver 590.48.01 / CUDA 13.1；**宿主机装 DCGM 4.2.3**；Docker + `nvidia` runtime；MIG Disabled；
**NVLink-switched：4 颗物理 NVSwitch（`dcgmi discovery -l` 报"12"是逻辑口径）+ Fabric Manager active**（单节点）。
- profiling 字段在本宿主机**无需额外 sudo** 即可采；容器内采 profiling 需 `--cap-add SYS_ADMIN`。
- 跑实验/采集用空卡，且采集器按**物理编号/UUID**选卡（应用容器内部会重编号）。

## 仓库地图（想改哪块 → 去读对应文档）
| 目录/文件 | 是什么 | 想深入 → 看这里 |
|---|---|---|
| `docs/metrics_reference.md` | **指标手册（细节查询库）**：NVML/DCGM/GPM 关系、选字段、权限、坑、NVSwitch 第二部分 | 选字段/理解指标/查机制细节时来这查 |
| `docs/host_probe/` | 本机静态 dump（字段目录 / 拓扑 UUID / 一份 idle dmon 示例）| 需要 ground-truth 事实时 |
| `README.md` | 项目入口（两类工具 + 一键脚本 TODO）—— **给用户看** | — |
| `PLAN.md` | **当前/未来待做**（Phase 1 目标、工具架构、待做/待审核项）| 动手实现前 |
| `PLAN_DONE.md` | **已完成/已锁定**（拍板的设计决策、已实现 MVP、实验对工具的结论）| 确认某决策是否已定 |
| `tool/profile.sh`·`metrics.py`·`stamp.py` | **两后端高层入口/共用脚本**（`--backend dcgm\|nvswitch` 各自独立采集；解析·绘图按 backend 自适应 + 打戳；`metrics.py parse`/`plot`）| 用法 → `tool/dcgmi/README.md` / `tool/nvswitch_traffic/README.md`；架构 → `PLAN.md`「工具架构」|
| `tool/dcgmi/` | **① DCGM 工具**：README(快速上手) + 字段组 `<name>.txt`（`profile.sh --fields` 读它）| `tool/dcgmi/README.md` |
| `tool/nvswitch_traffic/` | **② 自研 NSCQ 工具**（C + Makefile + README + 配置组 `<name>.conf`）：过交换机的 NVLink 流量；可 `profile.sh --backend nvswitch` 编排 | `tool/nvswitch_traffic/README.md` |
| `tool/nsys/` | **③ Nsight Systems 工具**：`nsys-tool` 采集 + `check_report.py` 判报告能不能用（PASS/PARTIAL/FAIL/ERROR）+ **后处理四步** `post_export.py`(报告→CSV) → `post_plot.py overview`(全局图) → `dram_analyze.py`(统计/主周期/建议切分，**唯一 adhoc 的一步**) → `post_plot.py detail`(逐点细节图，每卡一格 total/rx/tx)，四步共用 `post_io.py` 契约（文件名+CSV+接口口径）+ metric 组 `groups/<name>.conf` + nsys 原生 set `sets/<name>.config`。GPU 硬件轨 **10 kHz**（比 DCGM 快 1000×）+ CPU 轨 + CUDA kernel 甘特图；不走 `profile.sh`，输出同样落 `runs/<id>/` | **操作** → `tool/nsys/README.md`；**机制/口径/实测澄清** → `tool/nsys/CLAUDE.md`；**某次实验结论** → `tool/nsys/EXP_*.md` |
| `tool/workloads/` | 测试负载替身（`gpu_busy.py`）| — |
| `tool/sglang_bench/` | **SGLang dummy 权重性能实验一键编排**：宿主机一条命令 `run.sh <配置短名>` 起容器→起 server→发 bench、按统一命名归档；server 日志落 `<sglang_repo>/logs/<name>.server.log`。配置清单在 `experiments/<短名>.yaml`（`--list` 看有哪些，`archive/` 放历史备份）。yaml 的 `profiler` 选后端：`dcgm`（默认，编排 `profile.sh`+`metrics.py`，采 startup/bench 两个全程窗口）／`nsys`（编排 `nsys-tool`，server 起在 nsys 下、bench 期间开几个短窗口取 kernel 时间线）| **操作** → `tool/sglang_bench/README.md`；**机制/口径/实测澄清** → `tool/sglang_bench/CLAUDE.md`；**某次实验结论** → `tool/sglang_bench/analysis/` |
| `experiments/cumemcpy/` | memcpy 三类点亮矩阵 + L2 影响（ground-truth）| README |
| `experiments/nvlink_bench/` | **NVLink 卡间带宽上限**（DMA 单/双向 405/802；SM kernel 远程访问 375；含并发度+数值类型扫描）+ **user/protocol 拆分**（nsys 8 路 @200kHz：DMA 有效率 94% ⟷ fp8 远程读 54%）| README；**多播/在网归约单独成篇** → `EXP_multimem.md`（2/4/8 卡：nsys 证到交换机吃进 N× 吐出 1×；`ld_reduce` team 聚合恒 ~605 GB/s 加卡不涨）；**TMA/包结构单独成篇** → `EXP_tma.md`（`cp.async.bulk` 跨卡 2/4/8 卡 × p2p/ring/fanout：包仍 128 B 但只要 512 线程；含「一次 memcpy 搬多大」与访问宽度两个配套实验）|
| `experiments/dcgmi_pure/` | 采样频率地板(10Hz) + idle 底噪(只有 PCIe) | **先读 `SUMMARY.md`（地图）**再 README |
| `experiments/dcgmi_overhead/` | dcgmi 采集开销 <0.3% | README |
| `experiments/nvswitch/` | DCGM switch 字段❌ / NSCQ 直读✅ | README + `BACKGROUND.md`(scoping/依据) |
| `runs/` | 每次采集一个子目录（原始日志 + metrics.csv + 图，数据）| — |

- **想改/加"高层工具"**：每个工具 = `tool/<name>/` 子目录（自带 README，操作型快速上手）；共用脚本留 `tool/` 顶层。规则见 `PLAN.md`「工具架构」+ 记忆 [[tool-folder-convention]]。
- 文档分工（README vs 手册 vs 实验）见记忆 [[doc-structure-convention]]。
- **工具目录内部再分三层**（`tool/nsys/` 是样板）：`README.md` **只讲操作**（怎么调用/参数/产出什么，给用户看）；
  `CLAUDE.md` 讲**机制·口径·实测澄清**（给自己看，动手改代码前先读）；**某次实验的数值与解读**单独开 `EXP_<名字>.md`，
  不塞进 README。

## 实验约定（`experiments/`）
任何实验放 `experiments/<大方向>/`，**代码 + 日志都留档**（可复现 + 作为文档 ground-truth）。每个大方向自带
`README.md`(证明了什么+结果矩阵+日志索引) · `run_*.sh`(跑测器：宿主机 dmon + 容器 workload) · `workloads/` · `logs/`。
结论写进 `docs/metrics_reference.md` 时引用对应脚本/日志路径。跑法（以 cumemcpy 统一字段集为例）：
```bash
F=204,1005,1002,1009,1010,449,1011,1012
experiments/cumemcpy/run_probe.sh <name> <host_gpu_ids> $F <container_devices> <workload.py> [dmon_count]
```
**日志入库方式**：`logs/` 被 `.gitignore` 忽略（`experiments/*/logs/`），入库的是实验根目录下打好的包——
在实验目录里 `tar czf logs.tar.gz logs`（**包内保留 `logs/` 前缀**），本地 `logs/` 留不留都行。
README 的「日志索引」一节开头加一句：`> **原始日志已打包为 `logs.tar.gz`**（解压后即 `logs/` 树，本页引用的 `logs/...` 路径均指压缩包内）。`

## 关键坑速记（实测，勿再踩；详解见 metrics_reference 对应 §）
- **DCGM 数据源 = NVML GPM（H100）**：DCGM 不读硬件，只"调 NVML+缓存+分发"；`dram_active`/`*_bytes` 底层就是 NVML GPM。→ §0
- **采样地板 ~100ms 是 NVML GPM 的、不是 DCGM 的**：`-d` 可填 1ms 但只是重读缓存；>10Hz 只产空行。想 <100ms 只能 Nsight/CUPTI。→ §0.4/§3.1
- **warmup**：byte/profiling 字段头 ~2 拍 N/A；dmon 至少跑 5 拍；采集/parser/validate 要容忍开头 N/A。→ §2
- **不可用字段**：`pcie_*_throughput`(200/201) Deprecated 全程 N/A；用 `pcie_*_bytes`(1009/1010)。→ §7.1
- **`mem_copy_utilization`(204) 不采**：混合所有流量的粗粒度 HBM 利用率、无接口拆分。→ §5.1
- **引擎 vs 接口**：跨边界拷贝走 Copy Engine（`sm_active=0`）；单卡内 D2D 才是 SM kernel。别用 `sm_active` 判"是否在搬数据"，接口带宽用 `dram_active`/`*_bytes`。→ §5.6
- **L2 会挡 HBM 指标**：cache-resident 拷贝让 `dram_active≈0` 却在猛搬；判"有没有搬"认接口 byte 计数器。→ §5.1
- **P2P 坑**：卡间 `cudaMemcpyPeer` 不启 P2P 偷偷走 host 中转(PCIe，~37GB/s、nvlink=0)；务必用 nvlink metric 核对。→ §5.3
- 🔴 **NVLink 的带宽上限 / 百分比分母全仓已清空、尚无定论**：`tool/nsys` 曾用 478 GB/s/方向、`tool/metrics.py`+`tool/dcgmi` 曾用 450/900，**两个数服务两套不同的计数器**。
  **定论之前别往代码/文档里写死任何 NVLink 峰值数**（比值类结论不受影响：有效率、rx vs tx、相位对比照常可用）。
  波及模块清单 + 要定的事 + 遗留产物 → `PLAN.md` §3 开头那块；删掉的原文逐字备份 → `backup/nvlink_bandwidth_ceiling_2026-07-30.md`。
- **NVSwitch 侧 DCGM 字段(780/781/861/862…)本机恒 0**（[issue #236](https://github.com/NVIDIA/DCGM/issues/236)）；过交换机流量用 `tool/nvswitch_traffic`（NSCQ 直读）。→ §5.2 / 第二部分
- **nsys：GPU 硬件轨/CPU 轨能 attach 已在跑的进程，CUDA kernel 甘特图不能**（`--trace` 是 Application scope，必须启动时注入）。
  两把独立的锁：GPU Metrics 恒需 root（`RmProfilingAdminOnly:1`）；CPU 采样看 `kernel.perf_event_paranoid`
  （process-tree 要 ≤2、system-wide 要 ≤0，**本机已从 4 调到 2，未持久化、重启回 4**）；CUDA trace 两把都不管、永远免 sudo。
  → `tool/nsys/README.md`
- **掉卡/down-link 会让 NSCQ 端口计数器返回固定假值**（rc 仍=SUCCESS、rx==tx）：`nvswitch_traffic` 差分后**间歇打印巨大假尖峰**（大到不可能是真流量）；是掉卡假值非真流量，工具暂无防护。→ §11
