# experiments/dcgmi_overhead —— dcgmi 采集对被测程序的性能影响（可复现，留档）

量化"旁路开着 `dcgmi dmon` 采集 metric"到底会不会拖慢被测程序，以及**采样频率/字段集**如何影响开销。
这是决定本工具默认采样频率、默认字段集的直接依据。

本机：`zkrh-58`，8× H100 80GB HBM3，NVLink-switched（12×NVSwitch），DCGM 4.2.3，driver 590.48.01。
镜像 `nvcr.io/nvidia/pytorch:25.12-py3`。用空闲卡 **GPU 6（单卡）/ 6,7（通信）**（GPU0 常被 sglang 占）。

## 方法学（重要）

### 为什么用程序内 CUDA event 计时，不用 nsys/nvprof
- 我们要测的是"dcgmi 开/关对 **workload 本身快慢**的影响"。**CUDA event 只读 GPU 时钟、不碰 profiling
  硬件计数器**，所以①和 dcgmi **零冲突**、②不给 workload 引入额外开销 → 测出来的差异干净地归因于 dcgmi。
- **不能用 nsys/nvprof/ncu**：它们本身会扰动 workload，而且 **DCGM profiling 与 Nsight/CUPTI 有官方冲突**
  （A100 及更早严格互斥；Hopper 放宽但仍有争用风险）。用它们会把"dcgmi 的开销"和"profiler 自己的开销/冲突"
  混在一起，污染结论。参考：DCGM Feature Overview（profiling 与 Nsight/CUPTI 冲突）
  <https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html>。
- 关于"nvprof 是否和 dcgmi 冲突"——是的，会冲突（同上），所以本实验**绕开** profiler，用 CUDA event。

### 计时结构
每个 workload 在容器内用 CUDA event **分块计时并持续 ~18s**（comm 用固定 chunk 数以保持两 rank 同步）：
分块 = 计时 `CHUNK` 次算子的总时间 / CHUNK = 单次 kernel 时间；持续 ~18s 是为了让计时区**跨越几十~几百个
dcgmi 采样周期**，才能测到稳态干扰（若只跑 0.2s，1000ms 采样期间可能一个采样点都不重叠）。
报告样本分布的 `ms_median / ms_p95 / ms_std` + 派生性能（TFLOPS / GB·s⁻¹）。

### 三个 workload（各压一个子系统）
| workload | 压什么 | 算子 | 派生指标 |
|---|---|---|---|
| `gemm.py` | SM / Tensor Core（compute-bound）| 8192³ bf16 matmul | TFLOPS |
| `mem_bound.py` | HBM 带宽（memory-bound）| `z=a+s·b` 大 1D bf16 融合逐元素（LLM 残差/scale 类）| GB/s（~92% HBM 峰值）|
| `comm_nccl.py` | NVLink（communication）| NCCL 双向 sendrecv（2 卡）| GB/s（单向 busbw）|

> mem 起初用 RMSNorm，但 eager 版 fp32 上转/reduction 是 launch-bound（仅 ~225 GB/s，打不满 HBM），
> 换成融合 add-scale 才逼近 HBM 峰值、最大化对 HBM 计数器采集的敏感度。

### 条件矩阵（每个 workload × 每个条件 × REPEATS 遍）
| 条件 | dcgmi | 间隔 | 字段集 | 意图 |
|---|---|---|---|---|
| `baseline` | 关 | — | — | 基准 |
| `dev@100` | 开 | 100ms | `449,252,250`（device-only）| 最轻：非 profiling 字段的开销 |
| `pass1@1000` | 开 | 1000ms | pass1 混合* | 低频 + 含 profiling |
| `pass1@100` | 开 | 100ms | pass1 混合* | 高频 + 含 profiling（= 拟用默认）|
| `full@100` | 开 | 100ms | 16 个字段** | 最重：逼近 multiplexing |

\* pass1 = `204,1005,1002,1009,1010,449,1011,1012`（mem_copy_util, dram_active, sm_active, pcie tx/rx, nvlink_bw_total, nvlink tx/rx）
\*\* full = `1001-1012 + 449,204,252,250`（全 profiling 管线 + device）

`slowdown% = (median_dcgmi / median_baseline − 1) × 100`，正值 = dcgmi 拖慢了 workload。

## 怎么跑
```bash
cd experiments/dcgmi_overhead
./run_overhead.sh gemm      # 单卡 6
./run_overhead.sh mem       # 单卡 6
./run_overhead.sh comm      # 2 卡 6,7 (NCCL)
python3 summarize.py        # 汇总 results/raw.tsv → 每 workload 的 slowdown 表
```
环境变量：`REPEATS`(默认2) `TARGET_S`(gemm/mem 时长, 默认18) `N_CHUNKS`(comm, 默认360)。
原始逐遍数据落 `results/raw.tsv`；workload/dmon 日志已打包为 `logs.tar.gz`（解压后为 `logs/`）。
为控体积只保留一组代表性 dmon（各 ≤10Hz 条件 rep1 + 一个 1000Hz 样例 `dmon_gemm_dev_1_rep1`）+ 全部 `wl_*`；
其余高频(1ms)/rep2 dmon 移到 `backup/overhead_logs_extra/`（不入库）。

## 结果（本机实测，REPEATS=2，每条件 ~18s）

`slowdown%` 相对本 workload 的 baseline；正=变慢，负=噪声（比 baseline 还快）。

### compute-bound：GEMM（8192³ bf16, ~694 TFLOPS）
| 条件 | ms_median | ms_std | TFLOPS | slowdown% |
|---|---|---|---|---|
| baseline | 1.5844 | 0.0370 | 694.0 | 0.00% |
| dev@100 | 1.5841 | 0.1053 | 694.1 | −0.02% |
| pass1@1000 | 1.5862 | 0.0713 | 693.2 | +0.11% |
| pass1@100 | 1.5820 | 0.0799 | 695.0 | −0.15% |
| full@100 | 1.5800 | 0.0933 | 695.9 | −0.27% |

### memory-bound：add-scale（打满 HBM, ~3106 GB/s）
| 条件 | ms_median | ms_std | GB/s | slowdown% |
|---|---|---|---|---|
| baseline | 1.0370 | 0.0002 | 3106.1 | 0.00% |
| dev@100 | 1.0372 | 0.0002 | 3105.8 | +0.01% |
| pass1@1000 | 1.0371 | 0.0002 | 3106.0 | 0.00% |
| pass1@100 | 1.0372 | 0.0001 | 3105.9 | +0.01% |
| full@100 | 1.0372 | 0.0001 | 3105.9 | +0.01% |

### communication：NCCL 双向 sendrecv over NVLink（~300 GB/s/dir）
| 条件 | ms_median | ms_std | GB/s | slowdown% |
|---|---|---|---|---|
| baseline | 0.8952 | 0.0016 | 299.9 | 0.00% |
| dev@100 | 0.8966 | 0.0014 | 299.4 | +0.16% |
| pass1@1000 | 0.8964 | 0.0014 | 299.4 | +0.14% |
| pass1@100 | 0.8968 | 0.0014 | 299.4 | +0.18% |
| full@100 | 0.8966 | 0.0014 | 299.4 | +0.17% |

### 极限频率：dcgmi `-d 1`（1ms=1000Hz 轮询；profiling 内部仍 100ms，host 端狂刷查询）
用 `run_overhead.sh --extreme`（EXTREME_MS=1）补测。`dev@1`=device-only@1000Hz（真高频）、`full@1`=16 字段@1000Hz（狂刷 host）。

| workload | dev@1 slowdown | full@1 slowdown | 说明 |
|---|---|---|---|
| gemm | +0.09% | +0.02% | 噪声内 |
| mem | +0.01% | 0.00% | 零 |
| comm | −0.06% | −0.07% | 噪声内（负=比 baseline 还快）|

**即便把 dcgmi 拉到 1000Hz + 全字段，三类 workload 的开销仍 <0.1%（落在运行间抖动内）。**

> ⚠️ 但 `-d 1` 的**数据**是无效的：profiling/byte 字段内部只有 ~100ms(10Hz) 才刷新（NVML GPM 地板，非 DCGM），1000Hz 下 ~90% 是空行
> （byte 计数器空读=`0`、ratio 空读=`N/A`）。见 `experiments/dcgmi_pure`——**限制频率的是数据有效性，不是开销。**

## 结论

1. **dcgmi 采集对被测程序的性能开销可忽略**：主矩阵(1Hz/10Hz) + 极限(1000Hz)、字段集从 3 个到 16 个，
   三类 workload（compute/HBM/NVLink）的 slowdown **全部 <0.2%，绝大多数落在运行间噪声内**。
   最"敏感"的是 comm(NVLink)，也仅 ~+0.15%@10Hz，且到 1000Hz 反而回落到噪声——不构成实质开销。
2. **开销不随频率/字段数显著增长**：1Hz→1000Hz、device-only→full 16 字段，均未见单调上升的开销。
   → **本工具默认 100ms、pass1 字段是安全的，甚至可以更激进而不担心拖慢 workload。**
3. **真正限制采样频率的是数据有效性，不是性能**：profiling/byte 字段的有效率被 **NVML GPM 的 ~100ms 内部刷新率**
   钉死（**这是 NVML/硬件地板，不是 DCGM 策略**；见 `docs/metrics_reference.md` §0.4/§3.1），
   >10Hz 只产空行（见 `experiments/dcgmi_pure` 实测：20Hz 有效率 55%、100Hz 仅 13%）。
   → **默认 100ms(10Hz) 同时是"开销安全"和"数据有效"的最优点。**
4. **方法学自证**：workload 的 `ms_std` 极低（mem 低至 0.0001ms），说明测量本身足够灵敏——
   若有 >0.5% 的真实开销一定会被检出；测不到即真没有。CUDA event 计时 + 避开 nsys/nvprof 是对的。
