# ThunderKittens P0 profiling scheduler

输入是 `tk_p0_profiling.yaml`。它由 tracer 的 P0 manifest 复制而来，但每个 `runner`
都指向 ThunderKittens checkout 中未经修改的官方 `benchmark.py`。调度顺序为：

1. 确认 8 张 GPU 均无 compute process（每轮只执行一次全局 `nvidia-smi` 查询，空闲后立即继续）；
2. 运行较短的官方 native benchmark；
3. 启动官方 benchmark 的超长 iterations，并顺序采集 Nsys、DCGM；
4. kill 长任务，后台导出并验证 Nsys/kernel/GPU Metrics/DCGM；
5. 继续下一个 case，并在终端输出总进度和各 case 的 PASS/FAIL。

默认 Nsys 窗口为 `0.2s`，硬件指标频率为 `100000Hz`，只采物理 GPU `0`：

```bash
~/Repos/ThunderKittens/.venv/bin/python3 scripts/profile_tk_p0.py
```

覆盖参数：

```bash
~/Repos/ThunderKittens/.venv/bin/python3 scripts/profile_tk_p0.py \
  --nsys-duration 0.5 \
  --nsys-gpus 1,2,3 \
  --nsys-frequency 50000
```

`--nsys-gpus` 是宿主机物理 GPU 编号，可传一个或逗号分隔的多个编号。高频同时采多卡更容易
overflow；scheduler 不静默接受坏报告，后台 validator 会在终端把对应 case 标成 `FAIL`。

只检查 16 个 case、官方路径、shape 和参数，不查询或使用 GPU：

```bash
~/Repos/ThunderKittens/.venv/bin/python3 scripts/profile_tk_p0.py --dry-run
```

三组 MoE `_C` variant 默认生成到当前仓库的 `scripts/thunderkittens/moe_variants/`，不会修改
ThunderKittens 或 accel-sim checkout。scheduler 在正式运行前自动补齐缺失 variant；若已经由外部
构建，可传 `--skip-moe-build`。

MoE 路由与 accel-sim P0 runner 一致：每个 rank 使用 seed `42 + rank`，rank 0 按官方
`torch.rand -> repeat -> multinomial(replacement=False)` 生成路由并 broadcast。所有 MoE case 的
`top_k` 均固定为 8；scheduler 会在启动前检查官方 benchmark 中的 seed 语句，manifest validator
也会拒绝其他 seed 或 TopK。
