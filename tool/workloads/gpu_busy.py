#!/usr/bin/env python3
"""gpu_busy —— 一个自包含的 GPU 测试负载（"被测程序"的占位替身）。

目的：给 dmon 工具链一个能明确"点亮"各接口 metric 的对象，用来把采集流程跑通。
分几个阶段，每个阶段点亮不同 metric，并打 [MARK] 行（带 epoch）标出阶段边界，
方便回头把 trace 和"程序跑到哪了"对齐。

阶段：
  matmul   大矩阵乘        -> SM / Tensor / 一定 HBM
  membound 大向量读写      -> HBM 带宽 (dram_active 拉高)
  h2d      Host->Device 拷 -> PCIe (pcie_*_bytes)
  nvlink   两卡间 p2p 拷   -> NVLink (仅当 --gpus 给了 >=2 张卡)

注意：这里的 --gpus 是"容器内可见的本地编号"（0,1,...）。宿主机 dmon 采集用的是
物理卡号，两者的对应由外层 `docker run --gpus '"device=..."'` 决定（见 README）。
"""
import argparse
import os
import time

import torch


def mark(label: str, marks_file: str) -> None:
    line = f"{time.time():.3f}\t[MARK]\t{label}\n"
    if marks_file:
        with open(marks_file, "a") as f:
            f.write(line)
    # 无论如何都打到 stdout，profile.sh 会把 workload 输出存进 runs/<id>/workload.log
    print(line, end="", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=24, help="总运行秒数（各阶段均分）")
    ap.add_argument("--gpus", default="0", help="容器内本地 GPU 编号，逗号分隔，如 0,1")
    ap.add_argument("--marks-file", default=os.environ.get("MARKS_FILE", ""),
                    help="可选：阶段 MARK 追加到此文件（epoch\\tlabel）")
    args = ap.parse_args()

    gpus = [int(x) for x in args.gpus.split(",") if x != ""]
    dev0 = f"cuda:{gpus[0]}"
    two = len(gpus) >= 2
    dev1 = f"cuda:{gpus[1]}" if two else None
    mf = args.marks_file

    phases = ["matmul", "membound", "h2d"] + (["nvlink"] if two else [])
    per = args.seconds / len(phases)

    N = 8192                       # 8192x8192 fp32 matmul
    a = torch.randn(N, N, device=dev0)
    b = torch.randn(N, N, device=dev0)
    big = torch.empty(512 * 1024 * 1024, dtype=torch.float32, device=dev0)  # 2GB
    host = torch.randn(256 * 1024 * 1024, dtype=torch.float32, pin_memory=True)  # 1GB pinned
    torch.cuda.synchronize(dev0)

    mark("workload_begin", mf)
    for ph in phases:
        mark(f"phase_{ph}_begin", mf)
        end = time.time() + per
        if ph == "matmul":
            while time.time() < end:
                c = a @ b
                a = c * 1e-4 + b        # 保持数据流动，避免被优化掉
            torch.cuda.synchronize(dev0)
        elif ph == "membound":
            while time.time() < end:
                big = big * 1.0000001 + 1.0   # 纯读写，压 HBM 带宽
            torch.cuda.synchronize(dev0)
        elif ph == "h2d":
            while time.time() < end:
                _ = host.to(dev0, non_blocking=True)  # Host->Device 走 PCIe
                torch.cuda.synchronize(dev0)          # 每次同步，避免异步 backlog 撑爆阶段时长
        elif ph == "nvlink":
            x = big
            while time.time() < end:
                y = x.to(dev1, non_blocking=True)     # 卡间 p2p 走 NVLink
                x = y.to(dev0, non_blocking=True)
                torch.cuda.synchronize(dev0)
    mark("workload_end", mf)


if __name__ == "__main__":
    main()
