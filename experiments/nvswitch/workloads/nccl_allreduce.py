"""NCCL 集合通信 workload：多卡 all_reduce，触发 NVLS(NVLink SHARP)/multimem 走 NVSwitch 计算引擎。

与 nvlink_p2p_saturate.py（单卡→单卡 cudaMemcpyPeer，纯 crossbar 单播 DMA）对照：
all_reduce 在 NVSwitch 系统上会被 NCCL 选 **NVLS 算法**，用 **multimem.ld_reduce/st/red** PTX 指令，
让 **NVSwitch 内的 multicast/SHARP 引擎**做 reduce——这是"真的在交换机里算/复制"的流量，
用来验证 DCGM 的 switch 侧字段(780/781/861/862)在这种流量下是否点亮（P2P 单播下它们全 0）。

单进程 spawn N 个 rank，每 rank 绑一张卡；容器 `--gpus '"device=..."'` 内部重编号 cuda:0..N-1。
NCCL 算法/协议由 NCCL_ALGO 环境变量控制（不设=auto；设 NVLS 强制走 multimem 路，设 Ring 做对照）。

用法：nccl_allreduce.py [SIZE_MB] [SECONDS]   （world_size = 容器可见 GPU 数）
"""
import os, sys, time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZE_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 512
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    n = SIZE_MB * 1024 * 1024 // 4
    x = torch.ones(n, dtype=torch.float32, device=f"cuda:{rank}")
    # warmup + 让 NCCL 建 NVLS team、打印 algo 选择（rank0 的 NCCL_DEBUG=INFO 里能看到）
    for _ in range(5):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.time(); end = t0 + SECONDS; it = 0
    if rank == 0:
        print(f"NCCL_ALLREDUCE_START world={world} size={SIZE_MB}MB algo={os.environ.get('NCCL_ALGO','auto')} {SECONDS}s", flush=True)
    while time.time() < end:
        dist.all_reduce(x)
        it += 1
        if it % 50 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize(); dist.barrier()
    elapsed = time.time() - t0
    if rank == 0:
        nbytes = SIZE_MB * 1024 * 1024
        # all_reduce busbw 约定：busbw = algbw * 2*(world-1)/world；algbw = nbytes/time
        algbw = it * nbytes / elapsed / 1e9
        busbw = algbw * 2 * (world - 1) / world
        print(f"NCCL_ALLREDUCE_END iters={it} elapsed={elapsed:.1f}s "
              f"algbw={algbw:.0f} GB/s busbw={busbw:.0f} GB/s", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    world = torch.cuda.device_count()
    print(f"props: {torch.cuda.get_device_name(0)} | world_size(visible GPUs)={world} | "
          f"NCCL_ALGO={os.environ.get('NCCL_ALGO','auto')} NCCL_PROTO={os.environ.get('NCCL_PROTO','auto')}", flush=True)
    if world < 2:
        print("ERROR: need >=2 visible GPUs", flush=True); sys.exit(1)
    mp.spawn(worker, args=(world,), nprocs=world, join=True)
