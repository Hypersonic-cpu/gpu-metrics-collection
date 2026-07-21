"""communication workload：NCCL 双向 sendrecv（两卡 over NVLink）。CUDA event 分块自计时。

torchrun --nproc_per_node=2：rank0=cuda:0(物理A)、rank1=cuda:1(物理B)。每次迭代两 rank 互 send+recv
（batch_isend_irecv）→ 打满 NVLink 双向。集合通信两 rank 必须同迭代数，故用**固定 chunk 数**（非 wall-time）。
报告单向 busbw = size / time。

用法: torchrun --nproc_per_node=2 comm_nccl.py [size_mb] [n_chunks]  默认 256MB, n_chunks=360(~16s)
输出(仅 rank0): RESULT workload=comm_nccl size_mb=.. samples=.. gbps=.. ms_median=.. ms_mean=.. ms_p95=.. ms_std=..
"""
import sys, torch, torch.distributed as dist

def pct(xs, q):
    s = sorted(xs); k = (len(s) - 1) * q; f = int(k)
    return s[f] if f + 1 >= len(s) else s[f] + (s[f + 1] - s[f]) * (k - f)

SIZE_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 256
N_CHUNKS = int(sys.argv[2]) if len(sys.argv) > 2 else 360
CHUNK = 50
dist.init_process_group("nccl")
rank = dist.get_rank(); world = dist.get_world_size()
assert world == 2, "本测试固定 2 卡"
torch.cuda.set_device(rank)
dev = torch.device(f'cuda:{rank}')
peer = 1 - rank
n = SIZE_MB * 1024 * 1024 // 2  # bf16 元素数
send = torch.ones(n, dtype=torch.bfloat16, device=dev)
recv = torch.empty(n, dtype=torch.bfloat16, device=dev)
bytes_dir = float(n * 2)  # 单向字节数

def op():
    ops = [dist.P2POp(dist.isend, send, peer), dist.P2POp(dist.irecv, recv, peer)]
    for r in dist.batch_isend_irecv(ops):
        r.wait()

for _ in range(200):  # warmup
    op()
torch.cuda.synchronize(); dist.barrier()
if rank == 0:
    p = torch.cuda.get_device_properties(0)
    print(f"props: {p.name} | NCCL sendrecv {SIZE_MB}MB/dir bf16, 2 ranks, chunk={CHUNK}, n_chunks={N_CHUNKS}", flush=True)
    print("COMM_START", flush=True)

e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
samples = []
for _ in range(N_CHUNKS):
    e0.record()
    for _ in range(CHUNK):
        op()
    e1.record(); torch.cuda.synchronize()
    samples.append(e0.elapsed_time(e1) / CHUNK)
dist.barrier()

if rank == 0:
    med = pct(samples, 0.5); mean = sum(samples) / len(samples); p95 = pct(samples, 0.95)
    std = (sum((v - mean) ** 2 for v in samples) / len(samples)) ** 0.5
    gbps = bytes_dir / (med / 1e3) / 1e9
    print("COMM_END", flush=True)
    print(f"RESULT workload=comm_nccl size_mb={SIZE_MB} samples={len(samples)} gbps={gbps:.1f} "
          f"ms_median={med:.4f} ms_mean={mean:.4f} ms_p95={p95:.4f} ms_std={std:.4f}", flush=True)
dist.destroy_process_group()
