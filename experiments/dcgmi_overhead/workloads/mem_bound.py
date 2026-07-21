"""memory-bound workload：打满 HBM 的融合逐元素算子（残差 add + scale，LLM 里最常见的访存密集算子）。
CUDA event 分块自计时，持续 ~TARGET_S 秒。

选它而非 RMSNorm：eager RMSNorm 有 fp32 上转/reduction，多 kernel/launch-bound（实测仅 ~225 GB/s），
打不满 HBM。这里 `z = a + s*b`（单 kernel、读a+读b+写z = 3×流量）实测 ~3100 GB/s，逼近 HBM 峰值(~3350)，
最大化对 dcgmi HBM 计数器(dram_active)采集的敏感度。

用法: python mem_bound.py [elems_M] [target_s]  默认 elems=512(M bf16), target_s=18
输出: RESULT workload=mem_addscale bytes=.. samples=.. gbps=.. ms_median=.. ms_mean=.. ms_p95=.. ms_std=..
"""
import sys, time, torch

def pct(xs, q):
    s = sorted(xs); k = (len(s) - 1) * q; f = int(k)
    return s[f] if f + 1 >= len(s) else s[f] + (s[f + 1] - s[f]) * (k - f)

ELEMS = (int(sys.argv[1]) if len(sys.argv) > 1 else 512) * 1024 * 1024
TARGET_S = float(sys.argv[2]) if len(sys.argv) > 2 else 18.0
CHUNK = 100
dev = torch.device('cuda:0')
p = torch.cuda.get_device_properties(0)
a = torch.randn(ELEMS, dtype=torch.bfloat16, device=dev)
b = torch.randn(ELEMS, dtype=torch.bfloat16, device=dev)
z = torch.empty(ELEMS, dtype=torch.bfloat16, device=dev)
s = 1.01
bytes_moved = 3.0 * ELEMS * 2  # 读a+读b+写z

def op():
    torch.add(a, b, alpha=s, out=z)  # z = a + s*b, 单 kernel

for _ in range(200):  # warmup
    op()
torch.cuda.synchronize()
print(f"props: {p.name} | add-scale z=a+{s}*b (单 kernel), {ELEMS/1e6:.0f}M bf16, ~{bytes_moved/1e9:.2f} GB/iter (3x), "
      f"chunk={CHUNK}, target~{TARGET_S}s", flush=True)
print("MEM_START", flush=True)

e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
samples = []
t_end = time.time() + TARGET_S
while time.time() < t_end:
    e0.record()
    for _ in range(CHUNK):
        op()
    e1.record(); torch.cuda.synchronize()
    samples.append(e0.elapsed_time(e1) / CHUNK)

med = pct(samples, 0.5); mean = sum(samples) / len(samples); p95 = pct(samples, 0.95)
std = (sum((v - mean) ** 2 for v in samples) / len(samples)) ** 0.5
gbps = bytes_moved / (med / 1e3) / 1e9
print("MEM_END", flush=True)
print(f"RESULT workload=mem_addscale bytes={int(bytes_moved)} samples={len(samples)} gbps={gbps:.1f} "
      f"ms_median={med:.4f} ms_mean={mean:.4f} ms_p95={p95:.4f} ms_std={std:.4f}", flush=True)
