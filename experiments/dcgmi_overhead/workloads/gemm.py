"""compute-bound workload：大 GEMM（bf16, tensor core）。CUDA event 分块自计时，持续 ~TARGET_S 秒。

用于 dcgmi overhead 实验：计时区跨越几十~几百个 dcgmi 采样周期，才能测到稳态干扰。
每个"样本"= 计时 CHUNK 次 matmul 的总时间 / CHUNK = 单次 kernel 时间；样本分布用于 median/p95。

用法: python gemm.py [size] [target_s]   默认 size=8192, target_s=18
输出: RESULT workload=gemm size=.. samples=.. tflops=.. ms_median=.. ms_mean=.. ms_p95=.. ms_std=..
"""
import sys, time, torch

def pct(xs, q):
    s = sorted(xs); k = (len(s) - 1) * q; f = int(k)
    return s[f] if f + 1 >= len(s) else s[f] + (s[f + 1] - s[f]) * (k - f)

SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
TARGET_S = float(sys.argv[2]) if len(sys.argv) > 2 else 18.0
CHUNK = 50
dev = torch.device('cuda:0')
torch.backends.cuda.matmul.allow_tf32 = True
p = torch.cuda.get_device_properties(0)
a = torch.randn(SIZE, SIZE, device=dev, dtype=torch.bfloat16)
b = torch.randn(SIZE, SIZE, device=dev, dtype=torch.bfloat16)
flop = 2.0 * SIZE * SIZE * SIZE

def op():
    return torch.matmul(a, b)

for _ in range(200):  # warmup
    op()
torch.cuda.synchronize()
print(f"props: {p.name} | GEMM {SIZE}^3 bf16, {flop/1e12:.2f} TFLOP/iter, chunk={CHUNK}, target~{TARGET_S}s", flush=True)
print("GEMM_START", flush=True)

e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
samples = []
t_end = time.time() + TARGET_S
while time.time() < t_end:
    e0.record()
    for _ in range(CHUNK):
        c = op()
    e1.record(); torch.cuda.synchronize()
    samples.append(e0.elapsed_time(e1) / CHUNK)

med = pct(samples, 0.5); mean = sum(samples) / len(samples); p95 = pct(samples, 0.95)
std = (sum((x - mean) ** 2 for x in samples) / len(samples)) ** 0.5
tflops = flop / (med / 1e3) / 1e12
print("GEMM_END", flush=True)
print(f"RESULT workload=gemm size={SIZE} samples={len(samples)} tflops={tflops:.1f} "
      f"ms_median={med:.4f} ms_mean={mean:.4f} ms_p95={p95:.4f} ms_std={std:.4f}", flush=True)
