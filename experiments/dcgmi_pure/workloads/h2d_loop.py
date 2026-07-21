#!/usr/bin/env python3
# 持续 H2D 拷贝，给 PCIe 制造**连续**流量(点亮 PCIRX=device 接收)，用于干净测量 DCGM 内部刷新率。
# 用法(容器内): python h2d_loop.py <seconds>
import sys, time, torch
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
dev = torch.device("cuda:0")
host = torch.empty(256*1024*1024//2, dtype=torch.float16).pin_memory()  # 256MB pinned
d = torch.empty_like(host, device=dev)
s = torch.cuda.Stream()
print("START h2d_loop", secs, "s", flush=True)
t0 = time.time(); n = 0
with torch.cuda.stream(s):
    while time.time() - t0 < secs:
        d.copy_(host, non_blocking=True)   # H2D
        n += 1
        if n % 2000 == 0:
            torch.cuda.synchronize()
torch.cuda.synchronize()
gb = n * host.numel() * 2 / 1e9
print(f"DONE copies={n} moved={gb:.1f}GB rate~{gb/(time.time()-t0):.1f}GB/s", flush=True)
