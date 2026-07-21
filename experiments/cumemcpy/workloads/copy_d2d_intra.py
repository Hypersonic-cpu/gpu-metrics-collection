"""实验 2：单卡内 D2D —— 纯 Copy Engine (DMA) 在同一张卡内拷贝。看哪些 DCGM metric 点亮。

容器 `--gpus '"device=X"'` -> 内部 cuda:0 = 物理卡 X。src/dst 都在这张卡上。
关键：用 ctypes 直接调 `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice)`，
**不用 torch 的 dst.copy_(src)**（torch 同卡 copy_ 走的是 kernel、会占满 SM，不是 copy engine）。
预期：dram_active/mem_copy_util 点亮（读 src + 写 dst，HBM 流量≈2×拷贝量）；sm_active 用来判定是否真走 DMA；
pcie/nvlink 应保持 0（片内拷贝不出卡）。
"""
import ctypes, sys, torch, time

cudart = ctypes.CDLL("libcudart.so")
cudart.cudaMemcpyAsync.restype = ctypes.c_int
cudart.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
D2D = 3  # cudaMemcpyKind: DeviceToDevice=3

# 拷贝大小（MB）可由 argv[1] 指定，默认 2048MB(=2GB, 原始实验)。
# L2-fit 实验：传一个 << L2 的值（如 16），src+dst 一起装进 L2(H100=50MB)，看 HBM 是否被 L2 挡掉。
SIZE_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
nbytes = SIZE_MB * 1024 * 1024
N = nbytes // 4
p = torch.cuda.get_device_properties(0)
print(f"props: {p.name} L2={p.L2_cache_size/1024/1024:.0f}MB | copy_size={SIZE_MB}MB "
      f"src+dst={2*SIZE_MB}MB ({'fits' if 2*SIZE_MB < p.L2_cache_size/1024/1024 else 'busts'} L2)", flush=True)
src = torch.empty(N, dtype=torch.float32, device='cuda:0')
dst = torch.empty(N, dtype=torch.float32, device='cuda:0')
torch.cuda.synchronize()

print(f"D2D_INTRA_START (同卡 cudaMemcpyDeviceToDevice, {SIZE_MB}MB/次)", flush=True)
end = time.time() + 20; n = 0
while time.time() < end:
    r = cudart.cudaMemcpyAsync(ctypes.c_void_p(dst.data_ptr()), ctypes.c_void_p(src.data_ptr()),
                               ctypes.c_size_t(nbytes), D2D, None)
    if r:
        raise RuntimeError(f"cudaMemcpyAsync rc={r}")
    n += 1
    if n % 20 == 0:
        torch.cuda.synchronize()
torch.cuda.synchronize()
print(f"D2D_INTRA_END iters={n} ~{n * nbytes / 20 / 1e9:.0f} GB/s (拷贝量; HBM 读+写≈2×，除非命中 L2)", flush=True)
