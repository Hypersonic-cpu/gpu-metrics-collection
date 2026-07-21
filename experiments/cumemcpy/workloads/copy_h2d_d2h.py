"""实验 1：H2D + D2H —— 纯 Copy Engine (DMA) 过 PCIe。看哪些 DCGM metric 点亮。

容器 `--gpus '"device=X"'` -> 内部 cuda:0 = 物理卡 X。
用 ctypes 直接调 `cudaMemcpyAsync(HostToDevice / DeviceToHost)`，pinned host 内存，
确保走 Copy Engine (DMA)、不落到 SM kernel。两段：先 H2D（host->device），后 D2H（device->host）。
预期：pcie_rx_bytes(H2D) / pcie_tx_bytes(D2H) 点亮；sm_active≈0；dram_active/mem_copy_util 按 HBM 流量小幅点亮。
"""
import ctypes, sys, torch, time

cudart = ctypes.CDLL("libcudart.so")
cudart.cudaMemcpyAsync.restype = ctypes.c_int
cudart.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
H2D, D2H = 1, 2  # cudaMemcpyKind: HostToDevice=1, DeviceToHost=2

def memcpy(dst_ptr, src_ptr, nbytes, kind):
    r = cudart.cudaMemcpyAsync(ctypes.c_void_p(dst_ptr), ctypes.c_void_p(src_ptr),
                               ctypes.c_size_t(nbytes), kind, None)
    if r:
        raise RuntimeError(f"cudaMemcpyAsync rc={r}")

# 拷贝大小（MB）可由 argv[1] 指定，默认 2048MB(=2GB, 原始实验)。
# L2-fit 实验：device buffer << L2(H100=50MB)，看 PCIe 计量是否照旧、HBM(dram_active) 是否被 L2 挡掉。
SIZE_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
nbytes = SIZE_MB * 1024 * 1024
N = nbytes // 4
p = torch.cuda.get_device_properties(0)
print(f"props: {p.name} L2={p.L2_cache_size/1024/1024:.0f}MB | copy_size={SIZE_MB}MB "
      f"({'fits' if SIZE_MB < p.L2_cache_size/1024/1024 else 'busts'} L2)", flush=True)
dev = torch.empty(N, dtype=torch.float32, device='cuda:0')
host = torch.empty(N, dtype=torch.float32, pin_memory=True)   # pinned -> copy engine
torch.cuda.synchronize()

def run(phase, kind, secs):
    print(f"{phase}_START", flush=True)
    end = time.time() + secs; n = 0
    while time.time() < end:
        if kind == H2D:
            memcpy(dev.data_ptr(), host.data_ptr(), nbytes, kind)
        else:
            memcpy(host.data_ptr(), dev.data_ptr(), nbytes, kind)
        n += 1
        if n % 20 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    print(f"{phase}_END iters={n} ~{n * nbytes / secs / 1e9:.0f} GB/s", flush=True)

run("H2D", H2D, 12)     # host -> device, 过 PCIe
time.sleep(4)
run("D2H", D2H, 12)     # device -> host, 过 PCIe
print("ALL_DONE", flush=True)
