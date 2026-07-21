"""实验 3：卡间 D2D —— 纯 Copy Engine (DMA) over NVLink。看哪些 DCGM metric 点亮。

容器 `--gpus '"device=A,B"'` -> 内部 cuda:0(ordinal 0)=物理A, cuda:1(ordinal 1)=物理B。
用 ctypes 直接调 `cudaMemcpyPeerAsync(dst@ord1, src@ord0)`，单向 A->B。
预期：源卡 nvlink_tx_bytes、目的卡 nvlink_rx_bytes、两卡 nvlink_bandwidth_total(449) 点亮；
     两卡 dram_active/mem_copy_util 小幅点亮（源读/目的写）；sm_active≈0；pcie 应保持 0。
若实测带宽只有几十 GB/s，说明 P2P 走了 host 中转而非 NVLink（本机 NVSwitch 应直连 ~400GB/s）。
"""
import ctypes, sys, torch, time

cudart = ctypes.CDLL("libcudart.so")
cudart.cudaMemcpyPeerAsync.restype = ctypes.c_int
cudart.cudaMemcpyPeerAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_size_t, ctypes.c_void_p]
cudart.cudaSetDevice.argtypes = [ctypes.c_int]
cudart.cudaDeviceEnablePeerAccess.restype = ctypes.c_int
cudart.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]

def enable_p2p(dev, peer):
    cudart.cudaSetDevice(dev)
    r = cudart.cudaDeviceEnablePeerAccess(peer, 0)
    if r not in (0, 704):    # 704 = cudaErrorPeerAccessAlreadyEnabled
        raise RuntimeError(f"cudaDeviceEnablePeerAccess {dev}->{peer} rc={r}")

# 拷贝大小（MB）可由 argv[1] 指定，默认 2048MB(=2GB, 原始实验)。
# L2-fit 实验：src/dst 各 << 各卡 L2(H100=50MB)，看 NVLink 计量是否照旧、两卡 HBM(dram_active) 是否被 L2 挡掉。
SIZE_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
nbytes = SIZE_MB * 1024 * 1024
N = nbytes // 4
p = torch.cuda.get_device_properties(0)
print(f"props: {p.name} L2={p.L2_cache_size/1024/1024:.0f}MB | copy_size={SIZE_MB}MB/卡 "
      f"({'fits' if SIZE_MB < p.L2_cache_size/1024/1024 else 'busts'} 每卡 L2)", flush=True)
src = torch.empty(N, dtype=torch.float32, device='cuda:0')   # 物理A, ordinal 0
dst = torch.empty(N, dtype=torch.float32, device='cuda:1')   # 物理B, ordinal 1
enable_p2p(0, 1); enable_p2p(1, 0)         # 启用 P2P -> 走 NVLink 而非 host 中转
cudart.cudaSetDevice(0)
torch.cuda.synchronize()

print(f"D2D_INTER_START (cudaMemcpyPeer ord0->ord1, 单向, {SIZE_MB}MB/次)", flush=True)
end = time.time() + 20; n = 0
while time.time() < end:
    r = cudart.cudaMemcpyPeerAsync(ctypes.c_void_p(dst.data_ptr()), 1,
                                   ctypes.c_void_p(src.data_ptr()), 0,
                                   ctypes.c_size_t(nbytes), None)
    if r:
        raise RuntimeError(f"cudaMemcpyPeerAsync rc={r}")
    n += 1
    if n % 20 == 0:
        torch.cuda.synchronize()
torch.cuda.synchronize()
print(f"D2D_INTER_END iters={n} ~{n * nbytes / 20 / 1e9:.0f} GB/s (单向; NVLink 计量不受 L2 影响)", flush=True)
