"""NVSwitch 实验 workload：cross-card cudaMemcpyPeer + P2P，打满 NVLink（流量过 NVSwitch）。

复用自 experiments/cumemcpy/workloads/copy_d2d_inter.py（卡间 D2D，纯 Copy Engine over NVLink），
这里加了 **P2P 可达性显式核对**（cudaDeviceCanAccessPeer）+ 更长的默认时长，方便宿主机 dcgmi dmon 抓 switch 侧字段。

容器 `--gpus '"device=A,B"'` -> 内部 cuda:0(ordinal 0)=物理A, cuda:1(ordinal 1)=物理B。
用 ctypes 直接调 `cudaMemcpyPeerAsync(dst@ord1, src@ord0)`，单向 A->B。
本机 8×H100 全部接 NVSwitch（非直连），启 P2P 后流量必过交换机 → 用来点亮 NVSwitch 侧字段。

⚠️ P2P 坑：若 P2P 未启用/不可达，cudaMemcpyPeer 退化成走 host 中转(PCIe)（实测只 37GB/s、nvlink=0），
   流量根本不过 NVSwitch。故先 cudaDeviceCanAccessPeer 核对、并打印实测带宽（~400GB/s=真 NVLink，
   ~37GB/s=退化到 PCIe），宿主机再用 nvlink_tx/rx_bytes(1011/1012) 交叉确认。

用法：nvlink_p2p_saturate.py [SIZE_MB] [SECONDS]
"""
import ctypes, sys, torch, time

cudart = ctypes.CDLL("libcudart.so")
cudart.cudaMemcpyPeerAsync.restype = ctypes.c_int
cudart.cudaMemcpyPeerAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                        ctypes.c_size_t, ctypes.c_void_p]
cudart.cudaSetDevice.argtypes = [ctypes.c_int]
cudart.cudaDeviceEnablePeerAccess.restype = ctypes.c_int
cudart.cudaDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
cudart.cudaDeviceCanAccessPeer.restype = ctypes.c_int
cudart.cudaDeviceCanAccessPeer.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]

def can_access_peer(dev, peer):
    flag = ctypes.c_int(-1)
    r = cudart.cudaDeviceCanAccessPeer(ctypes.byref(flag), dev, peer)
    if r:
        raise RuntimeError(f"cudaDeviceCanAccessPeer {dev}->{peer} rc={r}")
    return flag.value

def enable_p2p(dev, peer):
    cudart.cudaSetDevice(dev)
    r = cudart.cudaDeviceEnablePeerAccess(peer, 0)
    if r not in (0, 704):    # 704 = cudaErrorPeerAccessAlreadyEnabled
        raise RuntimeError(f"cudaDeviceEnablePeerAccess {dev}->{peer} rc={r}")

SIZE_MB = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
nbytes = SIZE_MB * 1024 * 1024
N = nbytes // 4
p = torch.cuda.get_device_properties(0)
print(f"props: {p.name} | copy_size={SIZE_MB}MB/次 | run={SECONDS}s", flush=True)

# 先核对 P2P 可达（0=不可达 -> 会走 host/PCIe；1=可达 -> 走 NVLink over NVSwitch）
c01 = can_access_peer(0, 1); c10 = can_access_peer(1, 0)
print(f"P2P_CANACCESS ord0->ord1={c01} ord1->ord0={c10} "
      f"({'OK: NVLink 路径' if c01 and c10 else 'WARN: 不可达 -> 会退化到 PCIe host 中转!'})", flush=True)

src = torch.empty(N, dtype=torch.float32, device='cuda:0')   # 物理A, ordinal 0
dst = torch.empty(N, dtype=torch.float32, device='cuda:1')   # 物理B, ordinal 1
enable_p2p(0, 1); enable_p2p(1, 0)         # 启用 P2P -> 走 NVLink 而非 host 中转
cudart.cudaSetDevice(0)
torch.cuda.synchronize()

print(f"D2D_INTER_START (cudaMemcpyPeer ord0->ord1, 单向, {SIZE_MB}MB/次, {SECONDS}s)", flush=True)
t0 = time.time(); end = t0 + SECONDS; n = 0
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
elapsed = time.time() - t0
bw = n * nbytes / elapsed / 1e9
print(f"D2D_INTER_END iters={n} elapsed={elapsed:.1f}s ~{bw:.0f} GB/s (单向) "
      f"[{'真 NVLink' if bw > 150 else 'WARN: 疑似退化到 PCIe host 中转'}]", flush=True)
