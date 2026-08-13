#!/usr/bin/env python3

import time
import torch


DEVICE = 0
N = 4096
PRE_SECONDS = 2.0
ROI_SECONDS = 5.0
POST_SECONDS = 2.0
NVTX_RANGE = "HTM_TARGET"


def run_for(seconds, fn):
    end = time.perf_counter() + seconds
    count = 0
    while time.perf_counter() < end:
        fn()
        count += 1
    return count


def main():
    torch.cuda.set_device(DEVICE)
    print(f"DEVICE={DEVICE}", flush=True)
    print(f"GPU={torch.cuda.get_device_name(DEVICE)}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda={torch.version.cuda}", flush=True)

    print("INIT_BEGIN", flush=True)
    a = torch.randn((N, N), device=f"cuda:{DEVICE}", dtype=torch.bfloat16)
    b = torch.randn((N, N), device=f"cuda:{DEVICE}", dtype=torch.bfloat16)
    tmp = torch.empty_like(a)
    c = torch.empty_like(a)
    torch.sin(a, out=tmp)
    torch.matmul(a, b, out=c)
    torch.cos(a, out=tmp)
    torch.cuda.synchronize()
    print("INIT_END", flush=True)

    print("PRE_BEGIN", flush=True)
    pre_count = run_for(PRE_SECONDS, lambda: torch.sin(a, out=tmp))
    torch.cuda.synchronize()
    print(f"PRE_END iterations={pre_count}", flush=True)

    print("ROI_RANGE_PUSH", flush=True)
    torch.cuda.nvtx.range_push(NVTX_RANGE)
    print("ROI_BEGIN", flush=True)
    roi_count = run_for(ROI_SECONDS, lambda: torch.matmul(a, b, out=c))
    torch.cuda.synchronize()
    print(f"ROI_END iterations={roi_count}", flush=True)
    torch.cuda.nvtx.range_pop()
    print("ROI_RANGE_POP", flush=True)

    print("POST_BEGIN", flush=True)
    post_count = run_for(POST_SECONDS, lambda: torch.cos(a, out=tmp))
    torch.cuda.synchronize()
    print(f"POST_END iterations={post_count}", flush=True)
    print("TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
