#include <cuda_runtime.h>
// minimal kernel exercising multimem PTX (NVLS multicast reduce/store), sm_90
__global__ void k(float* mc, float* out) {
    float v;
    asm volatile("multimem.ld_reduce.global.add.f32 %0, [%1];" : "=f"(v) : "l"(mc));
    asm volatile("multimem.st.global.f32 [%0], %1;" :: "l"(mc), "f"(v));
    *out = v;
}
