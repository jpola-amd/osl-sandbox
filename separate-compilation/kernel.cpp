#include <hip/hip_runtime.h>

extern __device__ float osl_round(float x);

extern "C" __global__ void my_kernel(float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = osl_round(a[i]);
    }
}