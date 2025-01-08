

__device__ float ops_sub(float a, float b) {
    return a - b;
}

__device__ float ops_fma(float a, float b, float c) {
    return fmaf(a, b, c);
}

__device__ float ops_sqrt(float a) {
    return sqrtf(a);
}