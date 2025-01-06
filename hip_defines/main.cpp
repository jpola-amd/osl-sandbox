#include <iostream>
#include <hip/hip_runtime.h>

// Kernel function
__global__ void test_hip_macros() {
#ifdef __HIP_DEVICE_COMPILE__
    printf("__HIP_DEVICE_COMPILE__ is defined on the device.\n");
#else
    printf("__HIP_DEVICE_COMPILE__ is not defined on the device.\n");
#endif

#ifdef __HIP__
   printf("__HIP__ is defined on the device\n");
#else
   printf("__HIP__ is NOT defined on the device \n");
#endif
}

int main() {
#ifdef __HIP__
    std::cout << "__HIP__ is defined on the host." << std::endl;
#else
    std::cout << "__HIP__ is not defined on the host." << std::endl;
#endif

    // Launch the kernel
    test_hip_macros<<<1, 1>>>();
    hipDeviceSynchronize();

    return 0;
}

