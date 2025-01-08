#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <filesystem>
const char* kernel_code = R"(
extern "C" __global__ void my_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;
}
)";


int main(int argc, char **argv) {
  
    hipError_t result = hipInit(0);
    if (result != hipSuccess) {
        std::cerr << "Failed to initialize HIP" << std::endl;
        return 1;
    }

    hipDeviceProp_t props;
    result = hipGetDeviceProperties(&props, 0);
    if (result != hipSuccess) {
        std::cerr << "Failed to get device properties" << std::endl;
        return 1;
    }
    std::cout << "Using device " << props.name << std::endl;
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    std::cout << "Preparing to compile the kernel" << std::endl;

    hiprtcProgram prog;
    hiprtcResult hiprtcResult = hiprtcCreateProgram(&prog, kernel_code, "my_kernel.cu", 0, nullptr, nullptr);

    if (hiprtcResult != HIPRTC_SUCCESS) {
        std::cerr << "Failed to create program" << std::endl;
        return 1;
    }

    //add options 
    const char* options[] = {"--offload-arch=gfx1030", "-fgpu-rdc", nullptr};

    hiprtcResult = hiprtcCompileProgram(prog, 2, options);
    if (hiprtcResult != HIPRTC_SUCCESS) {
        std::cerr << "Failed to compile program" << std::endl;
        {
            //get the compilation log
            size_t logSize;
            hiprtcResult = hiprtcGetProgramLogSize(prog, &logSize);
            if (hiprtcResult != HIPRTC_SUCCESS) {
                std::cerr << "Failed to get log size" << std::endl;
                return 1;
            }
            char* log = new char[logSize];
            hiprtcResult = hiprtcGetProgramLog(prog, log);
            if (hiprtcResult != HIPRTC_SUCCESS) {
                std::cerr << "Failed to get log" << std::endl;
                delete[] log;
                return 1;
            }
            std::cerr << log << std::endl;
            delete[] log;

        } 
        return 1;
    }

    size_t codeSize {0};
    hiprtcResult = hiprtcGetBitcodeSize(prog, &codeSize);
    if (hiprtcResult != HIPRTC_SUCCESS) {
        std::cerr << "Failed to get code size" << std::endl;
        return 1;
    }
    std::vector<char> code(codeSize);
    hiprtcResult = hiprtcGetBitcode(prog, code.data());
    if (hiprtcResult != HIPRTC_SUCCESS) {
        std::cerr << "Failed to get code" << std::endl;
        return 1;
    }

    hiprtcDestroyProgram(&prog);

    std::cout << "Kernel compiled successfully" << std::endl;
    std::cout << "Saving the bitcode to kernel.bc" << std::endl;
    {
        std::ofstream file("kernel.bc", std::ios::binary);
        file.write(code.data(), code.size());
        file.close();
    }
    

    hiprtcLinkState linkState;
    hiprtcResult = hiprtcLinkCreate(0, nullptr, nullptr, &linkState);
    if(hiprtcResult != HIPRTC_SUCCESS) {
        std::cerr << "Failed to create link state" << std::endl;
        return 1;
    }

    hiprtcResult = hiprtcLinkAddFile(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, "shadeops_hip.gcn.bc", 0, nullptr, nullptr);
    if (hiprtcResult != HIPRTC_SUCCESS) {
        std::cerr << "Failed to add file" << std::endl;
        return 1;
    }

    hiprtcResult = hiprtcLinkAddFile(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, "kernel.bc", 0, nullptr, nullptr);
    if (hiprtcResult != HIPRTC_SUCCESS) {
        std::cerr << "Failed to add data" << std::endl;
        return 1;
    }

    void* tmpLinkedCode {nullptr};
    size_t linkedCodeSize {0};
    hiprtcResult = hiprtcLinkComplete(linkState, &tmpLinkedCode, &linkedCodeSize);
    if (hiprtcResult != HIPRTC_SUCCESS) {
        std::cerr << "Failed to link" << std::endl;
        return 1;
    }

    std::vector<char> linkedCode (linkedCodeSize);
    std::memcpy(linkedCode.data(), tmpLinkedCode, linkedCodeSize);

    hiprtcResult = hiprtcLinkDestroy(linkState);

    hipModule_t module;
    result = hipModuleLoadData(&module, linkedCode.data());
    if (result != hipSuccess) {
        std::cerr << "Failed to load module" << std::endl;
        return 1;
    }   

    std::cout << "Module loaded successfully" << std::endl;
    std::cout << "End of the program" << std::endl;

    return 0;
}