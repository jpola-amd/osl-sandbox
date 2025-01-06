#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

const char* kernel_code = R"(
extern "C" __global__ void my_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = data[idx] * 2.0f;
}
)";


// load binary code into vector<char>
std::vector<char> loadBinaryFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (file.fail()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    const auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> binary(size);
    file.read(binary.data(), size);
    return binary;
}


int main(int argc, char **argv) {
  
    hipError_t result = hipInit(0);
    if (result != hipSuccess) {
        std::cerr << "Failed to initialize HIP" << std::endl;
        return 1;
    }



    hipDeviceProp_t props;
    result = hipGetDeviceProperties(&props, 0);
    std::cout << "Device name: " << props.name << std::endl;


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

    // Load the binary code into a module
    auto shadeops = loadBinaryFile("shadeops_hip.hsaco");
    if (shadeops.empty()) {
        std::cerr << "Failed to load binary file" << std::endl;
        return 1;
    }

    hiprtcLinkState linkState;
    hiprtcResult = hiprtcLinkCreate(0, nullptr, nullptr, &linkState);
    
    hiprtcResult = hiprtcLinkAddFile(linkState, HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE, "shadeops_hip.hsaco", 0, nullptr, nullptr);
    if (hiprtcResult != HIPRTC_SUCCESS) {
        std::cerr << "Failed to add file" << std::endl;
        return 1;
    }

    hiprtcResult = hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, code.data(), code.size(), "code", 0, nullptr, nullptr);
    if (hiprtcResult != HIPRTC_SUCCESS) {
        std::cerr << "Failed to add data" << std::endl;
        return 1;
    }

    void* tmpLinkedCode {nullptr};
    size_t linkedCodeSize {0};
    hiprtcResult = hiprtcLinkComplete(linkState, &tmpLinkedCode, &linkedCodeSize);

    std::vector<char> linkedCode (linkedCodeSize);
    std::memcpy(linkedCode.data(), tmpLinkedCode, linkedCodeSize);

    hiprtcResult = hiprtcLinkDestroy(linkState);

    hipModule_t module;
    result = hipModuleLoadData(&module, linkedCode.data());
    if (result != hipSuccess) {
        std::cerr << "Failed to load module" << std::endl;
        return 1;
    }   




    return 0;
}