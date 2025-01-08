#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <filesystem>

//Kernel is using OSL function osl_round_ff that is defined in shadeops_hip.bc
const char* kernel_code = R"(

extern "C" __device__ float osl_round_ff(float x);

extern "C" __global__ void my_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = osl_round_ff(data[idx]) * 2.0f;
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
    std::cout << "Device name: " << props.name << std::endl;

    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

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

    //save code to file
    {
        std::ofstream file("code.bc", std::ios::binary);
        file.write(code.data(), code.size());
        file.close();
    }
    
/* Maybe usefull fol workaround how to pass spcific linker options */
/*
    const char* isaopts[] = {
        "-mllvm",
        "-inline-threshold=1",
        "-mllvm", 
        "-inlinehint-threshold=1",
        "-v",
        "-save-temps"
    };

    // Set up JIT options
    std::vector<hiprtcJIT_option> jit_options = {
        HIPRTC_JIT_IR_TO_ISA_OPT_EXT,
        HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT
    };
    
    // Number of LLVM options
    size_t isaoptssize = sizeof(isaopts) / sizeof(isaopts[0]);
    // Create options array
    const void* lopts[] = {
        static_cast<const void*>(isaopts),
        static_cast<const void*>(&isaoptssize)
    };
*/

    hiprtcLinkState linkState;
    hiprtcResult = hiprtcLinkCreate(0, nullptr, nullptr, &linkState);
    
    hiprtcResult = hiprtcLinkAddFile(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, "shadeops_hip.bc", 0, nullptr, nullptr);
    if (hiprtcResult != HIPRTC_SUCCESS) {
        std::cerr << "Failed to add file" << std::endl;
        return 1;
    }

    // hiprtcResult = hiprtcLinkAddData(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, code.data(), code.size(), "code", 0, nullptr, nullptr);
    hiprtcResult = hiprtcLinkAddFile(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, "code.bc", 0, nullptr, nullptr);
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




    return 0;
}