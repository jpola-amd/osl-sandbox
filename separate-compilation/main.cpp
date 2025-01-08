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


//TEST1 - load and link bc files together and load the linked bc file. The bc files are genrated by clang++20. The files are not bundled. 
// Target architecture is gfx1030
int link_test()
{
    hiprtcLinkState linkState;
    hiprtcResult resultHiprtc = hiprtcLinkCreate(0, nullptr, nullptr, &linkState);
    if (resultHiprtc != HIPRTC_SUCCESS) {
        std::cerr << "Failed to create link state" << std::endl;
        std::cerr << "Error code: " << resultHiprtc << " " << hiprtcGetErrorString(resultHiprtc) << std::endl;
        return 1;
    }
    // check if file exists
    
    if (!std::filesystem::exists("ops.bc")) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }
    resultHiprtc = hiprtcLinkAddFile(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, "ops.bc", 0, nullptr, nullptr);
    if (resultHiprtc != HIPRTC_SUCCESS) {
        std::cerr << "Failed to add file" << std::endl;
        std::cerr << "Error code: " << resultHiprtc << " " << hiprtcGetErrorString(resultHiprtc) << std::endl;
        return 1;
    }

    if (!std::filesystem::exists("kernel.bc")) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }
    resultHiprtc = hiprtcLinkAddFile(linkState, HIPRTC_JIT_INPUT_LLVM_BITCODE, "kernel.bc", 0, nullptr, nullptr);
    if (resultHiprtc != HIPRTC_SUCCESS) {
        std::cerr << "Failed to add file" << std::endl;
        std::cerr << "Error code: " << resultHiprtc << " " << hiprtcGetErrorString(resultHiprtc) << std::endl;
        return 1;
    }

    void* tmpLinkedCode {nullptr};
    size_t linkedCodeSize {0};
    resultHiprtc = hiprtcLinkComplete(linkState, &tmpLinkedCode, &linkedCodeSize);
    if (resultHiprtc != HIPRTC_SUCCESS) {
        std::cerr << "Failed to link" << std::endl;
        std::cerr << "Error code: " << resultHiprtc << " " << hiprtcGetErrorString(resultHiprtc) << std::endl;
        return 1;
    }

    std::vector<char> linkedCode (linkedCodeSize);
    std::memcpy(linkedCode.data(), tmpLinkedCode, linkedCodeSize);

    //save the code
    std::ofstream out("runtime-compiled.hsaco", std::ios::binary);
    out.write(linkedCode.data(), linkedCodeSize);
    out.close();

    hipModule_t module;
    hipError_t result = hipModuleLoadData(&module, linkedCode.data());
    if (result != hipSuccess) {
        std::cerr << "Failed to load module" << std::endl;
        std::cerr << "Error code: " << result << std::endl;
        std::cerr << "Error message: " << hipGetErrorString(result) << std::endl;
        return 1;
    }
    
    // get the kernel function
    hipFunction_t kernelFunction;
    result = hipModuleGetFunction(&kernelFunction, module, "my_kernel");
    if (result != hipSuccess) {
        std::cerr << "Failed to get kernel function" << std::endl;
        std::cerr << "Error code: " << result << std::endl;
        std::cerr << "Error message: " << hipGetErrorString(result) << std::endl;
        return 1;
    }
    return 0;
}


//TEST2 - 
int load_llc_generated_hsaco()
{
    hipModule_t module;
    if (!std::filesystem::exists("device_code.clang.hsaco")) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }
    hipError_t result = hipModuleLoad(&module, "device_code.llc.hsaco");
    if (result != hipSuccess) {
        std::cerr << "Failed to load module" << std::endl;
        std::cerr << "Error code: " << result << std::endl;
        std::cerr << "Error message: " << hipGetErrorString(result) << std::endl;
        return 1;
    }
    return 0;
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
    // display the current working directory
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;

    //TEST1
    if (link_test())
    {
        std::cerr << "Failed to link" << std::endl;
    }

    if (load_llc_generated_hsaco())
    {
        std::cerr << "Failed to generate by linker" << std::endl;
    }
   
    return EXIT_SUCCESS;
}