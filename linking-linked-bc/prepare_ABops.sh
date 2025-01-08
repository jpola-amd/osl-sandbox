
export CLANG_EXE="../../3rdParty/llvm-artifacts-release/bin/clang++"
export LLVM_LINK_EXE="../../3rdParty/llvm-artifacts-release/bin/llvm-link"
export LLVM_OPT_EXE="../../3rdParty/llvm-artifacts-release/bin/opt"
export GPU_ARCH="gfx1030"

${CLANG_EXE} -x hip -fgpu-rdc --cuda-device-only --offload-arch=${GPU_ARCH} -emit-llvm -c -o opsA.bc opsA.cpp
${CLANG_EXE} -x hip -fgpu-rdc --cuda-device-only --offload-arch=${GPU_ARCH} -emit-llvm -c -o opsB.bc opsB.cpp
${LLVM_LINK_EXE} opsA.bc opsB.bc -o opsAB.bc

