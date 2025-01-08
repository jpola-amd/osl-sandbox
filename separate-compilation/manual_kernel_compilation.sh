
../../3rdParty/llvm-artifacts-release/bin/clang++ -x hip -fgpu-rdc --cuda-device-only --offload-arch=gfx1030 -emit-llvm  -c -o kernel.bc kernel.cpp
../../3rdParty/llvm-artifacts-release/bin/clang++ -x hip -fgpu-rdc --cuda-device-only --offload-arch=gfx1030 -emit-llvm  -c -o ops.bc ops.cpp

../../3rdParty/llvm-artifacts-release/bin/llvm-link ops.bc kernel.bc  -o device_code_linked.bc
../../3rdParty/llvm-artifacts-release/bin/opt device_code_linked.bc -o device_code_linked.opt.bc

#../../3rdParty/llvm-artifacts-release/bin/llc --march=amdgcn -mcpu=gfx1030 -filetype=obj device_code_linked.opt.bc -o device_code.llc.hsaco
../../3rdParty/llvm-artifacts-release/bin/clang++ --hip-link device_code_linked.opt.bc -v --offload-arch=gfx1030 --cuda-device-only -o device_code.clang.hsaco