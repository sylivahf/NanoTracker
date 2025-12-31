# toolchain.cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g2")

set(CMAKE_C_FLAGS_DEBUG "-O0 -g2 -fno-omit-frame-pointer" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -fno-omit-frame-pointer" CACHE STRING "" FORCE)

# Also apply these flags directly (for safety)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g2 -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g2 -fno-omit-frame-pointer")

set(HOME_PATH /home/itds)
set(TOOL_PATH ${HOME_PATH}/rk3588/rk3588_linux_release_v1.2.1/extra-tools)
set(SYSROOT_PATH ${TOOL_PATH}/gcc-linaro-10.2.1-2021.01-x86_64_aarch64-linux-gnu/aarch64-linux-gnu/libc
                 ${CMAKE_CURRENT_SOURCE_DIR}/mainApp/algorithms/3rd_party_libs/libopencv4
                 ${CMAKE_CURRENT_SOURCE_DIR}/lib)

set(CMAKE_FIND_ROOT_PATH ${SYSROOT_PATH})

set(CMAKE_C_COMPILER "${TOOL_PATH}/gcc-linaro-10.2.1-2021.01-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${TOOL_PATH}/gcc-linaro-10.2.1-2021.01-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++")
set(CMAKE_LINKER "${TOOL_PATH}/gcc-linaro-10.2.1-2021.01-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-ld")