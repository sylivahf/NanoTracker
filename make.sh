#!/bin/bash

# 设置工具链路径
TOOL_PATH="${HOME}/rk3588/rk3588_linux_release_v1.2.1/extra-tools"
C_COMPILER="${TOOL_PATH}/gcc-linaro-10.2.1-2021.01-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc"
CXX_COMPILER="${TOOL_PATH}/gcc-linaro-10.2.1-2021.01-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++"

# 检查工具链是否存在
if [ ! -f "${C_COMPILER}" ]; then
    echo "错误：交叉编译器不存在: ${C_COMPILER}"
    echo "请检查 TOOL_PATH 是否正确"
    exit 1
fi

echo "使用交叉编译器:"
echo "  C编译器: ${C_COMPILER}"
echo "  C++编译器: ${CXX_COMPILER}"

# 检查当前目录是否存在 build 文件夹
if [ -d "build" ]; then
  echo "发现 build 文件夹，清空文件夹内容..."
  rm -rf build/*
  rm -rf build/CMakeCache.txt build/CMakeFiles
else
  echo "未发现 build 文件夹，创建 build 文件夹..."
  mkdir build
fi

# 进入 build 文件夹
cd build

# 执行 CMake 命令
echo "执行 cmake 配置..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS=" -g2 -O3" \
      -DCMAKE_C_FLAGS=" -g2 -O3" ..

# 检查 CMake 是否成功
if [ $? -eq 0 ]; then
  echo "CMake 配置成功！"
else
  echo "CMake 配置失败，请检查错误信息！"
  #exit 1
fi
# 执行 make 编译
echo "开始执行 目标文件 debug编译"
#make -j6 VERBOSE=1
make -j4

# 检查 make 是否成功
if [ $? -eq 0 ]; then
  echo "make 编译成功！"
else
  echo "make 编译失败，请检查错误信息！"
fi

# 返回到 build 文件夹的上级目录
cd ..
echo "已退回到 build 的上级目录：$(pwd)"
