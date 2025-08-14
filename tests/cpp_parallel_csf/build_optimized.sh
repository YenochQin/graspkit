#!/bin/bash

# 优化的构建脚本
set -e

echo "开始优化构建..."

# 创建构建目录
mkdir -p build
cd build

# 使用优化的编译标志
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DNDEBUG -fno-math-errno -ffast-math" \
    -DCMAKE_CXX_COMPILER_LAUNCHER="ccache" \
    -DCMAKE_VERBOSE_MAKEFILE=ON

# 并行编译
make -j$(nproc)

echo "构建完成！优化后的程序位于: ./csf_descriptor"
echo "运行示例:"
echo "  ./csf_descriptor -t 8 your_file.c"
echo "  ./csf_descriptor -t 8 -e your_file.c  # 使用扩展格式"