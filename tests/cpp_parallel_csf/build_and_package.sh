#!/bin/bash

# Complete build and package script for CSF Descriptor
# This script demonstrates the full packaging workflow

set -e

echo "=== CSF Descriptor Build and Package Script ==="
echo ""

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 显示系统信息
echo "System Information:"
echo "  OS: $(uname -s)"
echo "  Architecture: $(uname -m)"
echo "  C++ Compiler: $(c++ --version | head -n1)"
echo "  CMake: $(cmake --version | head -n1)"
echo ""

# 步骤1: 清理并创建构建目录
echo "Step 1: Preparing build environment..."
rm -rf build packages
mkdir -p build packages

# 步骤2: 配置构建
echo "Step 2: Configuring build..."
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DCPACK_PACKAGE_VERSION_MAJOR=1 \
    -DCPACK_PACKAGE_VERSION_MINOR=0 \
    -DCPACK_PACKAGE_VERSION_PATCH=0

# 步骤3: 编译
echo "Step 3: Building..."
make -j$(nproc)

# 步骤4: 运行测试
echo "Step 4: Running tests..."
ctest --output-on-failure || echo "Warning: Some tests failed"

# 步骤5: 创建安装包
echo "Step 5: Creating packages..."
make package

# 步骤6: 显示生成的包
echo "Step 6: Generated packages:"
ls -la *.tar.gz *.deb *.rpm 2>/dev/null || echo "No packages generated"

# 步骤7: 移动到packages目录
mv *.tar.gz *.deb *.rpm ../packages/ 2>/dev/null || true

# 步骤8: 测试安装
echo "Step 8: Testing installation..."
make install DESTDIR=/tmp/csf-test-install

# 步骤9: 显示结果
echo ""
echo "=== Build Complete ==="
echo "Packages created in: $SCRIPT_DIR/packages/"
echo "Installation test in: /tmp/csf-test-install"
echo ""
echo "To install system-wide:"
echo "  cd build && sudo make install"
echo ""
echo "To create specific packages:"
echo "  make package TGZ    # Create tar.gz"
echo "  make package DEB    # Create .deb (Debian/Ubuntu)"
echo "  make package RPM    # Create .rpm (CentOS/RHEL)"
echo ""
echo "To uninstall:"
echo "  cd build && make uninstall"