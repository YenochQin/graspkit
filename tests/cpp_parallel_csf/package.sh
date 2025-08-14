#!/bin/bash

# CSF Descriptor Packaging Script
# This script creates packages for different platforms

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
PACKAGE_DIR="$PROJECT_DIR/packages"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# 检查依赖
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v cmake &> /dev/null; then
        error "CMake is required but not installed"
    fi
    
    if ! command -v make &> /dev/null; then
        error "Make is required but not installed"
    fi
    
    # 检查平台特定工具
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v dpkg &> /dev/null; then
            log "Found dpkg - can create .deb packages"
        fi
        if command -v rpmbuild &> /dev/null; then
            log "Found rpmbuild - can create .rpm packages"
        fi
    fi
}

# 创建构建目录
setup_build() {
    log "Setting up build environment..."
    
    mkdir -p "$BUILD_DIR"
    mkdir -p "$PACKAGE_DIR"
    
    cd "$BUILD_DIR"
    
    # 配置构建
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local
    
    # 编译
    make -j$(nproc)
    
    # 运行测试
    if [ -f "$BUILD_DIR/CTestTestfile.cmake" ]; then
        log "Running tests..."
        ctest --output-on-failure || warn "Some tests failed"
    fi
}

# 创建源码包
create_source_package() {
    log "Creating source package..."
    
    cd "$PROJECT_DIR"
    
    # 创建干净的源码目录
    local version=$(grep 'project.*VERSION' CMakeLists.txt | sed 's/.*VERSION \([0-9.]*\).*/\1/')
    local src_dir="csf-descriptor-${version}"
    local temp_dir="/tmp/${src_dir}"
    
    # 复制源码
    rm -rf "$temp_dir"
    cp -r . "$temp_dir"
    
    # 清理不必要的文件
    cd "$temp_dir"
    rm -rf build packages .git .gitignore
    find . -name "*.o" -delete 2>/dev/null || true
    find . -name "*.a" -delete 2>/dev/null || true
    
    # 创建压缩包
    cd /tmp
    tar -czf "$PACKAGE_DIR/${src_dir}.tar.gz" "$src_dir"
    
    log "Source package created: $PACKAGE_DIR/${src_dir}.tar.gz"
}

# 创建二进制包
create_binary_packages() {
    log "Creating binary packages..."
    
    cd "$BUILD_DIR"
    
    # 创建安装包
    make package || warn "Package creation failed"
    
    # 移动生成的包到packages目录
    if ls *.tar.gz *.deb *.rpm *.dmg *.exe *.zip 1> /dev/null 2>&1; then
        mv *.tar.gz *.deb *.rpm *.dmg *.exe *.zip "$PACKAGE_DIR/" 2>/dev/null || true
    fi
}

# 创建特定平台的安装器
create_platform_packages() {
    case "$OSTYPE" in
        linux-gnu*)
            log "Creating Linux packages..."
            create_linux_packages
            ;;
        darwin*)
            log "Creating macOS packages..."
            create_macos_packages
            ;;
        msys*|mingw*|cygwin*)
            log "Creating Windows packages..."
            create_windows_packages
            ;;
        *)
            warn "Unknown platform: $OSTYPE"
            ;;
    esac
}

# Linux特定包创建
create_linux_packages() {
    cd "$BUILD_DIR"
    
    # 创建DEB包
    if command -v dpkg &> /dev/null; then
        log "Creating DEB package..."
        make package || warn "DEB package creation failed"
    fi
    
    # 创建RPM包
    if command -v rpmbuild &> /dev/null; then
        log "Creating RPM package..."
        make package || warn "RPM package creation failed"
    fi
    
    # 创建通用tar.gz
    log "Creating Linux tar.gz package..."
    make package TGZ || warn "TAR.GZ package creation failed"
}

# macOS特定包创建
create_macos_packages() {
    cd "$BUILD_DIR"
    
    log "Creating macOS packages..."
    make package || warn "macOS package creation failed"
}

# Windows特定包创建
create_windows_packages() {
    cd "$BUILD_DIR"
    
    log "Creating Windows packages..."
    make package || warn "Windows package creation failed"
}

# 显示使用帮助
show_help() {
    echo "CSF Descriptor Packaging Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -s, --source   Create source package only"
    echo "  -b, --binary   Create binary packages only"
    echo "  -a, --all      Create all packages (default)"
    echo "  -c, --clean    Clean build directory before packaging"
    echo ""
    echo "Examples:"
    echo "  $0             # Create all packages"
    echo "  $0 -s          # Create source package only"
    echo "  $0 -b          # Create binary packages only"
    echo "  $0 -c          # Clean and create all packages"
}

# 清理构建目录
clean_build() {
    log "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
    rm -rf "$PACKAGE_DIR"
}

# 主函数
main() {
    local action="all"
    local clean=false
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -s|--source)
                action="source"
                shift
                ;;
            -b|--binary)
                action="binary"
                shift
                ;;
            -a|--all)
                action="all"
                shift
                ;;
            -c|--clean)
                clean=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
    
    # 执行操作
    if [ "$clean" = true ]; then
        clean_build
    fi
    
    check_dependencies
    
    case "$action" in
        "source")
            create_source_package
            ;;
        "binary")
            setup_build
            create_binary_packages
            ;;
        "all")
            setup_build
            create_source_package
            create_binary_packages
            ;;
    esac
    
    log "Packaging complete! Check $PACKAGE_DIR for generated packages."
}

# 运行主函数
main "$@"