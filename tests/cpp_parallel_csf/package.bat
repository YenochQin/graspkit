@echo off
setlocal enabledelayedexpansion

REM CSF Descriptor Packaging Script for Windows
echo CSF Descriptor Packaging Script

set PROJECT_DIR=%~dp0
set BUILD_DIR=%PROJECT_DIR%build
set PACKAGE_DIR=%PROJECT_DIR%packages

REM 检查CMake
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] CMake is required but not found in PATH
    exit /b 1
)

REM 创建目录
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%PACKAGE_DIR%" mkdir "%PACKAGE_DIR%"

REM 设置构建环境
cd "%BUILD_DIR%"

REM 配置构建
echo [INFO] Configuring build...
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="%PROGRAMFILES%\csf-descriptor"

REM 编译
echo [INFO] Building...
cmake --build . --config Release

REM 运行测试
echo [INFO] Running tests...
ctest -C Release --output-on-failure

REM 创建安装包
echo [INFO] Creating packages...
cpack -C Release

REM 移动包到packages目录
if exist "*.zip" move "*.zip" "%PACKAGE_DIR%\"
if exist "*.exe" move "*.exe" "%PACKAGE_DIR%\"

echo [INFO] Packaging complete! Check %PACKAGE_DIR% for generated packages.

pause