@echo off
setlocal enabledelayedexpansion

echo === CSF Descriptor Build and Package Script ===
echo.

set SCRIPT_DIR=%~dp0
echo Script Directory: %SCRIPT_DIR%

REM 显示系统信息
echo System Information:
echo   OS: Windows
echo   Architecture: %PROCESSOR_ARCHITECTURE%
where cmake >nul 2>nul && (
    for /f "tokens=*" %%i in ('cmake --version') do (
        echo   CMake: %%i
goto :cmake_done
    )
) || (
    echo [ERROR] CMake not found in PATH
    exit /b 1
)
:cmake_done
echo.

REM 步骤1: 清理并创建构建目录
echo Step 1: Preparing build environment...
if exist build rmdir /s /q build
if exist packages rmdir /s /q packages
mkdir build
mkdir packages

REM 步骤2: 配置构建
echo Step 2: Configuring build...
cd build
cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release

if %errorlevel% neq 0 (
    echo [ERROR] CMake configuration failed
    exit /b 1
)

REM 步骤3: 编译
echo Step 3: Building...
cmake --build . --config Release

if %errorlevel% neq 0 (
    echo [ERROR] Build failed
    exit /b 1
)

REM 步骤4: 运行测试
echo Step 4: Running tests...
ctest -C Release --output-on-failure

REM 步骤5: 创建安装包
echo Step 5: Creating packages...
cpack -C Release

REM 步骤6: 显示生成的包
echo Step 6: Generated packages:
dir *.zip *.exe 2>nul

REM 步骤7: 移动到packages目录
move *.zip ..\packages\ 2>nul
move *.exe ..\packages\ 2>nul

REM 步骤8: 显示结果
echo.
echo === Build Complete ===
echo Packages created in: %SCRIPT_DIR%packages\
echo.
echo To install:
echo   Run the .exe installer in packages directory
echo.
echo To test locally:
echo   cd build
echo   ctest -C Release --output-on-failure

echo.
pause