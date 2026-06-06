@echo off
REM graspkit Package Build Script for Windows
REM This script builds the graspkit package and moves the generated packages
REM to ../graspkit-tools/package directory.
REM Works with UV, Pixi, or traditional pip environments.

echo ============================================================
echo graspkit Package Build Script
echo ============================================================
echo.

REM Check which environment manager is being used
if exist "uv.lock" (
    echo Detected UV environment
    if not exist ".venv\Scripts\python.exe" (
        echo Error: UV virtual environment not found in .venv folder
        echo Please set up the environment first using:
        echo   uv venv
        echo   .venv\Scripts\activate
        echo   uv pip install -e .
        pause
        exit /b 1
    )
    set PYTHON_CMD=.venv\Scripts\python.exe
) else if exist "pixi.toml" (
    echo Detected Pixi environment
    set PYTHON_CMD=pixi run python
) else if exist ".venv\Scripts\python.exe" (
    echo Detected traditional pip environment
    set PYTHON_CMD=.venv\Scripts\python.exe
) else (
    echo Error: No supported virtual environment found
    echo Please set up an environment first using one of:
    echo   UV: uv venv && .venv\Scripts\activate && uv pip install -e .
    echo   Pixi: pixi install && pixi shell
    echo   Pip: python -m venv .venv && .venv\Scripts\activate && pip install -e .
    pause
    exit /b 1
)

echo Using Python command: %PYTHON_CMD%
echo.

REM Run the Python build script
%PYTHON_CMD% build_package.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Build failed! Check the error messages above.
    pause
    exit /b 1
)

echo.
echo Build completed successfully!
echo Packages are available in: ..\graspkit-tools\package
echo.
pause