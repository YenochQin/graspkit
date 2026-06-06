#!/bin/bash
# graspkit Package Build Script for Unix-like systems
# This script builds the graspkit package and moves the generated packages
# to ../graspkit-tools/package directory.
# Works with UV, Pixi, or traditional pip environments.

echo "============================================================"
echo "graspkit Package Build Script"
echo "============================================================"
echo

# Check which environment manager is being used
if [ -f "uv.lock" ]; then
    echo "Detected UV environment"
    if [ ! -f ".venv/bin/python" ]; then
        echo "Error: UV virtual environment not found in .venv folder"
        echo "Please set up the environment first using:"
        echo "  uv venv"
        echo "  source .venv/bin/activate"
        echo "  uv pip install -e ."
        exit 1
    fi
    PYTHON_CMD=".venv/bin/python"
elif [ -f "pixi.toml" ]; then
    echo "Detected Pixi environment"
    PYTHON_CMD="pixi run python"
elif [ -f ".venv/bin/python" ]; then
    echo "Detected traditional pip environment"
    PYTHON_CMD=".venv/bin/python"
else
    echo "Error: No supported virtual environment found"
    echo "Please set up an environment first using one of:"
    echo "  UV: uv venv && source .venv/bin/activate && uv pip install -e ."
    echo "  Pixi: pixi install && pixi shell"
    echo "  Pip: python -m venv .venv && source .venv/bin/activate && pip install -e ."
    exit 1
fi

echo "Using Python command: $PYTHON_CMD"
echo

# Run the Python build script
$PYTHON_CMD build_package.py "$@"

if [ $? -ne 0 ]; then
    echo
    echo "Build failed! Check the error messages above."
    exit 1
fi

echo
echo "Build completed successfully!"
echo "Packages are available in: ../graspkit-tools/package"
echo