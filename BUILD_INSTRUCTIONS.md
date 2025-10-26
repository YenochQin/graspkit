# GraspKit Package Build Instructions

This document provides instructions for building the GraspKit package.

## Prerequisites

1. **Python 3.12+** - Required for the package
2. **Virtual Environment** - Set up using UV (recommended), Pixi, or pip
3. **Build Tools** - The build script will automatically install required build dependencies

## Environment Setup

### Method 1: UV (Recommended)

UV is an ultra-fast Python package and project manager.

```bash
# Install UV
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### Method 2: Pixi

```bash
# Install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Install dependencies
pixi install

# Activate environment
pixi shell
```

### Method 3: Traditional pip

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -e .
```

## Quick Start

### Method 1: Using the Python Script (Recommended)

```bash
# Activate your environment (if using UV, Pixi, or pip)
# UV: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Linux/macOS)
# Pixi: pixi shell
# pip: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Linux/macOS)

# Build the package (clean build)
python build_package.py --clean

# Or build without cleaning
python build_package.py

# Build development version
python build_package.py --dev --clean
```

### Method 2: Using the Batch File (Windows)

```cmd
# Double-click or run from command line
build_package.bat

# Or with options
build_package.bat --clean
```

### Method 3: Using the Shell Script (Unix-like systems)

```bash
# Run from terminal
./build_package.sh

# Or with options
./build_package.sh --clean
```

### Method 4: Manual Build

```bash
# Install build dependencies
pip install build

# Build the package
python -m build

# Move packages manually to ../Graspkit-tools/package
```

## Script Options

- `--clean`: Clean previous build artifacts before building
- `--dev`: Build development version (currently no effect, but reserved for future use)

## Output

The script will generate two types of packages:

1. **Wheel File** (`.whl`) - Binary distribution for easy installation
2. **Source Distribution** (`.tar.gz`) - Source code distribution

Both packages will be automatically moved to `../Graspkit-tools/package/`.

## Installation

After building, you can install the package using:

```bash
# Install from wheel (recommended)
pip install ../Graspkit-tools/package/grasp_kit-*.whl

# Or install from source distribution
pip install ../Graspkit-tools/package/grasp_kit-*.tar.gz
```

## Troubleshooting

### Build Fails
- Ensure you have Python 3.12+ installed
- Check that your virtual environment is activated
- Run with `--clean` flag to remove old artifacts
- The script automatically detects UV, Pixi, or pip environments

### Environment Detection Issues
- UV: Make sure `uv.lock` file exists and `.venv` folder is present
- Pixi: Ensure `pixi.toml` exists and `pixi` command is available
- pip: Verify `.venv` folder exists with `Scripts/python.exe` (Windows) or `bin/python` (Linux/macOS)

### Permission Issues
- Make sure you have write permissions to both the current directory and `../Graspkit-tools/package/`
- On Windows, run the command prompt as Administrator if needed

### Missing Dependencies
- The script will automatically install the `build` package if not present
- If other dependencies are missing, install them manually:
  - UV: `uv pip install build`
  - Pixi: `pixi add build`
  - pip: `pip install build`

## Environment Managers

### UV Features
- Ultra-fast dependency resolution (10-100x faster than pip)
- Modern lock file mechanism (`uv.lock`)
- Cross-platform compatibility
- Seamless pip compatibility

### Pixi Features
- Conda-compatible environment management
- Multiple environment configurations (CPU/GPU)
- Dependency isolation
- Cross-platform support

### Traditional pip
- Universal compatibility
- Simple and familiar
- Works with existing workflows

## Package Information

- **Package Name**: grasp-kit
- **Current Version**: 2.8.dev1
- **Python Version**: >= 3.12
- **Build Backend**: hatchling
- **Supported Environments**: UV, Pixi, pip

For more information about the package configuration, see `pyproject.toml`.