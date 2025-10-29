# GraspKit

A Python package for data collection and processing of results from GRASP (General-purpose Relativistic Atomic Structure Package). This tool enhances GRASP's built-in data handling capabilities with more flexible Python-based processing, machine learning optimization, and automated workflow management.

## Features

- **ML-driven CSF Selection Pipeline** - Uses machine learning to optimize Configuration State Function selection for quantum mechanical calculations
- **GRASP Integration** - Automated workflow management for GRASP2018 calculations via shell scripts
- **Data Processing** - Comprehensive tools for atomic physics data analysis and visualization
- **Multi-format Support** - Handle various GRASP output formats with specialized loaders

## Requirements

- **Python**: 3.12+
- **Operating System**: Linux, Windows 10+, macOS 10.15+
- **Memory**: 4GB RAM (recommended 8GB+)

## Installation

### 🚀 Method 1: UV (Recommended)

UV is an ultra-fast Python package and project manager that provides extremely fast dependency resolution and installation.

```bash
# Install UV (if not already installed)
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/YenochQin/graspkit-tools.git
cd graspkit-tools

# Create virtual environment
uv venv

# Activate environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies and package (CPU version)
# Note: Tsinghua mirror is automatically configured in pyproject.toml for faster downloads
uv sync

# Or install with GPU support (if you have NVIDIA GPU with CUDA)
uv sync --extra gpu

# For development
uv sync --extra dev
```

**UV Environment Features**:
- Ultra-fast dependency resolution (10-100x faster than pip)
- Automatic Python version management (>=3.12)
- Support for CPU and GPU environment configurations
- Cross-platform support (Linux, Windows, macOS)
- Modern lock file mechanism (uv.lock)
- Isolated development environment
- Fully compatible with pip

### 📦 Method 2: Traditional pip Installation

```bash
# Clone the repository
git clone https://github.com/YenochQin/graspkit-tools.git
cd graspkit-tools

# Create virtual environment
python -m venv grasp_env

# Activate environment
# Windows
grasp_env\Scripts\activate
# macOS/Linux
source grasp_env/bin/activate

# Choose your environment:
pip install -r requirements-cpu.txt    # CPU environment
pip install -r requirements-gpu.txt    # GPU environment (requires CUDA)

# Install the package
pip install -e .
```

### 🔍 Verification

```bash
# Verify installation
python -c "import graspkit; print('✅ Package OK')"

# Check version
python -c "import graspkit; print(graspkit.__version__)"
```

📖 **For detailed installation instructions and troubleshooting**, see [INSTALL.md](INSTALL.md).

## Usage

Examples are provided in the test folder. Simply modify the data file locations, parameters, and `calculation_parameters` to suit your needs.

## Development

### Linting

```bash
# Run Ruff linting
ruff check .

# Auto-fix linting issues
ruff check . --fix
```

For more detailed development instructions, see [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md).

