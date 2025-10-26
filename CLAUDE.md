# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python package for data collection and processing of results from GRASP (General-purpose Relativistic Atomic Structure Package). This tool enhances GRASP's built-in data handling capabilities with more flexible Python-based processing, machine learning optimization, and automated workflow management.

## Core Architecture

### Package Structure
- **graspkit/** - Main Python package (in `src/`)
  - **CSFs_processor/** - Configuration State Function processing and selection
  - **data_IO/** - Data input/output handling, including specialized loaders for different data formats
  - **grasp_data_extractor/** - Data extraction from GRASP2018 calculations (ASF and transition data)
  - **ml_module/** - Machine learning infrastructure (neural networks, training, analysis)
  - **utils/** - Utility functions (environment config, progress management, tool functions)

### Key Components

1. **ML-driven CSF Selection Pipeline** - Uses machine learning to optimize Configuration State Function selection for quantum mechanical calculations
2. **GRASP Integration** - Automated workflow management for GRASP2018 calculations via shell scripts
3. **Data Processing** - Comprehensive tools for atomic physics data analysis and visualization

## Development Commands

### Environment Setup

#### 方法一：使用UV (推荐)
UV 是超快速的Python包和项目管理器，提供极快的依赖解析和安装。

```bash
# 安装 UV (如果尚未安装)
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目并进入目录
git clone https://github.com/YenochQin/graspkit-tools.git
cd graspkit-tools

# 创建虚拟环境
uv venv

# 激活环境
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 安装依赖
uv pip install -e .

# 运行特定命令
uv run python your_script.py
```

**UV 环境特性**：
- 超快的依赖解析和安装（比pip快10-100倍）
- 自动管理Python版本 (>=3.12)
- 支持CPU和GPU两种环境配置
- 跨平台支持 (Linux, Windows, macOS)
- 现代化的锁文件机制 (uv.lock)
- 隔离的开发环境
- 与pip完全兼容

#### 方法二：使用传统pip安装
```bash
# Choose appropriate environment
pip install -r requirements-cpu.txt    # CPU environment
pip install -r requirements-gpu.txt    # GPU environment

# Development installation
pip install -e .

# Build package
python -m build
```

### Linting
```bash
# Run Ruff linting
ruff check .

# Auto-fix linting issues
ruff check . --fix
```

### Common Workflows

#### Package Installation and Verification
```bash
# Install from source
pip install -e .

# Verify installation
python -c "import graspkit; print('✅ Package OK')"

# Check version
python -c "import graspkit; print(graspkit.__version__)"
```

#### Data Processing Workflow
```bash
# Process HDF5 descriptors
python read_hdf5_descriptors.py
```

### Configuration Management

#### Main Package Configuration
- **pyproject.toml** - Modern Python packaging configuration using Hatchling
- **uv.lock** - UV lock file for reproducible dependency management
- **Version management** - Dynamic versioning from `src/graspkit/version.py`
- **Linting** - Ruff with NumPy 2.0 compatibility rules

#### Dependencies
- **UV environment** - Managed environments with ultra-fast dependency resolution
- **requirements-cpu.txt** - CPU-optimized dependencies with PyTorch CPU version (legacy)
- **requirements-gpu.txt** - GPU-enabled dependencies with CUDA support (legacy)

## Important Implementation Notes

### Current Repository Status
- **Version**: 2.7.dev1 (development version)
- **Python Version**: Requires 3.12+
- **Package Manager**: Hatchling (modern Python packaging)
- **No test suite** - Tests mentioned in previous CLAUDE.md are not present in current repository
- **No script directories** - The ml_CSFs_selection_scripts/ directory exists in parent project structure but not in this repository

### Key Dependencies
- **PyTorch >= 2.0.0** - Machine learning framework (CPU/GPU versions)
- **NumPy >= 2.0.0** - Numerical computing
- **Pandas >= 2.2.2** - Data manipulation and analysis
- **Scikit-learn >= 1.3.0** - Traditional machine learning algorithms
- **Matplotlib >= 3.8.4** - Data visualization
- **rtoml >= 0.9.0** - TOML configuration file handling

### Development Patterns

#### Package Import Structure
The package uses a centralized import system in `src/graspkit/__init__.py` that exposes key functionality:
- Data I/O operations (GraspFileLoad, descriptor loading/saving)
- Utility functions (CSFs, energy calculations, transition data)
- Machine learning modules
- Data processing tools

#### Module Organization
- **data_IO/** - Handles all file I/O operations with support for multiple formats
- **ml_module/** - Contains neural network implementations and training logic
- **grasp_data_extractor/** - Specialized tools for extracting data from GRASP calculations
- **utils/** - Shared utilities and data structures
- **CSFs_processor/** - Core CSF processing and selection algorithms

### Build System
- Uses modern Python packaging with `pyproject.toml`
- Dynamic version management from `version.py`
- Ruff linting with NumPy 2.0 compatibility focus
- Pre-built distributions available in `dist/` directory

### Performance Considerations
- PyTorch thread count should be configured based on available CPU cores
- Large CSF datasets require careful memory management
- Multi-block processing available for descriptor generation
- HDF5 format supported for efficient large dataset handling