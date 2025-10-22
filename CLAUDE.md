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

#### 方法一：使用Pixi (推荐)
Pixi 是现代化的包管理器，支持跨平台环境管理和依赖解析。

```bash
# 安装 Pixi (如果尚未安装)
curl -fsSL https://pixi.sh/install.sh | bash

# 克隆项目并进入目录
git clone https://github.com/YenochQin/graspkit-tools.git
cd graspkit-tools

# 安装默认环境 (CPU)
pixi install

# 安装 GPU 环境
pixi install --feature gpu

# 激活环境
pixi shell

# 或者运行特定命令
pixi run python your_script.py
```

**Pixi 环境特性**：
- 自动管理Python版本 (>=3.12)
- 支持CPU和GPU两种环境配置
- 跨平台支持 (Linux, Windows, macOS)
- 自动解决依赖冲突
- 隔离的开发环境

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
- **pixi.toml** - Pixi environment configuration with CPU/GPU features
- **Version management** - Dynamic versioning from `src/graspkit/version.py`
- **Linting** - Ruff with NumPy 2.0 compatibility rules

#### Dependencies
- **Pixi environments** - Managed environments with automatic dependency resolution
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