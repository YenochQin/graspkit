# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python package for data collection and processing of results from GRASP (General-purpose Relativistic Atomic Structure Package). This tool enhances GRASP's built-in data handling capabilities with more flexible Python-based processing, machine learning optimization, and automated workflow management.

## Core Architecture

### Package Structure
- **graspkit/** - Main Python package (in `src/`)
  - **CSFs_processor/** - Configuration State Function processing and selection algorithms
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

#### UV Package Manager (Recommended)
```bash
# Create and activate environment
uv venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies (must choose CPU or GPU)
uv sync --extra cpu --extra dev    # CPU version with dev tools
uv sync --extra gpu --extra dev     # GPU version with dev tools (NVIDIA CUDA)

# Alternative legacy installation
uv pip install -e .
```

#### Traditional pip Installation
```bash
# Create and activate environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -e .  # Uses CPU dependencies by default
```

### Building and Quality

#### Package Building
```bash
# Build the package (cross-platform scripts)
python build_package.py --clean    # Clean build
./build_package.sh --clean         # Unix-like systems
build_package.bat --clean          # Windows

# Manual build
python -m build

# Development build with all dependencies
python build_package.py --dev --clean
```

#### Linting and Quality
```bash
# Run Ruff linting with NumPy 2.0 compatibility rules
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Type checking (optional)
mypy src/
```

#### Testing
```bash
# Run test files in tests/ directory
python tests/test_coverage_simple.py
python tests/test_coverage_function.py

# Run example scripts
python tests/ANN.py                    # ML classifier example
python tests/rwfn_plotter.py          # Wavefunction plotting
python tests/Nightingale_rose.py      # Visualization example

# Run notebook examples
jupyter notebook tests/test.ipynb
```

#### Package Verification
```bash
# Verify installation
python -c "import graspkit; print('Package OK')"

# Check version
python -c "import graspkit; print(graspkit.__version__)"
```

### Key Development Patterns

#### Centralized Import Structure
The package exposes all functionality through `src/graspkit/__init__.py` with comprehensive imports:
- Data I/O operations (GraspFileLoad, descriptor loading/saving)
- Utility functions (CSFs, energy calculations, transition data)
- Machine learning modules (ANNClassifier, training functions)
- Data processing tools (ASF composition, transition analysis)

#### Module Interdependencies
- **data_IO** serves as foundation, providing file loading and persistence
- **CSFs_processor** contains core quantum mechanics algorithms
- **ml_module** depends on processed data from CSFs_processor
- **grasp_data_extractor** handles GRASP-specific data formats
- **utils** provides shared functionality across all modules

#### Environment-Aware Logging
The codebase includes environment detection for HPC/SLURM environments:
- `utils/environment_config.py` - Detects SLURM jobs and debug mode
- `utils/progress_manager.py` - Hides progress bars in SLURM, shows in debug mode
- Use `log_stage_start()` / `log_stage_end()` functions for structured logging

#### Type Safety
- Data structures use `@dataclass` (e.g., `MixCoefficientData`, `CSFs` in `utils/data_modules.py`)
- Type-safe helper methods on `GraspFileLoad` class
- PyTorch models use type annotations

### Configuration Management

#### Main Package Configuration
- **pyproject.toml** - Modern Python packaging configuration using Hatchling
- **uv.lock** - UV lock file for reproducible dependency management
- **UV Environment** - Supports CPU/GPU optional dependencies via `--extra cpu` or `--extra gpu`
- **Ruff Configuration** - NumPy 2.0 compatibility rules in pyproject.toml

#### Dependencies Management
- **CPU Environment** - PyTorch CPU version for general compatibility
- **GPU Environment** - PyTorch CUDA version (cu128) for NVIDIA GPUs
- **Development Tools** - pytest, ruff, mypy, black, jupyter ecosystem
- **Data Processing** - pandas, numpy, matplotlib, h5py, polars, pyarrow

### Current Repository Status
- **Version**: 2.9dev2 (from `src/graspkit/version.py`)
- **Python Version**: Requires 3.13+ (<3.14)
- **Package Manager**: Hatchling with UV support
- **Build System**: Modern packaging with optional dependencies
- **Testing**: Example files in tests/ directory (ANN.py, rwfn_plotter.py, Nightingale_rose.py, test.ipynb)

## Common Development Workflows

### Data Processing Pipeline
```python
# Typical workflow for processing GRASP data
import graspkit as gk

# Load GRASP calculation results
data_loader = gk.GraspFileLoad("path/to/grasp/output")
energy_data = gk.iterative_levels_collection(data_loader)

# Process CSFs with ML-driven selection
selected_csfs = gk.radom_choose_csfs(csf_processor, n_select=1000)

# Train ML model for optimization
model = gk.ANNClassifier(...)
gk.train_model(model, training_data)
```

### Environment-Specific Commands
```bash
# Check which PyTorch environment is active
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Switch between CPU/GPU environments
uv sync --extra cpu --extra dev    # CPU with dev tools
uv sync --extra gpu --extra dev     # GPU with dev tools
```
