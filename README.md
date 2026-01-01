# GraspKit

**Version**: 2.9.1 | **Python**: 3.13

English | [简体中文](README_zh.md)

A Python package for data collection and processing of results from GRASP (General-purpose Relativistic Atomic Structure Package). This tool enhances GRASP's built-in data handling capabilities with more flexible Python-based processing, machine learning optimization, and automated workflow management.

## Overview

GraspKit provides a comprehensive toolkit for atomic physics researchers working with GRASP2018 calculations. It automates the extraction, processing, and analysis of atomic structure data, with a focus on ML-driven optimization of Configuration State Function (CSF) selection for large-scale MCDHF calculations.

## Key Features

- **ML-driven CSF Selection Pipeline** - Uses neural networks to intelligently select CSFs for quantum mechanical calculations, reducing computational cost while maintaining accuracy
- **GRASP Integration** - Automated workflow management for GRASP2018 calculations via shell scripts
- **Multi-format Data Processing** - Comprehensive tools for atomic physics data analysis supporting binary, Parquet, HDF5, and text formats
- **HPC-Ready** - Native SLURM environment detection with automatic progress bar hiding and environment-aware logging
- **GPU Acceleration** - CUDA support for PyTorch models with automatic device selection
- **C++ Integration** - Fast descriptor generation via C++ extensions

## Installation

GraspKit supports multiple installation methods. Choose the one that best fits your workflow.

### Method 1: UV (Recommended)

UV is an ultra-fast Python package manager with 10-100x faster dependency resolution than pip.

```bash
# Install UV (if not already installed)
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/YenochQin/graspkit-tools.git
cd graspkit-tools

# Create and activate virtual environment
uv venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# Install dependencies - choose CPU or GPU version
# CPU version (recommended for compatibility)
uv sync --extra cpu

# GPU version (NVIDIA CUDA required)
uv sync --extra gpu

# Development environment with testing tools
uv sync --extra dev --extra cpu    # CPU
uv sync --extra dev --extra gpu    # GPU
```

### Method 2: Pixi (Alternative)

Pixi is a cross-platform package manager that uses conda-forge for dependencies.

```bash
# Install Pixi (if not already installed)
# Windows (PowerShell)
powershell -c "irm https://pixi.sh/install.ps1 | iex"
# macOS/Linux
curl -fsSL https://pixi.sh/install.sh | bash

# Clone the repository
git clone https://github.com/YenochQin/graspkit-tools.git
cd graspkit-tools

# Install dependencies and activate environment
pixi install
pixi shell

# Pixi automatically configures the dev-gpu environment by default
# To switch environments:
pixi shell -e cpu     # CPU only
pixi shell -e gpu     # GPU enabled
pixi shell -e dev-gpu # Development with GPU
```

**Pixi Features**:
- Conda-forge based package management
- Cross-platform support (Linux, macOS, Windows)
- Lock file based reproducibility (pixi.lock)
- Automatic environment switching
- Integration with existing conda workflows

### Requirements

- **Python**: 3.13 (exclusive)
- **Operating System**: Linux, Windows 10+, macOS 10.15+
- **Memory**: 4GB RAM (recommended 8GB+ for large datasets)
- **GPU (optional)**: NVIDIA GPU with CUDA support for accelerated ML training

### Verification

```bash
# Verify installation
python -c "import graspkit; print('Package OK')"

# Check version
python -c "import graspkit; print(graspkit.__version__)"
```

For detailed installation instructions and troubleshooting, see [INSTALL.md](INSTALL.md).

## Package Architecture

```
graspkit/
+-- CSFs_processor/          # CSF selection and processing algorithms
|   +-- CSFs_choosing.py     # Random/threshold-based CSF selection
|   +-- CSFs_compress_extract.py  # Descriptor generation from CSFs
|
+-- data_IO/                 # Data input/output handling
|   +-- GraspFileLoad        # Main loader for GRASP output files
|   +-- EnergyFile2csv       # Energy file conversion
|   +-- data_writer.py       # Save processed data (pickle/parquet/HDF5)
|   +-- data_loader.py       # Load processed data
|
+-- grasp_data_extractor/    # Physical quantity extraction
|   +-- ASF_data_collection.py      # Energy levels, ASF composition
|   +-- transition_data_collection.py  # Transition rate data
|   +-- transition_data_analyzer.py    # Transition analysis
|
+-- ml_module/               # Machine learning pipeline
|   +-- neural_network.py    # ANN and TensorNet architectures
|   +-- ml_initializer.py    # Training data setup and validation
|   +-- ml_trainer.py        # Model training and evaluation
|   +-- ml_results_analyzer.py  # Results analysis and CSF selection
|
+-- utils/                   # Utility functions
    +-- data_modules.py      # Data structures (MixCoefficientData, CSFs)
    +-- environment_config.py  # HPC/SLURM detection
    +-- progress_manager.py  # Environment-aware progress bars
    +-- plot_functions.py    # Visualization utilities
```

## Quick Start

```python
import graspkit as gk

# 1. Load GRASP calculation results
loader = gk.GraspFileLoad("path/to/grasp/output")
energy_data = gk.mcdhf_energy_data_collection(loader)

# 2. Process CSFs with ML-driven selection
# Generate descriptors from CSF configurations
csf_descriptors = gk.batch_process_csfs_to_descriptors(csfs_data)

# 3. Train ML model for CSF prediction
model = gk.ANNClassifier(
    input_dim=len(csf_descriptors[0]),
    hidden_layers=[256, 128, 64],
    num_classes=2
)
gk.train_model(model, training_data, labels)

# 4. Select CSFs for next iteration
selected_csfs = gk.select_csfs_for_coverage(
    model,
    candidate_csfs,
    coverage_threshold=0.95
)

# 5. Check convergence
converged = gk.evaluate_calculation_convergence(
    current_energy,
    previous_energy
)
```

## Usage Examples

The `tests/` directory contains example scripts:

```bash
# ML classifier example
python tests/ANN.py

# Wavefunction visualization
python tests/rwfn_plotter.py

# Visualization examples
python tests/Nightingale_rose.py

# Run test coverage
python tests/test_coverage_simple.py
python tests/test_coverage_function.py
```

## Development

### Linting and Formatting

```bash
# Run Ruff linting (NumPy 2.0 compatible)
ruff check .

# Auto-fix linting issues
ruff check . --fix

# Type checking (optional)
mypy src/
```

### Building the Package

```bash
# Clean build
python build_package.py --clean

# Development build with all dependencies
python build_package.py --dev --clean
```

### Testing

```bash
# Run tests
pytest tests/

# Run specific test file
python tests/test_coverage_simple.py
```

For more detailed development instructions, see [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md).

## Main Workflow

1. **Data Loading** - Load GRASP output files (energy levels, mixing coefficients, CSFs)
2. **Descriptor Generation** - Convert CSFs to numerical descriptors via `parse_csf_2_descriptor()`
3. **ML Training** - Train neural networks to predict important CSFs using `train_model()`
4. **CSF Selection** - Use ML predictions to select CSFs for next iteration
5. **Convergence Check** - Monitor energy and CSF count via `evaluate_calculation_convergence()`
6. **Iteration** - Repeat until convergence criteria are met

## Environment-Specific Behavior

GraspKit automatically detects HPC/SLURM environments:
- Progress bars are hidden in SLURM jobs
- Debug mode enables verbose logging
- GPU support with automatic CUDA device selection

```python
# Check current environment
from graspkit.utils.environment_config import is_slurm_job, is_debug_mode
print(f"SLURM job: {is_slurm_job()}")
print(f"Debug mode: {is_debug_mode()}")
```

## Project Links

- **Homepage**: https://github.com/YenochQin/graspkit
- **Documentation**: [INSTALL.md](INSTALL.md), [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)
- **License**: See LICENSE file

## Citation

If you use GraspKit in your research, please cite:

```bibtex
@software{graspkit2024,
  author = {Qin, Yi},
  title = {GraspKit: ML-driven CSF Selection for GRASP Calculations},
  version = {2.9.1},
  year = {2024},
  url = {https://github.com/YenochQin/graspkit}
}
```

## License

This project is licensed under the terms specified in the LICENSE file.
