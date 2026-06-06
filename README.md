# graspkit

English | [简体中文](README_zh.md)

graspkit is a Python toolkit for reading, restructuring, and analyzing output from GRASP atomic-structure calculations. The package focuses on three practical workflows:

- loading GRASP text and binary files into typed Python objects or Polars DataFrames
- converting CSFs into descriptor matrices for downstream analysis and machine learning
- running iterative, config-driven CSF screening with PyTorch models

## What Is In The Package

`src/graspkit/` is organized around the workflows exposed from `graspkit.__init__`:

- `data_IO/`: typed loaders for `.level`, `.lsj.lbl`, `.c`, `.ct`/`.t`, mixing-coefficient binaries, descriptor/config persistence, and the `MLCalConfig` model
- `grasp_data_extractor/`: energy-table formatting, iterative level comparison, LSJ composition merging, transition post-processing
- `CSFs_processor/`: CSF parsing, descriptor generation, and selection helpers based on CI weights or simple sampling
- `ml_module/`: `ANNClassifier`, `ANNRegressor`, training/evaluation helpers, iterative screening utilities
- `utils/`: shared dataclasses, environment detection, plotting helpers, shell/quantum-number utilities

## Logical Packages

Stage 3 of the package-splitting plan keeps the repository unified but settles on `graspkit` itself as the core package entry point:

- `graspkit_config`: lightweight Pydantic configuration models
- `graspkit`: data loaders, CSF processing, result extraction, and lightweight utilities
- `graspkit_ml`: machine-learning training, inference, and iterative screening helpers
- `graspkit_plot`: optional plotting and matplotlib styling helpers

Recommended imports for new code:

```python
from graspkit_config import MLCalConfig
from graspkit.data_IO import load_config
from graspkit.grasp_data_extractor import format_energy_configurations
from graspkit_ml import train_model, evaluate_model
from graspkit_plot import configure_matplotlib_for_publication
```

Legacy flat root exports for ML and plotting are no longer part of the supported API. New code should treat `graspkit` as the core layer and import ML helpers from `graspkit_ml`.

## Installation

The project currently requires Python `>=3.14`.

### UV

```bash
uv venv
source .venv/bin/activate

# runtime install
uv sync --extra cpu

# development install
uv sync --extra cpu --extra dev

# CUDA-capable systems
uv sync --extra gpu
uv sync --extra gpu --extra dev
```

### Pixi

```bash
pixi install
pixi shell

# optional environment variants
pixi shell -e cpu
pixi shell -e gpu
pixi shell -e dev-gpu
```

### Verify The Import

```bash
python -c "import graspkit; print(graspkit.__version__)"
```

See [INSTALL.md](INSTALL.md) for the longer installation guide.


## Development

Common commands for local development:

```bash
ruff check .
ruff check . --fix
mypy src/
pytest tests/
python test/test.py
python -m build
```

## Notes On Current Scope

- `graspkit` is the core package; `graspkit_ml` and `graspkit_plot` are the layered extensions.
- DataFrames are built with Polars; descriptor and model data use NumPy.
- PyTorch is optional at install time and selected through the `cpu` or `gpu` extras.
- SLURM/debug environment detection lives in `graspkit.utils.environment_config`.

## Repository References

- [INSTALL.md](INSTALL.md)
- [pyproject.toml](pyproject.toml)
- [src/graspkit](src/graspkit)
