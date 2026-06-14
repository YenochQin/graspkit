# Repository Guidelines

## Project Structure & Module Organization
`graspkit` is a Python package for collecting and processing GRASP (General-purpose Relativistic Atomic Structure Package) results. It provides Python-based data handling, machine learning optimization, plotting, and workflow support for atomic-structure calculations.

Core package code lives under `src/` and is split across several packages:

- `src/graspkit/`: main package, including `data_IO/`, `CSFs_processor/`, `grasp_data_extractor/`, `ml_module/`, and `utils/`.
- `src/graspkit_config/`: Pydantic configuration models.
- `src/graspkit_ml/`: ML utilities.
- `src/graspkit_plot/`: plotting utilities.

`data_IO` is the foundation for loading and persistence. `CSFs_processor` contains core quantum-mechanics and CSF-selection algorithms. `ml_module` depends on processed data from `CSFs_processor`. `grasp_data_extractor` handles GRASP-specific ASF and transition formats. `utils` provides shared helpers such as environment detection, structured logging, progress management, and tool functions.

Tests are split across `test/` for legacy script-style checks and `tests/` for pytest-style tests and fixtures. Supporting docs and change notes live in `docs/` and `modify_logs/`.

## Build, Test, and Development Commands
This repository can be tested directly, but day-to-day pipeline debugging from the full workspace should usually use the `graspkit-tools/.venv` environment because Tools installs this package editable and includes `rcsfs` plus pipeline dependencies.

```bash
uv venv
source .venv/bin/activate
uv sync --extra cpu --extra dev
```

Use `uv sync --extra gpu --extra dev` on CUDA-capable systems that need GPU PyTorch wheels. Windows activation is `.venv\Scripts\activate`.

- `uv pip install -e .`: legacy editable installation.
- `python build_package.py --clean`: clean cross-platform package build.
- `./build_package.sh --clean`: Unix-like package build wrapper.
- `build_package.bat --clean`: Windows package build wrapper.
- `python -m build`: manual package build.
- `python build_package.py --dev --clean`: development build with all dependencies.
- `ruff check .`: run Ruff with NumPy 2.0 compatibility rules.
- `ruff check . --fix`: auto-fix supported Ruff issues.
- `mypy src/`: type-check the package.
- `pytest tests/`: run pytest tests.
- `python test/test.py`: run legacy script-style checks when relevant.
- `python -c "import graspkit; print('Package OK')"`: verify installation.
- `python -c "import graspkit; print(graspkit.__version__)"`: check the installed version.

## Architecture & Development Patterns
The package exposes core functionality through `src/graspkit/__init__.py` with lazy imports, including `MLCalConfig`, `CalPath`, and `load_config`. Use subpackage imports such as `graspkit.ml_module` or `graspkit.CSFs_processor` for deeper access.

The codebase includes HPC/SLURM-aware logging and progress behavior. `utils/environment_config.py` detects SLURM jobs, debug mode, and progress-display settings. Prefer the existing `log_stage_start()` and `log_stage_end()` helpers for structured logging.

Data structures commonly use `@dataclass`, including types such as `MixCoefficientData` and `CSFs` in `utils/data_modules.py`. PyTorch models should keep type annotations on public paths.

Configuration is managed through `pyproject.toml` using Hatchling and uv extras. CPU and GPU dependency sets select PyTorch variants (`torch==2.10.0`), while dev tooling includes pytest, ruff, mypy, black, and Jupyter-related tools. Current repository status from `src/graspkit/version.py` is `3.2.dev2`, and Python support requires `>=3.14`.

## Coding Style & Naming Conventions
Use Python with 4-space indentation and explicit type hints on public functions. Prefer `pathlib.Path` for file paths. Follow `snake_case` for modules, functions, and variables; `PascalCase` for classes and Pydantic models; and `UPPER_SNAKE_CASE` for constants. Keep imports grouped as standard library, third-party, then local modules. Ruff has `NPY201` rules enabled.

## Testing Guidelines
Add tests under `tests/` for new functionality and keep fixtures under `tests/fixtures/`. Name test files `test_*.py` and test functions `test_*` for pytest discovery. For parser, loader, CSF-processing, or ML changes, include realistic sample inputs and edge cases.

Run the relevant checks before submitting changes:

```bash
pytest tests/
ruff check .
mypy src/
```

For environment-specific issues, verify the active PyTorch variant with:

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Common Workflows
A typical processing workflow imports the public package, loads configuration, and reaches into subpackages only when needed:

```python
import graspkit as gk

config = gk.load_config("path/to/config.toml")
ml_config = gk.MLCalConfig(...)

from graspkit.CSFs_processor import ...
from graspkit.ml_module import ANNClassifier
```

When changing code used by `graspkit-tools`, remember that the Tools uv environment sees `graspkit/src/...` immediately through the editable install. Rebuild this package only when validating package artifacts or distribution behavior.

## Commit & Pull Request Guidelines
Recent history includes short messages such as `update` and `bug fixed`, but prefer clear scoped subjects such as `data_IO: fix binary loader index bounds`. Keep commits focused and atomic. PRs should include purpose, key changes, test commands, linked issues if available, and before/after output snippets for behavior changes in loaders, processing, or ML workflows.

## Security & Configuration Tips
Do not commit secrets, local datasets, generated artifacts, virtual environments, or cluster-specific paths. Use virtual environments and align with the Python requirement in `pyproject.toml` (`>=3.14`). Treat GRASP input/output paths as environment-specific unless they are stable fixtures.
