# Repository Guidelines

## Project Structure & Module Organization
`graspkit` is the stable Python foundation for collecting, processing, and plotting GRASP (General-purpose Relativistic Atomic Structure Package) results. ML training and pipeline orchestration belong to the sibling `graspkit-tools` repository.

Core package code lives under `src/` and is split across several packages:

- `src/graspkit/`: main package, including `data_IO/`, `CSFs_processor/`, `grasp_data_extractor/`, and `utils/`.
- `src/graspkit_plot/`: plotting utilities.

`data_IO` is the foundation for GRASP loading and stable CSF persistence. `CSFs_processor` contains core quantum-mechanics and CSF-selection algorithms. `grasp_data_extractor` handles GRASP-specific ASF and transition formats. `utils` provides shared data models, environment detection, and tool functions.

Tests are split across `test/` for legacy script-style checks and `tests/` for pytest-style tests and fixtures. Supporting docs and change notes live in `docs/` and `modify_logs/`.

## Build, Test, and Development Commands
All Python work in this repository must use `../graspkit-tools/.venv`, the workspace's single environment created and synchronized by running `uv sync` in `graspkit-tools/`. Do not run `uv venv` or `uv sync` here, and do not create, activate, or use `graspkit/.venv`. Tools installs this package editable and includes `rcsfs` plus the shared pipeline and development dependencies.

```bash
cd ../graspkit-tools
uv sync
source .venv/bin/activate
cd ../graspkit
```

Keep the shared environment activated while working here; for a single non-interactive Python command, call `../graspkit-tools/.venv/bin/python` directly. Do not use `uv run` from this repository because uv may select or create a local project environment. On Windows activate `..\graspkit-tools\.venv\Scripts\activate`. GPU, PyTorch, and all other Python dependencies are owned by `graspkit-tools`, not this repository.

- Do not run a local editable install; `uv sync` in `graspkit-tools/` installs this repository editable.
- `python build_package.py --clean`: clean cross-platform package build.
- `./build_package.sh --clean`: Unix-like package build wrapper.
- `build_package.bat --clean`: Windows package build wrapper.
- `python -m build`: manual package build.
- `python build_package.py --dev --clean`: development build with all dependencies.
- `ruff check .`: run Ruff with NumPy 2.0 compatibility rules.
- `ruff check . --fix`: auto-fix supported Ruff issues.
- `basedpyright src/`: type-check the package.
- `pytest tests/`: run pytest tests.
- `python test/test.py`: run legacy script-style checks when relevant.
- `python -c "import graspkit; print('Package OK')"`: verify installation.
- `python -c "import graspkit; print(graspkit.__version__)"`: check the installed version.

## Architecture & Development Patterns
The package root exposes only metadata. Use explicit imports such as `graspkit.data_IO`, `graspkit.grasp_data_extractor`, `graspkit.CSFs_processor`, and `graspkit_plot`.

The codebase includes HPC/SLURM-aware logging and progress behavior. `utils/environment_config.py` detects SLURM jobs, debug mode, and progress-display settings. Prefer the existing `log_stage_start()` and `log_stage_end()` helpers for structured logging.

Data structures commonly use `@dataclass`, including types such as `MixCoefficientData` and `CSFs` in `utils/data_modules.py`.

Packaging is managed through `pyproject.toml` using Hatchling. Current repository status from `src/graspkit/version.py` is `3.4.dev1`, and Python support requires `>=3.14`.

## Coding Style & Naming Conventions
Use Python with 4-space indentation and explicit type hints on public functions. Prefer `pathlib.Path` for file paths. Follow `snake_case` for modules, functions, and variables; `PascalCase` for classes and Pydantic models; and `UPPER_SNAKE_CASE` for constants. Keep imports grouped as standard library, third-party, then local modules. Ruff has `NPY201` rules enabled.

## Testing Guidelines
Add tests under `tests/` for new functionality and keep fixtures under `tests/fixtures/`. Name test files `test_*.py` and test functions `test_*` for pytest discovery. For parser, loader, or CSF-processing changes, include realistic sample inputs and edge cases.

Run the relevant checks before submitting changes:

```bash
pytest tests/
ruff check .
basedpyright src/
```

## Common Workflows
A typical processing workflow uses explicit stable subpackage imports:

```python
from graspkit.data_IO import EnergyFileLoader
from graspkit.grasp_data_extractor import format_energy_configurations
from graspkit.CSFs_processor import select_csf_indices_by_ci_squared_cutoff
from graspkit_plot import configure_matplotlib_for_publication
```

When changing code used by `graspkit-tools`, remember that the Tools uv environment sees `graspkit/src/...` immediately through the editable install. Rebuild this package only when validating package artifacts or distribution behavior.

## Commit & Pull Request Guidelines
Recent history includes short messages such as `update` and `bug fixed`, but prefer clear scoped subjects such as `data_IO: fix binary loader index bounds`. Keep commits focused and atomic. PRs should include purpose, key changes, test commands, linked issues if available, and before/after output snippets for behavior changes in loaders or processing workflows.

## Security & Configuration Tips
Do not commit secrets, local datasets, generated artifacts, virtual environments, or cluster-specific paths. Use only the shared `graspkit-tools/.venv` and align with the Python requirement in `pyproject.toml` (`>=3.14`). Treat GRASP input/output paths as environment-specific unless they are stable fixtures.
