# Repository Guidelines

## Project Structure & Module Organization
- Core package code lives in `src/graspkit/`.
- Main modules are organized by responsibility: `data_IO/` (loaders/writers), `CSFs_processor/` (physics/CSF processing), `ml_module/` (training/inference), `grasp_data_extractor/`, and shared helpers in `utils/`.
- Tests are split across `test/` (legacy script-style checks such as `test/test.py`) and `tests/` (pytest-style tests, fixtures in `tests/fixtures/`).
- Supporting docs and change notes are in `docs/` and `modify_logs/`.

## Build, Test, and Development Commands
- `uv venv && source .venv/bin/activate`: create/activate local environment.
- `uv sync --extra cpu --extra dev`: install dependencies for CPU development.
- `uv sync --extra gpu --extra dev`: install GPU variant (CUDA-capable systems).
- `python -m build` or `python build_package.py --clean`: build distribution artifacts.
- `ruff check .` and `ruff check . --fix`: lint and auto-fix style/issues.
- `mypy src/`: strict type checking.
- `python test/test.py` or `pytest tests/`: run test suites.

## Coding Style & Naming Conventions
- Use Python with 4-space indentation and explicit type hints on public functions.
- Follow naming: `snake_case` (functions/variables), `PascalCase` (classes), `UPPER_SNAKE_CASE` (constants).
- Keep imports grouped: standard library, third-party, local modules.
- Prefer `pathlib.Path` for file paths.
- Use Ruff for linting and formatting compatibility (`NPY201` rules enabled).

## Testing Guidelines
- Add tests under `tests/` for new functionality; keep fixtures under `tests/fixtures/`.
- Name test files `test_*.py` and test functions `test_*` for pytest discovery.
- For parser/data-loader changes, include realistic sample inputs and edge cases.

## Commit & Pull Request Guidelines
- Recent history shows short commit messages (`update`, `bug fixed`), but contributors should use clear, scoped subjects, e.g. `data_IO: fix binary loader index bounds`.
- Keep commits focused and atomic.
- PRs should include: purpose, key changes, how to test, and linked issue (if available).
- Include before/after output snippets for behavior changes in loaders, processing, or ML workflows.

## Security & Configuration Tips
- Do not commit secrets, local datasets, or generated artifacts.
- Use virtual environments; align with project requirement in `pyproject.toml` (currently Python `>=3.14`).
