# GraspKit Guidelines

## Ownership and interfaces

`graspkit-atomic` is the stable GRASP processing library. The wheel contains `src/graspkit/` and `src/graspkit_plot/`; tests are under `tests/` and documentation under `docs/`.

- Use `data_IO/` for loaders and stable CSF persistence, `CSFs_processor/` for deterministic CSF algorithms, and `grasp_data_extractor/` for ASF, transition, and orbital post-processing.
- Shared models and environment detection belong in `utils/`; plotting belongs in `graspkit_plot/`.
- The package root exposes metadata. Import functionality from explicit subpackages.
- ML models, ML configuration, training, and workflow state belong in sibling `graspkit-tools`. Do not introduce PyTorch or scikit-learn dependencies here.
- Accept source-agnostic reference energies; NIST export parsing belongs in `nist_data`.

## Environment and checks

Use only `../graspkit-tools/.venv`. Setup and dependency synchronization happen in Tools with the selected `cpu` or `gpu` extra; do not run `uv venv`, `uv sync`, or `uv run` here. Tools installs this package editable, so normal source edits require no reinstall.

From this repository, activate the shared environment before invoking tools:

```sh
source ../graspkit-tools/.venv/bin/activate
```

Choose validation for the change:

- `python -m pytest tests/test_<area>.py`: affected behavior; use `tests/` for a broader regression run when shared interfaces change.
- `ruff check <changed-paths>` and `basedpyright src/`: lint and typing checks for Python changes.
- `python -m build` or `python build_package.py --clean`: package-artifact validation when packaging changes.

Use realistic GRASP inputs and edge cases for loader, parser, and CSF regressions. Keep test inputs small and local to `tests/`; do not depend on private calculation directories. Fix failures introduced by the requested change and report any remaining limitations.

## Conventions

Use Python 3.14-compatible code, explicit public type hints, `pathlib.Path`, and the Ruff/type settings in `pyproject.toml`. Reuse the actual models in `utils/data_modules.py` and environment behavior in `utils/environment_config.py` when relevant. Consult source for available helpers rather than adding aliases for obsolete APIs.

Keep generated artifacts, environments, private datasets, secrets, and cluster paths out of commits. Follow the workspace's separate submodule/parent commit workflow when a commit is requested.
