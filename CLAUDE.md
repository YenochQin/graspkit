# CLAUDE.md

## Project Overview

`graspkit` is the stable library for reading, processing, and plotting GRASP2018 calculation results. ML models, ML configuration, and iterative CSF training are owned by the sibling `graspkit-tools` repository.

## Architecture

- `src/graspkit/data_IO/`: typed GRASP text and binary loaders plus stable CSF writers.
- `src/graspkit/grasp_data_extractor/`: ASF, energy-level, LSJ, transition, and mixing-coefficient post-processing.
- `src/graspkit/CSFs_processor/`: CSF parsing, descriptors, and deterministic selection algorithms.
- `src/graspkit/utils/`: shared GRASP data models, environment detection, and utility functions.
- `src/graspkit_plot/`: plotting and publication-style helpers.

The package root exposes metadata only. Import functionality from explicit subpackages. Do not add PyTorch, scikit-learn, ML configuration models, or workflow state back into this repository.

## Development

Use only `../graspkit-tools/.venv` for every Python command in this repository. It is created and synchronized by `uv` in `graspkit-tools/`, which installs this repository editable. Do not run `uv sync` here or create/use `graspkit/.venv`.

```bash
cd ../graspkit-tools
uv sync
uv run pytest ../graspkit/tests
```

From `graspkit/`, activate the shared environment and run the repository checks normally:

```bash
source ../graspkit-tools/.venv/bin/activate
pytest tests/
ruff check .
basedpyright src/
python -m build
```

Python 3.14 or newer is required. Prefer explicit type hints, `pathlib.Path`, and realistic GRASP fixtures for parser changes.
