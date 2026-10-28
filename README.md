# graspkit

English | [简体中文](README_zh.md)

`graspkit` is the stable foundation for reading, processing, and plotting
GRASP atomic-structure calculation results.

## Package structure

- `graspkit.data_IO`: typed GRASP text/binary loaders and stable CSF writers.
- `graspkit.grasp_data_extractor`: ASF, level, composition, transition, and
  mixing-coefficient post-processing.
- `graspkit.CSFs_processor`: CSF parsing, descriptors, and deterministic
  selection helpers.
- `graspkit.utils`: shared GRASP data models, environment detection, and
  quantum-number helpers.
- `graspkit_plot`: plotting and publication-style helpers.

## Installation

Install [graspkit-atomic from PyPI](https://pypi.org/project/graspkit-atomic/)
with Python 3.14 or newer. The distribution name is `graspkit-atomic`;
its Python import packages are `graspkit` and `graspkit_plot`.

### With uv

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage the
environment and dependencies of your own analysis project. For a new project:

```bash
uv init --python 3.14 atomic-analysis
cd atomic-analysis
uv add graspkit-atomic
uv run python -c "import graspkit; print(graspkit.__version__)"
```

For an existing uv project using Python 3.14 or newer, run `uv add graspkit-atomic`
in its directory. This records the dependency in `pyproject.toml`, updates
`uv.lock`, and installs it in the project's `.venv`. Use `uv run python analysis.py`
to run your own analysis script; manual environment activation is not required.
To pin a particular release, use `uv add "graspkit-atomic==3.4.5"`.

You can also add the dependency to an existing `pyproject.toml` manually:

```toml
[project]
name = "atomic-analysis"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
    "graspkit-atomic==3.4.5",
]
```

Then run `uv sync` in that project's directory to resolve dependencies and
synchronize its environment. Commit `pyproject.toml` and `uv.lock` alongside
your analysis code to record the dependency choices.

### With pip

To install into an existing Python environment:

```bash
python -m pip install graspkit-atomic
```

Both methods provide the same Python API, including plotting. If your project
also uses rCSFs, add it separately with `uv add rcsfs`.
See [INSTALL.md](INSTALL.md) for local-wheel installation details.

## Imports

```python
from graspkit.data_IO import EnergyFileLoader, MixCoefLoader
from graspkit.grasp_data_extractor import format_energy_configurations
from graspkit.CSFs_processor import select_csf_indices_by_ci_squared_cutoff
from graspkit_plot import configure_matplotlib_for_publication
```

## Windows terminal encoding

On Windows terminals that use a legacy code page, Rich output from loaders
such as `MixCoefLoader` may raise `UnicodeEncodeError` when printing scientific
labels such as `cm⁻¹`. Run Python in UTF-8 mode:

```powershell
python -X utf8 your_script.py
```

For a one-line loader check:

```powershell
python -X utf8 -c "from graspkit.data_IO import MixCoefLoader; MixCoefLoader(r'path\to\file.m').load()"
```

This option changes terminal text encoding only; it does not change how GRASP
files are parsed.

## Release preparation

See [the release guide](docs/releasing.md) for local wheel/source builds and
artifact validation and the manual publishing workflow.

## License

GNU GPL version 3 (`GPL-3.0-only`). See [LICENSE](LICENSE).
