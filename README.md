# graspkit

English | [简体中文](README_zh.md)

`graspkit` is the stable foundation for reading, processing, and plotting
GRASP atomic-structure calculation results. Machine-learning workflow code is
owned by the sibling `graspkit-tools` repository.

## Package structure

- `graspkit.data_IO`: typed GRASP text/binary loaders and stable CSF writers.
- `graspkit.grasp_data_extractor`: ASF, level, composition, transition, and
  mixing-coefficient post-processing.
- `graspkit.CSFs_processor`: CSF parsing, descriptors, and deterministic
  selection helpers.
- `graspkit.utils`: shared GRASP data models, environment detection, and
  quantum-number helpers.
- `graspkit_plot`: plotting and publication-style helpers.

The package deliberately does not contain ML models, training configuration,
or iterative pipeline state. Those live under
`graspkit-tools/ml_CSFs_selection_scripts/ml_csf_choosing/`.

## Installation

Install [graspkit-atomic from PyPI](https://pypi.org/project/graspkit-atomic/)
with Python 3.14 or newer. The distribution name is `graspkit-atomic`;
its Python import packages are `graspkit` and `graspkit_plot`.

### With pip

> The new PyPI name is prepared for publication. The installation commands below
> apply after the first `graspkit-atomic` release.

```bash
python -m pip install graspkit-atomic
```

To reproduce the current release exactly:

```bash
python -m pip install "graspkit-atomic==3.4.1"
```

### In pyproject.toml

Add the dependency to your own project's existing `[project].dependencies`.
A minimal example is:

```toml
[project]
name = "atomic-analysis"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = [
    "graspkit-atomic==3.4.1",
]
```

From your consuming project's directory, install the project and its dependencies:

```bash
python -m pip install .
```

Both installation methods provide the same Python API. If your project also uses
rCSFs, add `"rcsfs==1.3.1"` to the same dependency list. GraspKit's package includes
the plotting API; ML training and workflow orchestration belong to `graspkit-tools`.
See [INSTALL.md](INSTALL.md) for local-wheel and contributor installation details.

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

## Development

In the full workspace, activate the shared environment first:

```bash
source ../graspkit-tools/.venv/bin/activate
pytest tests/
ruff check .
basedpyright src/
python -m build
```

See [INSTALL.md](INSTALL.md) for installation details.

## Release preparation

See [the release guide](docs/releasing.md) for local wheel/source builds and
artifact validation and the manual publishing workflow.

## License

GNU GPL version 3 (`GPL-3.0-only`). See [LICENSE](LICENSE).
