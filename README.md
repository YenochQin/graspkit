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

Python 3.14 or newer is required.

```bash
uv sync --extra dev
```

When developing in the full workspace, use the environment owned by
`graspkit-tools`; it installs this repository as an editable dependency.

## Imports

```python
from graspkit.data_IO import EnergyFileLoader, MixCoefLoader
from graspkit.grasp_data_extractor import format_energy_configurations
from graspkit.CSFs_processor import batch_asfs_mix_square_above_threshold
from graspkit_plot import configure_matplotlib_for_publication
```

## Development

```bash
pytest tests/
ruff check .
mypy src/
python -m build
```

See [INSTALL.md](INSTALL.md) for installation details.
