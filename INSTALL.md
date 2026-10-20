# Installation

`grasp-atomic-kit` requires Python 3.14 or newer. It provides the `graspkit` and
`graspkit_plot` import packages. The distribution was renamed from `grasp-kit`
before its first PyPI publication; existing import statements need no changes.

## Install a prepared release

Use the exact wheel filename supplied with the release:

```bash
python -m pip install /path/to/grasp_atomic_kit-3.4.0-py3-none-any.whl
```

Once the package has been published on PyPI:

```bash
python -m pip install grasp-atomic-kit
```

A standalone source checkout also supports `python -m pip install .`.
Pip installs the declared runtime dependencies; Rust and the GRASP executables
are not needed to install this Python library. Reading calculation results
requires the corresponding GRASP data files.

## Full workspace development

Use the environment owned by Tools. Run setup from `graspkit-tools/` only:

```bash
cd ../graspkit-tools
uv sync --extra cpu
source .venv/bin/activate
python -c "import graspkit; print(graspkit.__version__)"
```

Select `--extra gpu` instead on a CUDA host. On Windows activate
`.venv\Scripts\activate` after syncing. Tools installs GraspKit as an editable
path dependency; ordinary Python edits are immediately visible. Do not install
release wheels into this shared environment for testing: the release checker
loads an unpacked wheel in a disposable directory instead.

## Verification

```bash
python -c "from graspkit.data_IO import EnergyFileLoader; print(EnergyFileLoader)"
python -c "import graspkit_plot; print(graspkit_plot.__file__)"
```

For builds and artifact tests, see [the release guide](docs/releasing.md).
PyTorch, scikit-learn and training workflows belong to `graspkit-tools` and
are not dependencies of this library.
