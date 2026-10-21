# Installation

`graspkit-atomic` requires Python 3.14 or newer. It provides the `graspkit` and
`graspkit_plot` import packages. The distribution name is changing from `grasp-atomic-kit` to `graspkit-atomic`;
existing import statements need no changes. The commands below apply after the
first release under the new name. For existing projects, replace the old
dependency rather than declaring both distributions, which provide the same
Python packages.

## Install from PyPI

```bash
python -m pip install graspkit-atomic
```

For a fixed release:

```bash
python -m pip install "graspkit-atomic==3.4.1"
```

## Declare a project dependency

Add `"graspkit-atomic==3.4.1"` to the existing `[project].dependencies` list in
your own `pyproject.toml`. A minimal example:

```toml
[project]
name = "atomic-analysis"
version = "0.1.0"
requires-python = ">=3.14"
dependencies = ["graspkit-atomic==3.4.1"]
```

Install your consuming project and its dependencies from its directory with
`python -m pip install .`. Add `"rcsfs==1.3.1"` to the dependency list separately
if your application also needs the Rust CSF library.

Pip installs the declared runtime dependencies; Rust and the GRASP executables
are not needed to install this Python library. Reading calculation results
requires the corresponding GRASP data files.

## Local wheel or source installation

To install a downloaded wheel, use its exact filename:

```bash
python -m pip install /path/to/graspkit_atomic-3.4.1-py3-none-any.whl
```

A standalone source checkout also supports `python -m pip install .`.

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
