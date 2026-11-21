# Installation

This development checkout of `graspkit-atomic` requires Python 3.15 or newer. It provides the `graspkit` and
`graspkit_plot` import packages. The distribution name is `graspkit-atomic`, previously `grasp-atomic-kit`;
existing import statements need no changes. When migrating an existing project,
replace the old dependency rather than declaring both distributions, which
provide the same Python packages.

This checkout prepares 3.6.0 with regression coverage for rCSFs V3 coupling
signatures and their alignment with GRASP mixing blocks.
Use this source checkout until publication; the pinned PyPI examples below
apply after 3.6.0 is available on the package index.

## Install from PyPI

```fish
python -m pip install graspkit-atomic
```

For a fixed release:

```fish
python -m pip install "graspkit-atomic==3.6.0"
```

## Declare a project dependency

Add `"graspkit-atomic==3.6.0"` to the existing `[project].dependencies` list in
your own `pyproject.toml`. A minimal example:

```toml
[project]
name = "atomic-analysis"
version = "0.1.0"
requires-python = ">=3.15"
dependencies = ["graspkit-atomic==3.6.0"]
```

For a uv-managed project, run `uv sync` after editing `pyproject.toml`, then
`uv run python analysis.py` to run your own script in its environment. To let uv
add the dependency and update the lockfile/environment in one step, use
`uv add graspkit-atomic`. See the [README](README.md#with-uv) for a new-project
example. Alternatively, install a pip-managed project with `python -m pip install .`.
Install a V3-capable rCSFs build separately to read V3 descriptors.

Pip installs the declared runtime dependencies; Rust and the GRASP executables
are not needed to install this Python library. Reading calculation results
requires the corresponding GRASP data files.

## Local wheel or source installation

To install a downloaded wheel, use its exact filename:

```fish
python -m pip install /path/to/graspkit_atomic-3.6.0-py3-none-any.whl
```

A standalone source checkout also supports `python -m pip install .`.

## Verification

```fish
python -c "from graspkit.data_IO import EnergyFileLoader; print(EnergyFileLoader)"
python -c "import graspkit_plot; print(graspkit_plot.__file__)"
```

For builds and artifact tests, see [the release guide](docs/releasing.md).
PyTorch and scikit-learn are not dependencies of this library.
