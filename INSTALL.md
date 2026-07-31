# Installation

`graspkit` requires Python 3.14 or newer.

## Standalone development

```bash
uv venv
source .venv/bin/activate
uv sync --extra dev
```

On Windows, activate with `.venv\Scripts\activate`.

Verify the installation:

```bash
python -c "import graspkit; print(graspkit.__version__)"
python -c "from graspkit.data_IO import EnergyFileLoader; print(EnergyFileLoader)"
```

## Full workspace development

For changes consumed by `graspkit-tools`, use the Tools environment:

```bash
cd ../graspkit-tools
uv sync
uv run python -c "import graspkit; print(graspkit.__version__)"
```

The Tools project installs `../graspkit` as an editable dependency, so Python
source changes are visible immediately.

## ML dependencies

PyTorch, scikit-learn, ML configuration models, and training code are not part
of this package. Install and run them through `graspkit-tools`.
