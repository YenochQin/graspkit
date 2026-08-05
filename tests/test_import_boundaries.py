import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def preserve_graspkit_modules() -> None:
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "graspkit" or name.startswith("graspkit.") or name.startswith("graspkit_")
    }
    yield
    for module_name in list(sys.modules):
        if (
            module_name == "graspkit"
            or module_name.startswith("graspkit.")
            or module_name.startswith("graspkit_")
        ):
            sys.modules.pop(module_name, None)
    sys.modules.update(saved_modules)


def _run_python(code: str) -> dict[str, object]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{SRC}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else str(SRC)
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _clear_graspkit_modules() -> None:
    for module_name in list(sys.modules):
        if (
            module_name == "graspkit"
            or module_name.startswith("graspkit.")
            or module_name.startswith("graspkit_")
        ):
            sys.modules.pop(module_name, None)


def test_import_graspkit_keeps_root_lightweight() -> None:
    result = _run_python(
        """
import json
import sys

import graspkit

print(json.dumps({
    "all": graspkit.__all__,
    "ml_loaded": any(name == "graspkit.ml_module" or name.startswith("graspkit.ml_module.") for name in sys.modules),
    "plot_loaded": any(name == "graspkit_plot" or name.startswith("graspkit_plot.") for name in sys.modules),
}))
"""
    )

    assert result["all"] == ["__author__", "__version__"]
    assert result["ml_loaded"] is False
    assert result["plot_loaded"] is False


def test_import_graspkit_data_io_does_not_load_ml_module() -> None:
    result = _run_python(
        """
import json
import sys

import graspkit.data_IO

print(json.dumps({
    "ml_loaded": any(name == "graspkit.ml_module" or name.startswith("graspkit.ml_module.") for name in sys.modules),
    "plot_loaded": any(name == "graspkit_plot" or name.startswith("graspkit_plot.") for name in sys.modules),
}))
"""
    )

    assert result["ml_loaded"] is False
    assert result["plot_loaded"] is False


def test_root_does_not_expose_tools_configuration() -> None:
    _clear_graspkit_modules()
    graspkit = importlib.import_module("graspkit")

    assert "graspkit.data_IO" not in sys.modules
    assert not hasattr(graspkit, "MLCalConfig")
    assert not hasattr(graspkit, "CalPath")
    assert not hasattr(graspkit, "load_config")
    assert "graspkit.ml_module" not in sys.modules


def test_root_no_longer_exposes_ml_or_plot_compat_exports() -> None:
    _clear_graspkit_modules()
    graspkit = importlib.import_module("graspkit")
    graspkit_utils = importlib.import_module("graspkit.utils")

    assert not hasattr(graspkit, "train_model")
    assert not hasattr(graspkit, "inter_coupling_channel_bar")
    assert not hasattr(graspkit_utils, "inter_coupling_channel_bar")
    assert not hasattr(graspkit_utils, "fig_settings")

    assert "graspkit.ml_module" not in sys.modules
    assert "graspkit_plot" not in sys.modules


def test_removed_ml_packages_are_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("graspkit.ml_module")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("graspkit_ml")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("graspkit_config")
