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
        if module_name == "graspkit" or module_name.startswith("graspkit."):
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
    "plot_loaded": any(name == "graspkit.utils.plot_functions" or name.startswith("graspkit.utils.plot_functions.") for name in sys.modules),
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
    "plot_loaded": any(name == "graspkit.utils.plot_functions" or name.startswith("graspkit.utils.plot_functions.") for name in sys.modules),
}))
"""
    )

    assert result["ml_loaded"] is False
    assert result["plot_loaded"] is False


def test_root_legacy_export_warns_and_loads_on_demand() -> None:
    _clear_graspkit_modules()
    graspkit = importlib.import_module("graspkit")

    assert "graspkit.data_IO" not in sys.modules

    with pytest.warns(
        FutureWarning,
        match=r"`graspkit\.MLCalConfig` is deprecated.*graspkit\.data_IO\.MLCalConfig",
    ):
        ml_config_model = graspkit.MLCalConfig

    assert ml_config_model.__name__ == "MLCalConfig"
    assert "graspkit.data_IO" in sys.modules
    assert "graspkit.ml_module" not in sys.modules


def test_utils_plot_export_warns_and_loads_on_demand() -> None:
    _clear_graspkit_modules()
    graspkit_utils = importlib.import_module("graspkit.utils")

    assert "graspkit.utils.plot_functions" not in sys.modules

    with pytest.warns(
        FutureWarning,
        match=(
            r"`graspkit\.utils\.inter_coupling_channel_bar` is deprecated.*"
            r"graspkit\.utils\.plot_functions\.inter_coupling_channel_bar"
        ),
    ):
        plot_function = graspkit_utils.inter_coupling_channel_bar

    assert plot_function.__name__ == "inter_coupling_channel_bar"
    assert "graspkit.utils.plot_functions" in sys.modules
