import ast
import json
import os
import subprocess
import sys
from pathlib import Path

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


def test_import_graspkit_core_keeps_ml_and_plot_unloaded() -> None:
    result = _run_python(
        """
import json
import sys

import graspkit_core
from graspkit_core.data_IO import MLCalConfig

print(json.dumps({
    "ml_loaded": any(name == "graspkit.ml_module" or name.startswith("graspkit.ml_module.") for name in sys.modules),
    "plot_loaded": any(name == "graspkit.utils.plot_functions" or name.startswith("graspkit.utils.plot_functions.") for name in sys.modules),
    "config_name": MLCalConfig.__name__,
    "has_energy_loader": hasattr(graspkit_core, "EnergyFileLoader"),
}))
"""
    )

    assert result["ml_loaded"] is False
    assert result["plot_loaded"] is False
    assert result["config_name"] == "MLCalConfig"
    assert result["has_energy_loader"] is True


def test_import_graspkit_ml_loads_ml_layer() -> None:
    result = _run_python(
        """
import json
import sys

import graspkit_ml

print(json.dumps({
    "ml_loaded": any(name == "graspkit.ml_module" or name.startswith("graspkit.ml_module.") for name in sys.modules),
    "has_train_model": hasattr(graspkit_ml, "train_model"),
}))
"""
    )

    assert result["ml_loaded"] is True
    assert result["has_train_model"] is True


def test_import_graspkit_plot_loads_plot_layer_without_ml() -> None:
    result = _run_python(
        """
import json
import sys

import graspkit_plot

print(json.dumps({
    "ml_loaded": any(name == "graspkit.ml_module" or name.startswith("graspkit.ml_module.") for name in sys.modules),
    "plot_loaded": any(name == "graspkit.utils.plot_functions" or name.startswith("graspkit.utils.plot_functions.") for name in sys.modules),
    "has_plot_bar": hasattr(graspkit_plot, "inter_coupling_channel_bar"),
}))
"""
    )

    assert result["ml_loaded"] is False
    assert result["plot_loaded"] is True
    assert result["has_plot_bar"] is True


def test_core_sources_do_not_import_ml_module() -> None:
    core_roots = [
        ROOT / "src" / "graspkit" / "data_IO",
        ROOT / "src" / "graspkit" / "grasp_data_extractor",
        ROOT / "src" / "graspkit" / "CSFs_processor",
        ROOT / "src" / "graspkit" / "utils" / "data_modules.py",
        ROOT / "src" / "graspkit" / "utils" / "environment_config.py",
        ROOT / "src" / "graspkit" / "utils" / "quadrupole_deformation.py",
        ROOT / "src" / "graspkit" / "utils" / "tool_function.py",
        ROOT / "src" / "graspkit_core",
    ]

    offending_imports: list[str] = []

    for root in core_roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in paths:
            module_tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(module_tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "graspkit.ml_module" or alias.name.startswith(
                            "graspkit.ml_module."
                        ):
                            offending_imports.append(f"{path}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module is None:
                        continue
                    parts = node.module.split(".")
                    if "ml_module" in parts:
                        offending_imports.append(
                            f"{path}: from {'.' * node.level}{node.module} import ..."
                        )

    assert offending_imports == []
