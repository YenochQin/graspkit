# -*- encoding: utf-8 -*-
"""
Lightweight utilities exported by default.

Plotting helpers remain available via lazy compatibility shims so importing
`graspkit.utils` does not eagerly pull in matplotlib.
"""

from importlib import import_module
from typing import Any
from warnings import warn

from .data_modules import CSFs, MixCoefficientData, MLDataCounts
from .environment_config import (
    get_environment_config,
    is_debug_mode,
    is_production_mode,
    is_slurm_environment,
)
from .quadrupole_deformation import calculate_deformation
from .tool_function import (
    LS_shell_full_charged,
    chunk_string,
    doubleJ_to_J,
    str_subshell_2_kappa,
)

__all__ = [
    "MixCoefficientData",
    "CSFs",
    "MLDataCounts",
    "str_subshell_2_kappa",
    "doubleJ_to_J",
    "chunk_string",
    "LS_shell_full_charged",
    "get_environment_config",
    "is_slurm_environment",
    "is_debug_mode",
    "is_production_mode",
    "calculate_deformation",
]

_REMOVAL_VERSION = "GraspKit 3.4"

_LEGACY_PLOT_EXPORTS: dict[str, tuple[str, str | None, str]] = {
    "inter_coupling_channel_bar": (
        "graspkit.utils.plot_functions",
        "inter_coupling_channel_bar",
        "graspkit.utils.plot_functions.inter_coupling_channel_bar",
    ),
    "auto_plot_wavefunction_comparison": (
        "graspkit.utils.plot_functions",
        "auto_plot_wavefunction_comparison",
        "graspkit.utils.plot_functions.auto_plot_wavefunction_comparison",
    ),
    "fig_settings": (
        "graspkit.utils.fig_settings",
        None,
        "graspkit.utils.fig_settings",
    ),
}


def __getattr__(name: str) -> Any:
    legacy_export = _LEGACY_PLOT_EXPORTS.get(name)
    if legacy_export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name, replacement = legacy_export
    warn(
        (
            f"`graspkit.utils.{name}` is deprecated and will be removed no earlier "
            f"than {_REMOVAL_VERSION}. Import from `{replacement}` instead."
        ),
        FutureWarning,
        stacklevel=2,
    )

    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_LEGACY_PLOT_EXPORTS))
