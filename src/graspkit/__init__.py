# -*- encoding: utf-8 -*-
"""Core package root for GraspKit."""

from importlib import import_module
from typing import Any

from .version import __version__

__author__ = "YenochQin (秦毅)"

__all__ = [
    "__author__",
    "__version__",
    "MLCalConfig",
    "CalPath",
    "load_config",
]

_CORE_EXPORTS: dict[str, tuple[str, str | None]] = {
    "MLCalConfig": ("graspkit.data_IO", "MLCalConfig"),
    "CalPath": ("graspkit.data_IO", "CalPath"),
    "load_config": ("graspkit.data_IO", "load_config"),
}


def __getattr__(name: str) -> Any:
    """Lazily import selected public objects from subpackages.

    Args:
        name: Attribute name requested from the package root.

    Returns:
        Imported attribute cached in module globals.

    Raises:
        AttributeError: If ``name`` is not a supported lazy export.
    """
    export = _CORE_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name = export
    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return package-root attributes including lazy public exports.

    Returns:
        Sorted list of available attribute names.
    """
    return sorted(set(globals()) | set(__all__) | set(_CORE_EXPORTS))
