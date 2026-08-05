"""Plotting and publication-style helpers for graspkit.

This package isolates ``matplotlib``-dependent code so that importing
``graspkit`` or ``graspkit.utils`` never triggers matplotlib loading.
Import explicitly from here when publication-quality figures are needed::

    from graspkit_plot import configure_matplotlib_for_publication
"""

from . import fig_settings as _fig_settings
from . import plot_functions as _plot_functions
from .fig_settings import *  # noqa: F403
from .plot_functions import *  # noqa: F403

__all__ = [
    name
    for module in (_fig_settings, _plot_functions)
    for name in dir(module)
    if not name.startswith("_")
]
