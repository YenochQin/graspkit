"""Plotting and publication-style helpers for graspkit.

This package isolates ``matplotlib``-dependent code so that importing
``graspkit`` or ``graspkit.utils`` never triggers matplotlib loading.
Import explicitly from here when publication-quality figures are needed::

    from graspkit_plot import configure_matplotlib_for_publication
"""

from .fig_settings import *  # noqa: F403
from .mid_coupling import *  # noqa: F403
from .plot_functions import *  # noqa: F403

__all__ = [
    "FIGURE_SIZES",
    "JOURNAL_COLOR_SCHEMES",
    "LEGEND_SIZE_PRESETS",
    "MidCouplingPlotSeries",
    "PlotMode",
    "SAVE_FORMATS",
    "SUBPLOT_LAYOUTS",
    "SUBPLOT_SIZE_FACTORS",
    "SUBPLOT_SPACING",
    "SingleSeriesMode",
    "XTransform",
    "add_reference_lines_to_subplots",
    "apply_transformed_x_axis_settings",
    "calculate_subplot_figure_size",
    "configure_for_latex",
    "configure_matplotlib_for_publication",
    "configure_subplot_grid",
    "build_mid_coupling_plot_series",
    "create_multi_subplot_figure",
    "create_publication_figure",
    "create_shared_colorbar",
    "disable_font_warnings",
    "finalize_figure_layout",
    "get_cycled_linestyles",
    "get_cycled_plot_colors",
    "get_subplot_layout",
    "init_publication_style",
    "inter_coupling_channel_bar",
    "optimize_for_multi_subplot",
    "optimize_for_plot_type",
    "plot_mid_coupling_contribution",
    "resolve_transformed_xscale",
    "rwfn_plot",
    "rwfns_compare_plot",
    "save_figure",
    "save_multi_subplot_figure",
    "set_color_scheme",
    "set_figure_size",
    "set_legend_size",
    "use_plain_linear_x_axis",
]
