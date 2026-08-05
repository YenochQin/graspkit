# -*- encoding: utf-8 -*-
import warnings
from typing import Any, Literal

import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .fig_settings import (
        apply_transformed_x_axis_settings,
        configure_matplotlib_for_publication,
        create_multi_subplot_figure,
        finalize_figure_layout,
        get_cycled_linestyles,
        get_cycled_plot_colors,
        optimize_for_plot_type,
        resolve_transformed_xscale,
        set_color_scheme,
        set_figure_size,
    )


PlotMode = Literal["density", "P", "Q", "components"]
SingleSeriesMode = Literal["density", "P", "Q"]
XTransform = Literal["sqrt", "linear", "raw", "log1p"]


def _validate_plot_mode(plot_mode: str) -> None:
    if plot_mode not in {"density", "P", "Q", "components"}:
        raise ValueError("plot_mode must be one of 'density', 'P', 'Q', or 'components'")


def _component_column(orbital: str, component: Literal["P", "Q"]) -> str:
    return f"{component}({orbital})"


def _validate_rwfn_columns(data: pl.DataFrame, orbitals: list[str], x_col: str) -> None:
    missing_columns: list[str] = []
    if x_col not in data.columns:
        missing_columns.append(x_col)

    for orbital in orbitals:
        for component in ("P", "Q"):
            col_name = _component_column(orbital, component)
            if col_name not in data.columns:
                missing_columns.append(col_name)

    if missing_columns:
        raise ValueError(
            f"Wavefunction DataFrame is missing required columns: {missing_columns}"
        )


def _x_expr(x_col: str, x_transform: XTransform) -> pl.Expr:
    expr = pl.col(x_col)
    if x_transform == "sqrt":
        return expr.sqrt()
    if x_transform in {"linear", "raw"}:
        return expr
    if x_transform == "log1p":
        return (expr + 1).log()
    raise ValueError(
        "x_transform must be one of 'sqrt', 'linear', 'raw', or 'log1p'"
    )


def _x_series(data: pl.DataFrame, x_col: str, x_transform: XTransform) -> pl.Series:
    series = data.select(_x_expr(x_col, x_transform).alias(x_col)).to_series()
    if bool(series.is_nan().any()) or bool(series.is_infinite().any()):
        raise ValueError(f"x_transform={x_transform} produced non-finite x values")
    return series


def _rwfn_y_series(
        data: pl.DataFrame,
        orbital: str,
        plot_mode: SingleSeriesMode,
    ) -> pl.Series:
    p_col = _component_column(orbital, "P")
    q_col = _component_column(orbital, "Q")

    if plot_mode == "density":
        return data.select(
            (pl.col(p_col) ** 2 + pl.col(q_col) ** 2).alias(orbital)
        ).to_series()
    if plot_mode == "P":
        return data[p_col]
    if plot_mode == "Q":
        return data[q_col]
    raise ValueError("plot_mode must be one of 'density', 'P', or 'Q'")


def _auto_rwfn_layout(n_items: int) -> str:
    ncols = int(np.ceil(np.sqrt(n_items)))
    nrows = int(np.ceil(n_items / ncols))
    return f"{nrows}x{ncols}"


def _parse_layout(layout: str) -> tuple[int, int]:
    try:
        nrows_text, ncols_text = layout.split("x")
        nrows = int(nrows_text)
        ncols = int(ncols_text)
    except ValueError as exc:
        raise ValueError("layout must use the '<rows>x<columns>' format") from exc

    if nrows <= 0 or ncols <= 0:
        raise ValueError("layout rows and columns must be greater than 0")
    return nrows, ncols


def _resolve_rwfn_layout(layout: str | None, plot_mode: PlotMode, n_orbitals: int) -> str:
    if plot_mode == "components":
        if layout is None:
            return f"2x{n_orbitals}"
        group_nrows, group_ncols = _parse_layout(layout)
        return f"{group_nrows * 2}x{group_ncols}"
    if layout is not None:
        return layout
    return _auto_rwfn_layout(n_orbitals)


def _build_plot_items(
        orbitals: list[str],
        plot_mode: PlotMode,
    ) -> list[tuple[SingleSeriesMode, str, str]]:
    if plot_mode == "components":
        plot_items: list[tuple[SingleSeriesMode, str, str]] = []
        for orbital in orbitals:
            plot_items.append(("P", orbital, _component_column(orbital, "P")))
            plot_items.append(("Q", orbital, _component_column(orbital, "Q")))
        return plot_items
    if plot_mode == "density":
        return [("density", orbital, orbital) for orbital in orbitals]
    if plot_mode == "P":
        return [("P", orbital, _component_column(orbital, "P")) for orbital in orbitals]
    if plot_mode == "Q":
        return [("Q", orbital, _component_column(orbital, "Q")) for orbital in orbitals]
    raise ValueError("plot_mode must be one of 'density', 'P', 'Q', or 'components'")


def _rwfn_plot_position(
        index: int,
        plot_mode: PlotMode,
        ncols: int,
    ) -> tuple[int, int]:
    if plot_mode == "components":
        group_index = index // 2
        component_index = index % 2
        return (group_index // ncols) * 2 + component_index, group_index % ncols
    return index // ncols, index % ncols


def _rwfn_single_plot_style(
        series_mode: SingleSeriesMode,
        orbital: str,
        orbitals: list[str],
        orbital_colors: list[str],
        color: str | None,
        linestyle: Any,
        plot_mode: PlotMode,
    ) -> tuple[str | None, Any]:
    if color is not None or plot_mode != "components":
        return color, linestyle

    orbital_color = orbital_colors[orbitals.index(orbital)]
    return orbital_color, linestyle


def _significant_mask_expr(
        orbitals: list[str],
        plot_mode: PlotMode,
        threshold: float,
    ) -> pl.Expr:
    masks: list[pl.Expr] = []
    for orbital in orbitals:
        p_col = _component_column(orbital, "P")
        q_col = _component_column(orbital, "Q")
        if plot_mode == "density":
            masks.append((pl.col(p_col) ** 2 + pl.col(q_col) ** 2).abs() > threshold)
        elif plot_mode == "P":
            masks.append(pl.col(p_col).abs() > threshold)
        elif plot_mode == "Q":
            masks.append(pl.col(q_col).abs() > threshold)
        elif plot_mode == "components":
            masks.append(pl.col(p_col).abs() > threshold)
            masks.append(pl.col(q_col).abs() > threshold)

    if not masks:
        raise ValueError("orbitals must contain at least one orbital")

    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = combined_mask | mask
    return combined_mask


def _last_significant_x_from_polars(
        data: pl.DataFrame,
        orbitals: list[str],
        x_col: str,
        plot_mode: PlotMode,
        x_transform: XTransform,
        threshold: float,
    ) -> float | None:
    selected = (
        data.with_columns(_x_expr(x_col, x_transform).alias("__x_plot"))
        .filter(_significant_mask_expr(orbitals, plot_mode, threshold))
        .select(pl.col("__x_plot").last())
    )
    value = selected.item()
    return None if value is None else float(value)


def _calculate_default_max_x_from_polars(
        data: pl.DataFrame,
        x_col: str,
        x_transform: XTransform,
    ) -> float:
    last_x = float(_x_series(data, x_col, x_transform)[-1])
    return float(int(np.ceil(last_x / 10)) * 10)


def _calculate_auto_max_x_from_polars(
        data_list: list[pl.DataFrame],
        orbitals: list[str],
        x_col: str,
        plot_mode: PlotMode,
        x_transform: XTransform,
        threshold: float,
        padding: float,
    ) -> float | None:
    max_x_candidates = [
        last_x
        for data in data_list
        if (
            last_x := _last_significant_x_from_polars(
                data,
                orbitals,
                x_col,
                plot_mode,
                x_transform,
                threshold,
            )
        ) is not None
    ]
    if not max_x_candidates:
        return None
    return max(max_x_candidates) * padding


def _validate_auto_max_x_settings(threshold: float, padding: float) -> None:
    if threshold < 0:
        raise ValueError("x_tail_threshold must be greater than or equal to 0")
    if padding <= 0:
        raise ValueError("x_tail_padding must be greater than 0")


def inter_coupling_channel_bar(
        categories: Any,
        quantity: Any,
        sum_squared_ci: Any,
        colors: list[Any] | None = None,
        figsize: str | tuple[float, float] = 'double_column',
        color_scheme: str = 'nature',
        min_threshold: int = 10,
        fontsize_scale: float = 1.0,
    ) -> tuple[Figure, Axes, Axes]:
    """
    创建专业的双轴柱状图，展示占比和贡献值

    Features:
    - 左轴：占比百分比（柱状图）
    - 右轴：贡献值（折线图）
    - 自动聚合小占比类别
    - 专业发表级图表样式
    - 完整边框和标签

    Args:
        categories: 类别名称列表
        quantity: 各类别数量/占比
        sum_squared_ci: 各类别贡献值
        colors: 自定义颜色列表（可选）
        figsize: 图表尺寸名称或具体尺寸（默认: 'double_column'）
        color_scheme: 配色方案（默认: 'nature'）
        min_threshold: 最小占比阈值，小于此值的类别将聚合为"Others"（默认: 10）
        fontsize_scale: 字体缩放因子（默认: 1.0）

    Returns:
        tuple: (fig, ax1, ax2) matplotlib图表对象

    Example:
        >>> categories = ['A', 'B', 'C', 'D', 'E']
        >>> quantity = [15, 25, 8, 35, 12]
        >>> contribution = [0.15, 0.25, 0.08, 0.35, 0.12]
        >>> fig, ax1, ax2 = dual_axis_bar_chart(categories, quantity, contribution)
    """

    # 配置matplotlib为发表级别
    if not configure_matplotlib_for_publication():
        warnings.warn("Failed to configure matplotlib for publication, using default settings")

    # 转换为numpy数组
    categories = np.array(categories)
    quantity = np.array(quantity, dtype=float)
    sum_squared_ci = np.array(sum_squared_ci, dtype=float)

    # 处理元组类别名称
    def format_category(cat: Any) -> str:
        """Convert a category value to a display label.

        Args:
            cat: Category value, possibly represented as a tuple.

        Returns:
            String label suitable for axis ticks and legends.
        """
        if isinstance(cat, tuple):
            return '-'.join(str(item) for item in cat)
        else:
            return str(cat)

    categories = [format_category(cat) for cat in categories]
    categories = np.array(categories)

    # 归一化占比到100%
    quantity_percent = quantity / quantity.sum() * 100

    # 聚合小占比类别
    if min_threshold > 0:
        mask_above_threshold = quantity_percent >= min_threshold

        if np.any(~mask_above_threshold):
            # 计算聚合值
            aggregated_quantity = quantity[~mask_above_threshold].sum()
            aggregated_sum_squared_ci = sum_squared_ci[~mask_above_threshold].sum()
            aggregated_quantity_percent = quantity_percent[~mask_above_threshold].sum()
            # aggregated_count = np.sum(~mask_above_threshold)

            # 保留大于阈值的类别
            categories_keep = categories[mask_above_threshold]
            quantity_keep = quantity[mask_above_threshold]
            sum_squared_ci_keep = sum_squared_ci[mask_above_threshold]
            quantity_percent_keep = quantity_percent[mask_above_threshold]

            # 添加"Others"类别
            categories = np.append(categories_keep, f"Others")
            quantity = np.append(quantity_keep, aggregated_quantity)
            sum_squared_ci = np.append(sum_squared_ci_keep, aggregated_sum_squared_ci)
            quantity_percent = np.append(quantity_percent_keep, aggregated_quantity_percent)

    # 按贡献值排序，但保持"Others"在最后
    others_mask = np.char.startswith(categories, 'Others')
    normal_mask = ~others_mask

    if np.any(normal_mask):
        normal_idxs = np.where(normal_mask)[0]
        others_idxs = np.where(others_mask)[0]

        normal_sort_idx = normal_idxs[np.argsort(-sum_squared_ci[normal_idxs])]
        final_sort_idx = np.concatenate([normal_sort_idx, others_idxs])
    else:
        final_sort_idx = np.arange(len(categories))

    categories = categories[final_sort_idx]
    quantity = quantity[final_sort_idx]
    sum_squared_ci = sum_squared_ci[final_sort_idx]
    quantity_percent = quantity_percent[final_sort_idx]

    # 设置图表尺寸
    if isinstance(figsize, str):
        figsize = set_figure_size(figsize)

    # 创建双轴图表
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # 设置配色方案
    set_color_scheme(color_scheme)

    # 设置颜色
    if colors is None:
        # 使用matplotlib的默认配色
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i % 20) for i in range(len(categories))]

    # 优化柱状图设置
    optimize_for_plot_type('bar')

    # 创建柱状图（占比）
    bars = ax1.bar(range(len(categories)), quantity_percent,
                   color=colors, alpha=0.7, edgecolor='black', linewidth=0.8)

    # 创建折线图（贡献值）
    line = ax2.plot(
                range(len(categories)),
                sum_squared_ci,
                'o-',
                color='darkred',
                linewidth=2, markersize=6,
                alpha=0.8,
                label='Contribution Value', markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor='darkred')

    # 添加数据标签
    for i, (bar, pct, val, cat) in enumerate(zip(bars, quantity_percent, sum_squared_ci, categories)):
        height = bar.get_height()

        # 添加占比和贡献值标签
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(quantity_percent) * 0.02,
                f'{pct:.1f}%\n{val:.3f}',
                ha='center', va='bottom', fontsize=int(10 * fontsize_scale))

    # 设置坐标轴标签
    label_fontsize = int(12 * fontsize_scale)
    tick_fontsize = int(10 * fontsize_scale)

    ax1.set_ylabel('Percentage Share (%)', fontsize=label_fontsize, color='blue')
    ax1.set_xlabel('Intermediate coupling channel', fontsize=label_fontsize)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=tick_fontsize)
    ax1.tick_params(axis='x', labelsize=tick_fontsize, rotation=45)

    ax2.set_ylabel('Contribution Value', fontsize=label_fontsize, color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred', labelsize=tick_fontsize)

    # 设置Y轴范围，为顶部标签留出空间
    max_percentage = max(quantity_percent)


    # 设置主Y轴（百分比），增加20%的上边距为标签留空间
    ax1.set_ylim(0, max_percentage * 1.25)

    # 设置副Y轴（贡献值），增加20%的上边距
    ax2.set_ylim(0, 1.05)

    # 设置完整的边框
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['top'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

    # 确保Y轴刻度标签位置正确
    ax1.tick_params(
        axis='y',
        which='both',
        left=True,
        right=False,
        labelleft=True,
        labelright=False)
    ax2.tick_params(
        axis='y',
        which='both',
        left=False,
        right=True,
        labelleft=False,
        labelright=True)

    # 设置X轴标签
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(
        categories,
        rotation=45,
        ha='right',
        fontsize=tick_fontsize)

    # 设置图例
    legend_fontsize = int(10 * fontsize_scale)
    ax1.legend(
        handles=line,
        loc='upper right',
        fontsize=legend_fontsize)

    # 添加网格
    ax1.grid(True, alpha=0.3, axis='y')

    # 调整布局
    plt.tight_layout()

    return fig, ax1, ax2


def _resolve_rwfn_max_x(
        data_list: list[pl.DataFrame],
        orbitals: list[str],
        x_col: str,
        plot_mode: PlotMode,
        x_transform: XTransform,
        max_x: float | None,
        auto_max_x: bool,
        threshold: float,
        padding: float,
    ) -> float:
    if max_x is not None:
        return max_x

    if auto_max_x:
        detected_max_x = _calculate_auto_max_x_from_polars(
            data_list,
            orbitals,
            x_col,
            plot_mode,
            x_transform,
            threshold,
            padding,
        )
        if detected_max_x is not None:
            return detected_max_x

    return _calculate_default_max_x_from_polars(data_list[0], x_col, x_transform)


def _finalize_rwfn_plot(
        fig: Figure,
        axes: np.ndarray,
        layout: str,
        suptitle: str,
        xlabel: str | None,
        ylabel: str | None,
    ) -> None:
    _parse_layout(layout)
    for ax in axes.flat:
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7, linewidth=1)

    if suptitle is not None:
        suptitle_text = fig.suptitle(suptitle, fontsize=16)
        suptitle_text.set_y(0.995)

    if xlabel is not None:
        fig.text(0.5, 0.01, xlabel, ha="center", va="center", fontsize=12)
    if ylabel is not None:
        fig.text(0.01, 0.5, ylabel, ha="center", va="center", rotation="vertical", fontsize=12)

    finalize_figure_layout(
        fig,
        has_axis_labels=xlabel is not None or ylabel is not None,
    )


def rwfn_plot(
        data: pl.DataFrame,
        orbitals: list[str],
        x_col: str = "r(a.u)",
        plot_mode: PlotMode = "density",
        layout: str | None = None,
        base_size: str | tuple[float, float] = "single_column",
        spacing: str = "normal",
        color_scheme: str = "nature",
        legend_size: str = "medium",
        alpha: float = 0.75,
        max_x: float | None = None,
        xscale: str | None = None,
        linthresh: int = 1,
        suptitle: str = "Radial Wavefunction",
        color: str | None = None,
        linestyle: Any = "-",
        xlabel: str | None = None,
        ylabel: str | None = None,
        x_transform: XTransform = "sqrt",
        auto_max_x: bool = True,
        x_tail_threshold: float = 0.0,
        x_tail_padding: float = 1.05,
    ) -> tuple[Figure, np.ndarray]:
    """Plot radial wavefunctions from one Polars DataFrame."""
    _validate_plot_mode(plot_mode)
    _validate_auto_max_x_settings(x_tail_threshold, x_tail_padding)
    _validate_rwfn_columns(data, orbitals, x_col)

    resolved_layout = _resolve_rwfn_layout(layout, plot_mode, len(orbitals))
    nrows, ncols = _parse_layout(resolved_layout)
    plot_items = _build_plot_items(orbitals, plot_mode)
    effective_xscale = resolve_transformed_xscale(xscale, x_transform)
    resolved_max_x = _resolve_rwfn_max_x(
        [data],
        orbitals,
        x_col,
        plot_mode,
        x_transform,
        max_x,
        auto_max_x,
        x_tail_threshold,
        x_tail_padding,
    )

    fig, axes = create_multi_subplot_figure(
        layout=resolved_layout,
        base_size=base_size,
        spacing=spacing,
        color_scheme=color_scheme,
        legend_size=legend_size,
    )

    x_values = _x_series(data, x_col, x_transform)
    orbital_colors = get_cycled_plot_colors(None, len(orbitals), color_scheme)
    max_items = nrows * ncols
    if len(plot_items) > max_items:
        warnings.warn("More wavefunction plot items were requested than the layout can show")

    for index, (series_mode, orbital, title) in enumerate(plot_items[:max_items]):
        row, col = _rwfn_plot_position(index, plot_mode, ncols)
        y_values = _rwfn_y_series(data, orbital, series_mode)
        line_color, line_style = _rwfn_single_plot_style(
            series_mode,
            orbital,
            orbitals,
            orbital_colors,
            color,
            linestyle,
            plot_mode,
        )
        axes[row, col].plot(
            x_values,
            y_values,
            alpha=alpha,
            color=line_color,
            linestyle=line_style,
        )
        apply_transformed_x_axis_settings(
            axes[row, col],
            resolved_max_x,
            effective_xscale,
            linthresh,
            x_transform,
            show_legend=False,
        )
        axes[row, col].set_title(title)

    _finalize_rwfn_plot(
        fig,
        axes,
        resolved_layout,
        suptitle,
        xlabel,
        ylabel,
    )
    return fig, axes


def rwfns_compare_plot(
        data_list: list[pl.DataFrame],
        orbitals: list[str],
        x_col: str = "r(a.u)",
        labels: list[str] | None = None,
        plot_mode: PlotMode = "density",
        layout: str | None = None,
        base_size: str | tuple[float, float] = "single_column",
        spacing: str = "normal",
        color_scheme: str = "nature",
        legend_size: str = "medium",
        alpha: float = 0.75,
        max_x: float | None = None,
        xscale: str | None = None,
        linthresh: int = 1,
        suptitle: str = "Radial Wavefunction Comparison",
        colors: list[str] | None = None,
        linestyles: list[Any] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        x_transform: XTransform = "sqrt",
        auto_max_x: bool = True,
        x_tail_threshold: float = 0.0,
        x_tail_padding: float = 1.05,
    ) -> tuple[Figure, np.ndarray]:
    """Compare radial wavefunctions from multiple Polars DataFrames."""
    _validate_plot_mode(plot_mode)
    _validate_auto_max_x_settings(x_tail_threshold, x_tail_padding)
    if not data_list:
        raise ValueError("data_list must contain at least one DataFrame")
    for data in data_list:
        _validate_rwfn_columns(data, orbitals, x_col)

    n_datasets = len(data_list)
    if labels is None:
        labels = [f"Data {i + 1}" for i in range(n_datasets)]
    elif len(labels) != n_datasets:
        raise ValueError(f"Number of labels ({len(labels)}) must match number of datasets ({n_datasets})")

    resolved_layout = _resolve_rwfn_layout(layout, plot_mode, len(orbitals))
    nrows, ncols = _parse_layout(resolved_layout)
    plot_items = _build_plot_items(orbitals, plot_mode)
    colors = get_cycled_plot_colors(colors, n_datasets, color_scheme)
    linestyles = get_cycled_linestyles(linestyles, n_datasets)
    effective_xscale = resolve_transformed_xscale(xscale, x_transform)
    resolved_max_x = _resolve_rwfn_max_x(
        data_list,
        orbitals,
        x_col,
        plot_mode,
        x_transform,
        max_x,
        auto_max_x,
        x_tail_threshold,
        x_tail_padding,
    )

    fig, axes = create_multi_subplot_figure(
        layout=resolved_layout,
        base_size=base_size,
        spacing=spacing,
        color_scheme=color_scheme,
        legend_size=legend_size,
    )

    x_values_by_dataset = [
        _x_series(data, x_col, x_transform) for data in data_list
    ]
    max_items = nrows * ncols
    if len(plot_items) > max_items:
        warnings.warn("More wavefunction plot items were requested than the layout can show")

    for index, (series_mode, orbital, title) in enumerate(plot_items[:max_items]):
        row, col = _rwfn_plot_position(index, plot_mode, ncols)
        ax = axes[row, col]

        for dataset_index, data in enumerate(data_list):
            y_values = _rwfn_y_series(data, orbital, series_mode)
            ax.plot(
                x_values_by_dataset[dataset_index],
                y_values,
                alpha=alpha,
                label=labels[dataset_index],
                color=colors[dataset_index],
                linestyle=linestyles[dataset_index],
            )

        apply_transformed_x_axis_settings(
            ax,
            resolved_max_x,
            effective_xscale,
            linthresh,
            x_transform,
            show_legend=True,
        )
        ax.set_title(title)

    _finalize_rwfn_plot(
        fig,
        axes,
        resolved_layout,
        suptitle,
        xlabel,
        ylabel,
    )
    return fig, axes
