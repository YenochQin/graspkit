#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Id :fig_settings.py
@date :2026/02/04 16:24:02
@author :YenochQin (秦毅)
"""

"""
图表样式设置模块

提供统一的图表样式配置，适用于科学发表。
现在使用英文标签，无需中文字体支持。

功能包括：
- 基础matplotlib配置
- 科学期刊配色方案
- 预设图表尺寸
- 图例大小和样式配置
- 针对不同用途的保存设置
- LaTeX兼容性设置
- 图表类型专用优化
- 多子图布局和配置
- 多子图间距和尺寸管理
- 共享颜色条和图例
- 多子图保存优化
"""
import re
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler


def configure_matplotlib_for_publication():
    """
    配置matplotlib用于科学发表的图表样式

    Returns:
        bool: 配置是否成功
    """
    try:
        # 高质量图形设置
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["savefig.dpi"] = 300
        plt.rcParams["savefig.bbox"] = "tight"
        plt.rcParams["savefig.pad_inches"] = 0.1

        # 字体设置（英文）
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["legend.fontsize"] = 10
        plt.rcParams["figure.titlesize"] = 16

        # 线条和标记设置
        plt.rcParams["lines.linewidth"] = 1.5
        plt.rcParams["lines.markersize"] = 6
        plt.rcParams["patch.linewidth"] = 0.5

        # 坐标轴设置
        plt.rcParams["axes.linewidth"] = 1.0
        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["axes.spines.right"] = False
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.3
        plt.rcParams["grid.linewidth"] = 0.5

        # 负号显示修复
        plt.rcParams["axes.unicode_minus"] = False

        # 图例设置
        plt.rcParams["legend.frameon"] = True
        plt.rcParams["legend.framealpha"] = 0.9
        plt.rcParams["legend.fancybox"] = True
        plt.rcParams["legend.shadow"] = False
        plt.rcParams["legend.fontsize"] = 10
        plt.rcParams["legend.title_fontsize"] = 12
        plt.rcParams["legend.borderaxespad"] = 0.5
        plt.rcParams["legend.borderpad"] = 0.4
        plt.rcParams["legend.columnspacing"] = 1.0
        plt.rcParams["legend.handletextpad"] = 0.8
        plt.rcParams["legend.handlelength"] = 1.5
        plt.rcParams["legend.labelspacing"] = 0.5

        # 刻度设置
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["xtick.major.size"] = 3
        plt.rcParams["ytick.major.size"] = 3
        plt.rcParams["xtick.minor.size"] = 1.5
        plt.rcParams["ytick.minor.size"] = 1.5

        return True

    except Exception as e:
        warnings.warn(f"Failed to configure matplotlib: {e}")
        return False


def disable_font_warnings():
    """
    禁用matplotlib字体相关警告
    作为备用方案，在字体配置失败时使用

    Note: This suppresses common font missing warnings that may occur
    when specific fonts are not available on the system.
    """
    try:
        # 禁用字体警告 - more specific patterns
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="matplotlib",
            message=".*findfont.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="matplotlib.font_manager",
            message=".*Font family.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="matplotlib",
            message=".*Glyph.*missing.*",
        )

        # 设置matplotlib日志级别
        import logging

        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.ERROR)

        return True

    except Exception as e:
        warnings.warn(f"Could not disable font warnings: {e}")
        return False


# 科学期刊配色方案
JOURNAL_COLOR_SCHEMES = {
    "nature": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    "science": ["#0173b2", "#de8f05", "#029e73", "#cc78bc", "#ca9161", "#fbafe4"],
    "prl": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33"],
    "prb": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
    "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
}

# 预设图表尺寸（单位：英寸）
FIGURE_SIZES = {
    "single_column": (3.5, 2.6),  # 单栏图
    "double_column": (7.2, 5.4),  # 双栏图
    "square": (5.0, 5.0),  # 正方形
    "wide": (8.0, 4.0),  # 宽图
    "tall": (3.0, 6.0),  # 高图
    "poster": (12.0, 8.0),  # 海报尺寸
    "default": (6.4, 4.8),  # 默认尺寸
}

# 多子图布局预设
SUBPLOT_LAYOUTS = {
    "1x1": (1, 1),  # 单图
    "1x2": (1, 2),  # 1行2列
    "2x1": (2, 1),  # 2行1列
    "2x2": (2, 2),  # 2行2列
    "2x3": (2, 3),  # 2行3列
    "3x2": (3, 2),  # 3行2列
    "3x3": (3, 3),  # 3行3列
    "2x4": (2, 4),  # 2行4列
    "4x2": (4, 2),  # 4行2列
    "1x3": (1, 3),  # 1行3列
    "3x1": (3, 1),  # 3行1列
    "4x4": (4, 4),  # 4行4列
    "custom": None,  # 自定义布局
}

# 多子图间距预设（单位：英寸）
SUBPLOT_SPACING = {
    "tight": {"wspace": 0.1, "hspace": 0.1},  # 紧密间距
    "compact": {"wspace": 0.2, "hspace": 0.2},  # 紧凑间距
    "normal": {"wspace": 0.3, "hspace": 0.3},  # 正常间距
    "comfortable": {"wspace": 0.4, "hspace": 0.4},  # 舒适间距
    "spacious": {"wspace": 0.5, "hspace": 0.5},  # 宽松间距
}

# 多子图尺寸调整因子
SUBPLOT_SIZE_FACTORS = {
    "1x1": 1.0,
    "1x2": 2.0,
    "2x1": 2.0,
    "2x2": 2.0,
    "2x3": 3.0,
    "3x2": 3.0,
    "3x3": 3.0,
    "2x4": 4.0,
    "4x2": 4.0,
    "1x3": 3.0,
    "3x1": 3.0,
    "4x4": 4.0,
}

# 保存格式设置
SAVE_FORMATS = {
    "publication": {"dpi": 600, "format": "pdf", "quality": 100},
    "presentation": {"dpi": 150, "format": "png", "quality": 90},
    "web": {"dpi": 100, "format": "png", "quality": 85},
    "latex_pdf": {"dpi": 600, "format": "pdf", "quality": 100},
    "vector": {"dpi": None, "format": "svg", "quality": None},
}

# 图例尺寸预设
LEGEND_SIZE_PRESETS = {
    "small": {
        "fontsize": 8,
        "title_fontsize": 10,
        "handlelength": 1.0,
        "handletextpad": 0.4,
        "borderpad": 0.3,
        "labelspacing": 0.3,
        "columnspacing": 0.8,
    },
    "medium": {
        "fontsize": 10,
        "title_fontsize": 12,
        "handlelength": 1.5,
        "handletextpad": 0.8,
        "borderpad": 0.4,
        "labelspacing": 0.5,
        "columnspacing": 1.0,
    },
    "large": {
        "fontsize": 12,
        "title_fontsize": 14,
        "handlelength": 2.0,
        "handletextpad": 1.0,
        "borderpad": 0.5,
        "labelspacing": 0.6,
        "columnspacing": 1.2,
    },
    "poster": {
        "fontsize": 14,
        "title_fontsize": 16,
        "handlelength": 2.5,
        "handletextpad": 1.2,
        "borderpad": 0.6,
        "labelspacing": 0.8,
        "columnspacing": 1.5,
    },
}


# Legend parameter key mapping
_LEGEND_KEY_MAPPING = {
    "fontsize": "legend.fontsize",
    "title_fontsize": "legend.title_fontsize",
    "handlelength": "legend.handlelength",
    "handletextpad": "legend.handletextpad",
    "borderpad": "legend.borderpad",
    "labelspacing": "legend.labelspacing",
    "columnspacing": "legend.columnspacing",
}


def set_legend_size(preset="medium", **kwargs):
    """
    设置图例大小和样式

    Args:
        preset (str): 预设名称 ('small', 'medium', 'large', 'poster')
        **kwargs: 自定义图例参数，会覆盖预设值

    Returns:
        bool: 设置是否成功
    """
    try:
        if preset in LEGEND_SIZE_PRESETS:
            settings = LEGEND_SIZE_PRESETS[preset]

            # 应用预设设置
            for key, value in settings.items():
                if key in _LEGEND_KEY_MAPPING:
                    plt.rcParams[_LEGEND_KEY_MAPPING[key]] = value

            # 应用自定义参数（覆盖预设）
            for key, value in kwargs.items():
                if key in _LEGEND_KEY_MAPPING:
                    plt.rcParams[_LEGEND_KEY_MAPPING[key]] = value

            return True
        else:
            warnings.warn(f"Unknown legend size preset: {preset}")
            return False
    except Exception as e:
        warnings.warn(f"Failed to set legend size: {e}")
        return False


def set_color_scheme(scheme="default"):
    """
    设置配色方案

    Args:
        scheme (str): 配色方案名称 ('nature', 'science', 'prl', 'prb', 'default')

    Returns:
        bool: 设置是否成功
    """
    try:
        if scheme in JOURNAL_COLOR_SCHEMES:
            colors = JOURNAL_COLOR_SCHEMES[scheme]
            plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
            return True
        else:
            warnings.warn(f"Unknown color scheme: {scheme}")
            return False
    except Exception as e:
        warnings.warn(f"Failed to set color scheme: {e}")
        return False


def set_figure_size(size_name="default"):
    """
    设置图表尺寸

    Args:
        size_name (str): 尺寸名称 ('single_column', 'double_column', 'square', 'wide', 'tall', 'poster', 'default')

    Returns:
        tuple: (width, height) 图表尺寸（英寸）
    """
    if size_name in FIGURE_SIZES:
        return FIGURE_SIZES[size_name]
    else:
        warnings.warn(f"Unknown figure size: {size_name}, using default")
        return FIGURE_SIZES["default"]


def configure_for_latex():
    """
    配置LaTeX兼容性设置

    Returns:
        bool: 配置是否成功
    """
    try:
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Computer Modern Roman"]
        plt.rcParams["text.latex.preamble"] = (
            r"\usepackage{amsmath}\usepackage{amssymb}"
        )
        return True
    except Exception as e:
        warnings.warn(f"Failed to configure LaTeX: {e}")
        return False


def save_figure(fig, filename, purpose="publication", **kwargs):
    """
    保存图表，根据用途优化设置

    Args:
        fig: matplotlib figure对象
        filename (str): 保存文件名
        purpose (str): 用途 ('publication', 'presentation', 'web', 'latex_pdf', 'vector')
        **kwargs: 其他保存参数

    Returns:
        bool: 保存是否成功
    """
    try:
        if purpose in SAVE_FORMATS:
            settings = SAVE_FORMATS[purpose]
            save_kwargs = {
                "dpi": settings["dpi"],
                "bbox_inches": "tight",
                "pad_inches": 0.1,
                "facecolor": "white",
                "edgecolor": "none",
            }
            save_kwargs.update(kwargs)

            if settings["format"]:
                filename = f"{filename}.{settings['format']}"

            fig.savefig(filename, **save_kwargs)
            return True
        else:
            warnings.warn(f"Unknown save purpose: {purpose}")
            return False
    except Exception as e:
        warnings.warn(f"Failed to save figure: {e}")
        return False


def optimize_for_plot_type(plot_type):
    """
    针对不同图表类型进行优化设置

    Args:
        plot_type (str): 图表类型 ('line', 'scatter', 'bar', 'heatmap', 'contour')

    Returns:
        bool: 设置是否成功
    """
    try:
        if plot_type == "line":
            plt.rcParams["lines.linewidth"] = 2.0
            plt.rcParams["lines.markersize"] = 8
        elif plot_type == "scatter":
            plt.rcParams["lines.markersize"] = 10
            plt.rcParams["scatter.edgecolors"] = "black"
            plt.rcParams["scatter.linewidths"] = 0.5
        elif plot_type == "bar":
            plt.rcParams["axes.linewidth"] = 1.2
            plt.rcParams["patch.linewidth"] = 0.8
        elif plot_type == "heatmap":
            plt.rcParams["image.cmap"] = "viridis"
            plt.rcParams["image.interpolation"] = "nearest"
        elif plot_type == "contour":
            plt.rcParams["contour.linewidth"] = 1.5
        else:
            warnings.warn(f"Unknown plot type: {plot_type}")
            return False

        return True
    except Exception as e:
        warnings.warn(f"Failed to optimize for plot type: {e}")
        return False


def create_publication_figure(
    figsize="single_column",
    color_scheme="default",
    legend_size="medium"
):
    """
    创建适合发表的图表

    Args:
        figsize (str): 图表尺寸名称
        color_scheme (str): 配色方案名称
        legend_size (str): 图例大小预设 ('small', 'medium', 'large', 'poster')

    Returns:
        tuple: (fig, ax) matplotlib figure和axes对象
    """
    # 获取尺寸
    if isinstance(figsize, str):
        figsize = set_figure_size(figsize)

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 设置配色方案
    set_color_scheme(color_scheme)

    # 设置图例大小
    set_legend_size(legend_size)

    return fig, ax


def _get_axes_at(axes, i, j):
    """
    安全地获取axes数组中的特定位置的ax对象

    Args:
        axes: matplotlib axes对象（可能是一维或二维数组）
        i: 行索引
        j: 列索引

    Returns:
        matplotlib axes对象
    """
    # 如果axes是单个Axes对象（1x1布局）
    if not isinstance(axes, np.ndarray):
        return axes

    # 如果axes是一维数组（1xN或Nx1布局）
    if axes.ndim == 1:
        if axes.shape[0] == 1:  # 1x1
            return axes[0]
        elif i == 0:  # 1xN布局
            return axes[j]
        else:  # Nx1布局
            return axes[i]

    # 如果axes是二维数组（MxN布局）
    return axes[i, j]


def get_subplot_layout(layout_name):
    """
    获取多子图布局配置，支持正则匹配任意布局

    Args:
        layout_name (str): 布局名称 ('1x1', '1x2', '2x2', '1x4', '3x5', '10x3', etc.)

    Returns:
        tuple: (nrows, ncols) 行数和列数，如果布局格式无效则返回 (1, 1)
    """
    # 首先检查是否在预定义布局中
    if layout_name in SUBPLOT_LAYOUTS:
        return SUBPLOT_LAYOUTS[layout_name]

    # 使用正则匹配解析布局格式 (如 "1x4", "3x5", "10x2" 等)
    pattern = r"^(\d+)x(\d+)$"
    match = re.match(pattern, layout_name.strip())

    if match:
        nrows = int(match.group(1))
        ncols = int(match.group(2))

        # 检查行列数的合理性
        if nrows > 0 and ncols > 0 and nrows <= 20 and ncols <= 20:
            return (nrows, ncols)
        else:
            warnings.warn(f"Invalid subplot dimensions: {nrows}x{ncols}, using 1x1")
            return (1, 1)
    else:
        warnings.warn(f"Invalid layout format: {layout_name}, using 1x1")
        return (1, 1)


def calculate_subplot_figure_size(
    base_size,
    layout_name,
    spacing="normal"
):
    """
    计算多子图的整体尺寸

    Args:
        base_size (str or tuple): 基础尺寸名称或自定义尺寸 (width, height)
        layout_name (str): 布局名称
        spacing (str): 间距设置 ('tight', 'compact', 'normal', 'comfortable', 'spacious')

    Returns:
        tuple: (width, height) 调整后的图表尺寸
    """
    # 获取基础尺寸
    if isinstance(base_size, str):
        if base_size in FIGURE_SIZES:
            base_width, base_height = FIGURE_SIZES[base_size]
        else:
            base_width, base_height = FIGURE_SIZES["default"]
    else:
        base_width, base_height = base_size

    # 获取布局
    nrows, ncols = get_subplot_layout(layout_name)

    # 获取间距设置
    if spacing in SUBPLOT_SPACING:
        wspace, hspace = (
            SUBPLOT_SPACING[spacing]["wspace"],
            SUBPLOT_SPACING[spacing]["hspace"],
        )
    else:
        wspace, hspace = (
            SUBPLOT_SPACING["normal"]["wspace"],
            SUBPLOT_SPACING["normal"]["hspace"],
        )

    # 计算调整后的尺寸
    width = base_width * ncols + wspace * (ncols - 1)
    height = base_height * nrows + hspace * (nrows - 1)

    return (width, height)


def create_multi_subplot_figure(
    layout="2x2",
    base_size="single_column",
    spacing="normal",
    color_scheme="default",
    legend_size="medium",
    sharex=False,
    sharey=False,
    squeeze=False,
    subplot_kw=None,
    gridspec_kw=None,
):
    """
    创建多子图

    Args:
        layout (str): 布局名称 ('1x1', '1x2', '2x1', '2x2', '2x3', '3x2', '3x3', etc.)
        base_size (str or tuple): 基础尺寸名称或自定义尺寸
        spacing (str): 间距设置 ('tight', 'compact', 'normal', 'comfortable', 'spacious')
        color_scheme (str): 配色方案名称
        legend_size (str): 图例大小预设
        sharex (bool or str): 是否共享x轴
        sharey (bool or str): 是否共享y轴
        squeeze (bool): 是否压缩单行/单列的子图（默认False，保持2维数组）
        subplot_kw (dict): 传递给subplot的关键字参数
        gridspec_kw (dict): 传递给GridSpec的关键字参数

    Returns:
        tuple: (fig, axes) matplotlib figure和axes对象
    """
    # 获取布局
    nrows, ncols = get_subplot_layout(layout)

    # 计算整体尺寸
    figsize = calculate_subplot_figure_size(base_size, layout, spacing)

    # 准备gridspec参数
    if gridspec_kw is None:
        gridspec_kw = {}

    # 添加间距设置
    if spacing in SUBPLOT_SPACING:
        gridspec_kw.setdefault("wspace", SUBPLOT_SPACING[spacing]["wspace"])
        gridspec_kw.setdefault("hspace", SUBPLOT_SPACING[spacing]["hspace"])

    # 创建多子图
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        subplot_kw=subplot_kw,
        gridspec_kw=gridspec_kw,
    )

    # 设置配色方案
    set_color_scheme(color_scheme)

    # 设置图例大小
    set_legend_size(legend_size)

    return fig, axes


def add_reference_lines_to_subplots(
    axes,
    layout="2x2",
    y_values=None,
    x_values=None,
    y_styles=None,
    x_styles=None,
    y_colors=None,
    x_colors=None,
    y_labels=None,
    x_labels=None,
):
    """
    为所有子图添加参考线（如y=0的辅助线）

    Args:
        axes: matplotlib axes对象或axes数组
        layout (str): 布局名称
        y_values (float or list): y轴参考线的值（如[0]表示y=0线）
        x_values (float or list): x轴参考线的值
        y_styles (str or list): y轴参考线样式（如'--', ':', '-'）
        x_styles (str or list): x轴参考线样式
        y_colors (str or list): y轴参考线颜色
        x_colors (str or list): x轴参考线颜色
        y_labels (str or list): y轴参考线标签
        x_labels (str or list): x轴参考线标签

    Returns:
        bool: 设置是否成功
    """
    try:
        nrows, ncols = get_subplot_layout(layout)

        # 确保axes是数组格式
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            # Fix: Handle both list and numpy array inputs
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes]).reshape(-1, 1)
            elif axes.ndim == 1:
                axes = axes.reshape(-1, 1)

        # 设置默认值
        if y_values is None:
            y_values = [0]  # 默认添加y=0线
        if x_values is None:
            x_values = []
        if y_styles is None:
            y_styles = ["--"]
        if x_styles is None:
            x_styles = ["--"]
        if y_colors is None:
            y_colors = ["gray"]
        if x_colors is None:
            x_colors = ["gray"]
        if y_labels is None:
            y_labels = [None] * len(y_values)
        if x_labels is None:
            x_labels = [None] * len(x_values)

        # 确保样式、颜色、标签列表长度匹配
        if isinstance(y_styles, str):
            y_styles = [y_styles] * len(y_values)
        if isinstance(x_styles, str):
            x_styles = [x_styles] * len(x_values)
        if isinstance(y_colors, str):
            y_colors = [y_colors] * len(y_values)
        if isinstance(x_colors, str):
            x_colors = [x_colors] * len(x_values)
        if isinstance(y_labels, str):
            y_labels = [y_labels] * len(y_values)
        if isinstance(x_labels, str):
            x_labels = [x_labels] * len(x_values)

        # 为每个子图添加参考线
        for i in range(nrows):
            for j in range(ncols):
                try:
                    ax = _get_axes_at(axes, i, j)

                    # 添加y轴参考线
                    for y_val, style, color, label in zip(
                        y_values, y_styles, y_colors, y_labels
                    ):
                        ax.axhline(
                            y=y_val,
                            color=color,
                            linestyle=style,
                            alpha=0.7,
                            linewidth=1,
                            label=label,
                        )

                    # 添加x轴参考线
                    for x_val, style, color, label in zip(
                        x_values, x_styles, x_colors, x_labels
                    ):
                        ax.axvline(
                            x=x_val,
                            color=color,
                            linestyle=style,
                            alpha=0.7,
                            linewidth=1,
                            label=label,
                        )
                except Exception as e:
                    warnings.warn(
                        f"Failed to add reference lines to subplot ({i},{j}): {e}"
                    )
                    continue

        return True
    except Exception as e:
        warnings.warn(f"Failed to add reference lines: {e}")
        return False


def configure_subplot_grid(
    fig,
    axes,
    layout="2x2",
    title=None,
    subtitle=None,
    xlabel=None,
    ylabel=None,
    suptitle=None,
    suptitle_fontsize=16,
):
    """
    配置多子图的网格属性

    Args:
        fig: matplotlib figure对象
        axes: matplotlib axes对象或axes数组
        layout (str): 布局名称
        title (str or list): 主标题或标题列表
        subtitle (str or list): 副标题或副标题列表
        xlabel (str or list): x轴标签或标签列表
        ylabel (str or list): y轴标签或标签列表
        suptitle (str): 总标题
        suptitle_fontsize (int): 总标题字体大小

    Returns:
        bool: 配置是否成功
    """
    try:
        nrows, ncols = get_subplot_layout(layout)

        # 确保axes是数组格式
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = np.array([axes])
        elif ncols == 1:
            # Fix: Handle both list and numpy array inputs
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes]).reshape(-1, 1)
            elif axes.ndim == 1:
                axes = axes.reshape(-1, 1)

        # 设置总标题
        if suptitle:
            fig.suptitle(suptitle, fontsize=suptitle_fontsize, fontweight="bold")

        # 为每个子图设置属性
        for i in range(nrows):
            for j in range(ncols):
                try:
                    ax = _get_axes_at(axes, i, j)

                    # 设置标题
                    if title:
                        if isinstance(title, list):
                            idx = i * ncols + j
                            if idx < len(title) and title[idx]:
                                ax.set_title(title[idx])
                        else:
                            ax.set_title(title)

                    # 设置副标题
                    if subtitle:
                        if isinstance(subtitle, list):
                            idx = i * ncols + j
                            if idx < len(subtitle) and subtitle[idx]:
                                ax.set_title(
                                    subtitle[idx],
                                    loc="right",
                                    fontsize=10,
                                    style="italic",
                                )
                        else:
                            ax.set_title(
                                subtitle, loc="right", fontsize=10, style="italic"
                            )

                    # 设置x轴标签
                    if xlabel:
                        if isinstance(xlabel, list):
                            idx = i * ncols + j
                            if idx < len(xlabel) and xlabel[idx]:
                                ax.set_xlabel(xlabel[idx])
                        else:
                            ax.set_xlabel(xlabel)

                    # 设置y轴标签
                    if ylabel:
                        if isinstance(ylabel, list):
                            idx = i * ncols + j
                            if idx < len(ylabel) and ylabel[idx]:
                                ax.set_ylabel(ylabel[idx])
                        else:
                            ax.set_ylabel(ylabel)
                except Exception as e:
                    warnings.warn(f"Failed to configure subplot ({i},{j}): {e}")
                    continue

        return True
    except Exception as e:
        warnings.warn(f"Failed to configure subplot grid: {e}")
        return False


def create_shared_colorbar(
    fig,
    mappable,
    cbar_label=None,
    orientation="vertical",
    location="right",
    shrink=0.8,
    pad=0.05,
):
    """
    为多子图创建共享的颜色条

    Args:
        fig: matplotlib figure对象
        mappable: matplotlib mappable对象 (如imshow, contourf的返回值)
        cbar_label (str): 颜色条标签
        orientation (str): 方向 ('vertical' 或 'horizontal')
        location (str): 位置 ('right', 'left', 'bottom', 'top')
        shrink (float): 收缩因子 (0-1)
        pad (float): 间距 (inches)

    Returns:
        matplotlib Colorbar对象或None

    Note:
        Position format [left, bottom, width, height] in figure coordinates (0-1).
        These are default positions that work well for standard figures.
    """
    try:
        # 计算colorbar axes的位置 [left, bottom, width, height]
        if orientation == "vertical":
            if location == "right":
                # 右侧: [left=0.92, bottom=0.1, width=0.02, height=0.8]
                cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
            elif location == "left":
                # 左侧: [left=0.06, bottom=0.1, width=0.02, height=0.8]
                cax = fig.add_axes([0.06, 0.1, 0.02, 0.8])
            else:
                # 默认右侧
                cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        else:  # horizontal
            if location == "top":
                # 顶部: [left=0.1, bottom=0.92, width=0.8, height=0.02]
                cax = fig.add_axes([0.1, 0.92, 0.8, 0.02])
            elif location == "bottom":
                # 底部: [left=0.1, bottom=0.06, width=0.8, height=0.02]
                cax = fig.add_axes([0.1, 0.06, 0.8, 0.02])
            else:
                # 默认底部
                cax = fig.add_axes([0.1, 0.06, 0.8, 0.02])

        # 创建colorbar
        cbar = fig.colorbar(
            mappable, cax=cax, orientation=orientation, label=cbar_label
        )

        return cbar
    except Exception as e:
        warnings.warn(f"Failed to create shared colorbar: {e}")
        return None


def save_multi_subplot_figure(
    fig, filename, layout="2x2", purpose="publication", tight_layout=True, **kwargs
):
    """
    保存多子图

    Args:
        fig: matplotlib figure对象
        filename (str): 保存文件名
        layout (str): 布局名称（用于自动调整保存参数）
        purpose (str): 用途 ('publication', 'presentation', 'web', 'latex_pdf', 'vector')
        tight_layout (bool): 是否使用tight_layout
        **kwargs: 其他保存参数

    Returns:
        bool: 保存是否成功
    """
    try:
        # 自动调整布局
        if tight_layout:
            fig.tight_layout()

        # 根据布局调整保存参数
        nrows, ncols = get_subplot_layout(layout)

        # 多子图通常需要更高的分辨率
        if nrows * ncols > 4:
            if purpose in SAVE_FORMATS:
                settings = SAVE_FORMATS[purpose]
                if settings["dpi"] and settings["dpi"] < 300:
                    settings["dpi"] = 300  # 提高多子图的分辨率

        return save_figure(fig, filename, purpose, **kwargs)
    except Exception as e:
        warnings.warn(f"Failed to save multi-subplot figure: {e}")
        return False


def optimize_for_multi_subplot(plot_types, layout="2x2"):
    """
    为多子图的不同图表类型进行优化设置

    Args:
        plot_types (str or list): 图表类型或类型列表 ('line', 'scatter', 'bar', 'heatmap', 'contour')
        layout (str): 布局名称

    Returns:
        bool: 设置是否成功
    """
    try:
        nrows, ncols = get_subplot_layout(layout)

        if isinstance(plot_types, str):
            plot_types = [plot_types] * (nrows * ncols)

        # 为多子图优化字体大小
        font_scale = min(1.0, 2.0 / max(nrows, ncols))
        plt.rcParams["font.size"] = int(12 * font_scale)
        plt.rcParams["axes.labelsize"] = int(10 * font_scale)
        plt.rcParams["xtick.labelsize"] = int(8 * font_scale)
        plt.rcParams["ytick.labelsize"] = int(8 * font_scale)
        plt.rcParams["legend.fontsize"] = int(8 * font_scale)
        plt.rcParams["axes.titlesize"] = int(10 * font_scale)

        # 优化线条和标记大小
        marker_scale = min(1.0, 1.5 / max(nrows, ncols))
        plt.rcParams["lines.markersize"] = int(6 * marker_scale)
        plt.rcParams["lines.linewidth"] = max(0.5, 1.5 * marker_scale)

        return True
    except Exception as e:
        warnings.warn(f"Failed to optimize for multi-subplot: {e}")
        return False


def init_publication_style(
    color_scheme="default",
    figsize="single_column",
    legend_size="medium",
    use_latex=False,
):
    """
    一键初始化科学发表级别的matplotlib配置

    这是快速设置发表级别图表样式的便捷函数，组合了多个配置步骤。

    Args:
        color_scheme (str): 配色方案名称 ('nature', 'science', 'prl', 'prb', 'default')
        figsize (str): 图表尺寸名称 ('single_column', 'double_column', 'square', 'wide', 'tall', 'poster', 'default')
        legend_size (str): 图例大小预设 ('small', 'medium', 'large', 'poster')
        use_latex (bool): 是否启用LaTeX渲染（需要系统安装LaTeX）

    Returns:
        tuple: (width, height) 图表尺寸（英寸），如果配置失败则返回 None

    Example:
        >>> # 基本使用
        >>> figsize = init_publication_style()
        >>> fig, ax = plt.subplots(figsize=figsize)
        >>>
        >>> # 自定义配置
        >>> figsize = init_publication_style(
        ...     color_scheme="nature",
        ...     figsize="double_column",
        ...     legend_size="small"
        ... )
        >>> fig, ax = plt.subplots(figsize=figsize)
    """
    # 配置基础matplotlib设置
    success = configure_matplotlib_for_publication()
    if not success:
        warnings.warn("Failed to configure matplotlib for publication")

    # 设置配色方案
    success = set_color_scheme(color_scheme)
    if not success:
        warnings.warn(f"Failed to set color scheme: {color_scheme}")

    # 设置图例大小
    success = set_legend_size(legend_size)
    if not success:
        warnings.warn(f"Failed to set legend size: {legend_size}")

    # 可选：启用LaTeX
    if use_latex:
        success = configure_for_latex()
        if not success:
            warnings.warn("Failed to configure LaTeX rendering")

    # 返回图表尺寸
    return set_figure_size(figsize)
