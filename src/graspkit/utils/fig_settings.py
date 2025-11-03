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

import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from cycler import cycler
import numpy as np

def configure_matplotlib_for_publication():
    """
    配置matplotlib用于科学发表的图表样式
    
    Returns:
        bool: 配置是否成功
    """
    try:
        # 高质量图形设置
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.1
        
        # 字体设置（英文）
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.titlesize'] = 16
        
        # 线条和标记设置
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['lines.markersize'] = 6
        plt.rcParams['patch.linewidth'] = 0.5
        
        # 坐标轴设置
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linewidth'] = 0.5
        
        # 负号显示修复
        plt.rcParams['axes.unicode_minus'] = False
        
        # 图例设置
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.framealpha'] = 0.9
        plt.rcParams['legend.fancybox'] = True
        plt.rcParams['legend.shadow'] = False
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['legend.title_fontsize'] = 12
        plt.rcParams['legend.borderaxespad'] = 0.5
        plt.rcParams['legend.borderpad'] = 0.4
        plt.rcParams['legend.columnspacing'] = 1.0
        plt.rcParams['legend.handletextpad'] = 0.8
        plt.rcParams['legend.handlelength'] = 1.5
        plt.rcParams['legend.labelspacing'] = 0.5
        
        # 刻度设置
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.major.size'] = 3
        plt.rcParams['ytick.major.size'] = 3
        plt.rcParams['xtick.minor.size'] = 1.5
        plt.rcParams['ytick.minor.size'] = 1.5
        
        return True
        
    except Exception as e:
        warnings.warn(f"Failed to configure matplotlib: {e}")
        return False

def disable_font_warnings():
    """
    禁用matplotlib字体相关警告
    作为备用方案，在字体配置失败时使用
    """
    try:
        # 禁用字体警告
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        warnings.filterwarnings('ignore', message='.*font.*')
        warnings.filterwarnings('ignore', message='.*Glyph.*missing.*')
        
        # 设置matplotlib日志级别
        import logging
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.ERROR)
        
        return True
        
    except Exception as e:
        print(f"Warning: Could not disable font warnings: {e}")
        return False

# 科学期刊配色方案
JOURNAL_COLOR_SCHEMES = {
    'nature': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
    'science': ['#0173b2', '#de8f05', '#029e73', '#cc78bc', '#ca9161', '#fbafe4'],
    'prl': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33'],
    'prb': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
    'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
}

# 预设图表尺寸（单位：英寸）
FIGURE_SIZES = {
    'single_column': (3.5, 2.6),      # 单栏图
    'double_column': (7.2, 5.4),      # 双栏图
    'square': (5.0, 5.0),             # 正方形
    'wide': (8.0, 4.0),               # 宽图
    'tall': (3.0, 6.0),               # 高图
    'poster': (12.0, 8.0),            # 海报尺寸
    'default': (6.4, 4.8)             # 默认尺寸
}

# 多子图布局预设
SUBPLOT_LAYOUTS = {
    '1x1': (1, 1),                    # 单图
    '1x2': (1, 2),                    # 1行2列
    '2x1': (2, 1),                    # 2行1列
    '2x2': (2, 2),                    # 2行2列
    '2x3': (2, 3),                    # 2行3列
    '3x2': (3, 2),                    # 3行2列
    '3x3': (3, 3),                    # 3行3列
    '2x4': (2, 4),                    # 2行4列
    '4x2': (4, 2),                    # 4行2列
    '1x3': (1, 3),                    # 1行3列
    '3x1': (3, 1),                    # 3行1列
    '4x4': (4, 4),                    # 4行4列
    'custom': None                    # 自定义布局
}

# 多子图间距预设（单位：英寸）
SUBPLOT_SPACING = {
    'tight': {'wspace': 0.1, 'hspace': 0.1},      # 紧密间距
    'compact': {'wspace': 0.2, 'hspace': 0.2},    # 紧凑间距
    'normal': {'wspace': 0.3, 'hspace': 0.3},     # 正常间距
    'comfortable': {'wspace': 0.4, 'hspace': 0.4}, # 舒适间距
    'spacious': {'wspace': 0.5, 'hspace': 0.5}    # 宽松间距
}

# 多子图尺寸调整因子
SUBPLOT_SIZE_FACTORS = {
    '1x1': 1.0,
    '1x2': 2.0,
    '2x1': 2.0,
    '2x2': 2.0,
    '2x3': 3.0,
    '3x2': 3.0,
    '3x3': 3.0,
    '2x4': 4.0,
    '4x2': 4.0,
    '1x3': 3.0,
    '3x1': 3.0,
    '4x4': 4.0
}

# 保存格式设置
SAVE_FORMATS = {
    'publication': {'dpi': 600, 'format': 'pdf', 'quality': 100},
    'presentation': {'dpi': 150, 'format': 'png', 'quality': 90},
    'web': {'dpi': 100, 'format': 'png', 'quality': 85},
    'latex_pdf': {'dpi': 600, 'format': 'pdf', 'quality': 100},
    'vector': {'dpi': None, 'format': 'svg', 'quality': None}
}

# 图例尺寸预设
LEGEND_SIZE_PRESETS = {
    'small': {
        'fontsize': 8,
        'title_fontsize': 10,
        'handlelength': 1.0,
        'handletextpad': 0.4,
        'borderpad': 0.3,
        'labelspacing': 0.3,
        'columnspacing': 0.8
    },
    'medium': {
        'fontsize': 10,
        'title_fontsize': 12,
        'handlelength': 1.5,
        'handletextpad': 0.8,
        'borderpad': 0.4,
        'labelspacing': 0.5,
        'columnspacing': 1.0
    },
    'large': {
        'fontsize': 12,
        'title_fontsize': 14,
        'handlelength': 2.0,
        'handletextpad': 1.0,
        'borderpad': 0.5,
        'labelspacing': 0.6,
        'columnspacing': 1.2
    },
    'poster': {
        'fontsize': 14,
        'title_fontsize': 16,
        'handlelength': 2.5,
        'handletextpad': 1.2,
        'borderpad': 0.6,
        'labelspacing': 0.8,
        'columnspacing': 1.5
    }
}

def set_legend_size(preset='medium', **kwargs):
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
                if key == 'fontsize':
                    plt.rcParams['legend.fontsize'] = value
                elif key == 'title_fontsize':
                    plt.rcParams['legend.title_fontsize'] = value
                elif key == 'handlelength':
                    plt.rcParams['legend.handlelength'] = value
                elif key == 'handletextpad':
                    plt.rcParams['legend.handletextpad'] = value
                elif key == 'borderpad':
                    plt.rcParams['legend.borderpad'] = value
                elif key == 'labelspacing':
                    plt.rcParams['legend.labelspacing'] = value
                elif key == 'columnspacing':
                    plt.rcParams['legend.columnspacing'] = value

            # 应用自定义参数（覆盖预设）
            for key, value in kwargs.items():
                if key in ['fontsize', 'title_fontsize', 'handlelength', 'handletextpad',
                          'borderpad', 'labelspacing', 'columnspacing']:
                    rcparam_key = f'legend.{key}'
                    plt.rcParams[rcparam_key] = value

            return True
        else:
            warnings.warn(f"Unknown legend size preset: {preset}")
            return False
    except Exception as e:
        warnings.warn(f"Failed to set legend size: {e}")
        return False

def set_color_scheme(scheme='default'):
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
            plt.rcParams['axes.prop_cycle'] = mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)
            return True
        else:
            warnings.warn(f"Unknown color scheme: {scheme}")
            return False
    except Exception as e:
        warnings.warn(f"Failed to set color scheme: {e}")
        return False

def set_figure_size(size_name='default'):
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
        return FIGURE_SIZES['default']

def configure_for_latex():
    """
    配置LaTeX兼容性设置
    
    Returns:
        bool: 配置是否成功
    """
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{amssymb}'
        return True
    except Exception as e:
        warnings.warn(f"Failed to configure LaTeX: {e}")
        return False

def save_figure(fig, filename, purpose='publication', **kwargs):
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
                'dpi': settings['dpi'],
                'bbox_inches': 'tight',
                'pad_inches': 0.1,
                'facecolor': 'white',
                'edgecolor': 'none'
            }
            save_kwargs.update(kwargs)
            
            if settings['format']:
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
        if plot_type == 'line':
            plt.rcParams['lines.linewidth'] = 2.0
            plt.rcParams['lines.markersize'] = 8
        elif plot_type == 'scatter':
            plt.rcParams['lines.markersize'] = 10
            plt.rcParams['scatter.edgecolors'] = 'black'
            plt.rcParams['scatter.linewidths'] = 0.5
        elif plot_type == 'bar':
            plt.rcParams['axes.linewidth'] = 1.2
            plt.rcParams['patch.linewidth'] = 0.8
        elif plot_type == 'heatmap':
            plt.rcParams['image.cmap'] = 'viridis'
            plt.rcParams['image.interpolation'] = 'nearest'
        elif plot_type == 'contour':
            plt.rcParams['contour.linewidth'] = 1.5
        else:
            warnings.warn(f"Unknown plot type: {plot_type}")
            return False
        
        return True
    except Exception as e:
        warnings.warn(f"Failed to optimize for plot type: {e}")
        return False

def create_publication_figure(figsize='single_column', color_scheme='default', legend_size='medium'):
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

def get_subplot_layout(layout_name):
    """
    获取多子图布局配置

    Args:
        layout_name (str): 布局名称 ('1x1', '1x2', '2x2', '2x3', '3x2', '3x3', etc.)

    Returns:
        tuple: (nrows, ncols) 行数和列数，如果布局不存在则返回 (1, 1)
    """
    if layout_name in SUBPLOT_LAYOUTS:
        return SUBPLOT_LAYOUTS[layout_name]
    else:
        warnings.warn(f"Unknown subplot layout: {layout_name}, using 1x1")
        return SUBPLOT_LAYOUTS['1x1']

def calculate_subplot_figure_size(base_size, layout_name, spacing='normal'):
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
            base_width, base_height = FIGURE_SIZES['default']
    else:
        base_width, base_height = base_size

    # 获取布局
    nrows, ncols = get_subplot_layout(layout_name)

    # 获取间距设置
    if spacing in SUBPLOT_SPACING:
        wspace, hspace = SUBPLOT_SPACING[spacing]['wspace'], SUBPLOT_SPACING[spacing]['hspace']
    else:
        wspace, hspace = SUBPLOT_SPACING['normal']['wspace'], SUBPLOT_SPACING['normal']['hspace']

    # 计算调整后的尺寸
    width = base_width * ncols + wspace * (ncols - 1)
    height = base_height * nrows + hspace * (nrows - 1)

    return (width, height)

def create_multi_subplot_figure(layout='2x2', base_size='single_column',
                               spacing='normal', color_scheme='default',
                               legend_size='medium', sharex=False, sharey=False,
                               squeeze=False, subplot_kw=None, gridspec_kw=None):
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
        gridspec_kw.setdefault('wspace', SUBPLOT_SPACING[spacing]['wspace'])
        gridspec_kw.setdefault('hspace', SUBPLOT_SPACING[spacing]['hspace'])

    # 创建多子图
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                            sharex=sharex, sharey=sharey, squeeze=squeeze,
                            subplot_kw=subplot_kw, gridspec_kw=gridspec_kw)

    # 设置配色方案
    set_color_scheme(color_scheme)

    # 设置图例大小
    set_legend_size(legend_size)

    return fig, axes

def add_reference_lines_to_subplots(axes, layout='2x2', y_values=None, x_values=None,
                                   y_styles=None, x_styles=None, y_colors=None, x_colors=None,
                                   y_labels=None, x_labels=None):
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
            axes = np.array([axes]).reshape(-1, 1)

        # 设置默认值
        if y_values is None:
            y_values = [0]  # 默认添加y=0线
        if x_values is None:
            x_values = []
        if y_styles is None:
            y_styles = ['--']
        if x_styles is None:
            x_styles = ['--']
        if y_colors is None:
            y_colors = ['gray']
        if x_colors is None:
            x_colors = ['gray']
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
                ax = axes[i, j]

                # 添加y轴参考线
                for y_val, style, color, label in zip(y_values, y_styles, y_colors, y_labels):
                    ax.axhline(y=y_val, color=color, linestyle=style, alpha=0.7,
                             linewidth=1, label=label)

                # 添加x轴参考线
                for x_val, style, color, label in zip(x_values, x_styles, x_colors, x_labels):
                    ax.axvline(x=x_val, color=color, linestyle=style, alpha=0.7,
                             linewidth=1, label=label)

        return True
    except Exception as e:
        warnings.warn(f"Failed to add reference lines: {e}")
        return False

def configure_subplot_grid(fig, axes, layout='2x2',
                          title=None, subtitle=None,
                          xlabel=None, ylabel=None,
                          suptitle=None, suptitle_fontsize=16):
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
            axes = np.array([axes]).reshape(-1, 1)

        # 设置总标题
        if suptitle:
            fig.suptitle(suptitle, fontsize=suptitle_fontsize, fontweight='bold')

        # 为每个子图设置属性
        for i in range(nrows):
            for j in range(ncols):
                ax = axes[i, j]

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
                            ax.set_title(subtitle[idx], loc='right', fontsize=10, style='italic')
                    else:
                        ax.set_title(subtitle, loc='right', fontsize=10, style='italic')

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

        return True
    except Exception as e:
        warnings.warn(f"Failed to configure subplot grid: {e}")
        return False

def create_shared_colorbar(fig, axes, cbar_label=None, orientation='vertical',
                          location='right', shrink=0.8, pad=0.05):
    """
    为多子图创建共享的颜色条

    Args:
        fig: matplotlib figure对象
        axes: matplotlib axes对象或axes数组
        cbar_label (str): 颜色条标签
        orientation (str): 方向 ('vertical' 或 'horizontal')
        location (str): 位置 ('right', 'left', 'bottom', 'top')
        shrink (float): 收缩因子
        pad (float): 间距

    Returns:
        matplotlib Colorbar对象或None
    """
    try:
        import matplotlib.cm as cm
        from matplotlib.colorbar import Colorbar

        # 创建colorbar axes
        if orientation == 'vertical':
            if location in ['right', 'left']:
                cax = fig.add_axes([0.92, 0.1, 0.02, 0.8]) if location == 'right' else fig.add_axes([0.06, 0.1, 0.02, 0.8])
            else:
                cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        else:  # horizontal
            if location in ['top', 'bottom']:
                cax = fig.add_axes([0.1, 0.92, 0.8, 0.02]) if location == 'top' else fig.add_axes([0.1, 0.06, 0.8, 0.02])
            else:
                cax = fig.add_axes([0.1, 0.06, 0.8, 0.02])

        # 这里需要根据实际的contourf或imshow对象来创建colorbar
        # 由于没有具体的mappable对象，这里返回cax让用户自行创建colorbar
        return cax
    except Exception as e:
        warnings.warn(f"Failed to create shared colorbar: {e}")
        return None

def save_multi_subplot_figure(fig, filename, layout='2x2', purpose='publication',
                            tight_layout=True, **kwargs):
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
                if settings['dpi'] and settings['dpi'] < 300:
                    settings['dpi'] = 300  # 提高多子图的分辨率

        return save_figure(fig, filename, purpose, **kwargs)
    except Exception as e:
        warnings.warn(f"Failed to save multi-subplot figure: {e}")
        return False

def optimize_for_multi_subplot(plot_types, layout='2x2'):
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
        plt.rcParams['font.size'] = int(12 * font_scale)
        plt.rcParams['axes.labelsize'] = int(10 * font_scale)
        plt.rcParams['xtick.labelsize'] = int(8 * font_scale)
        plt.rcParams['ytick.labelsize'] = int(8 * font_scale)
        plt.rcParams['legend.fontsize'] = int(8 * font_scale)
        plt.rcParams['axes.titlesize'] = int(10 * font_scale)

        # 优化线条和标记大小
        marker_scale = min(1.0, 1.5 / max(nrows, ncols))
        plt.rcParams['lines.markersize'] = int(6 * marker_scale)
        plt.rcParams['lines.linewidth'] = max(0.5, 1.5 * marker_scale)

        return True
    except Exception as e:
        warnings.warn(f"Failed to optimize for multi-subplot: {e}")
        return False

def demo_multi_subplot_usage():
    """
    多子图功能使用示例

    演示如何使用多子图相关的功能：
    - 创建2x2布局的多子图
    - 设置不同的间距和尺寸
    - 配置标题和标签
    - 保存多子图

    Returns:
        bool: 示例是否成功执行
    """
    try:
        # 示例1: 创建2x2布局的多子图
        fig1, axes1 = create_multi_subplot_figure(
            layout='2x2',
            base_size='single_column',
            spacing='normal',
            color_scheme='nature'
        )

        # 为每个子图添加示例数据
        x = np.linspace(0, 10, 100)

        # 子图1: 线图
        axes1[0, 0].plot(x, np.sin(x), label='sin(x)')
        axes1[0, 0].plot(x, np.cos(x), label='cos(x)')
        axes1[0, 0].legend()

        # 子图2: 散点图
        axes1[0, 1].scatter(x[:50], np.random.random(50), alpha=0.6)

        # 子图3: 柱状图
        categories = ['A', 'B', 'C', 'D']
        values = np.random.random(4)
        axes1[1, 0].bar(categories, values)

        # 子图4: 热图示例
        data = np.random.random((10, 10))
        im = axes1[1, 1].imshow(data, cmap='viridis')

        # 配置多子图网格
        configure_subplot_grid(
            fig1, axes1, layout='2x2',
            title=['Sine & Cosine', 'Random Scatter', 'Bar Chart', 'Heatmap'],
            suptitle='Multi-Subplot Demo'
        )

        # 为所有子图添加y=0的参考线
        add_reference_lines_to_subplots(
            axes1, layout='2x2',
            y_values=[0],  # 添加y=0线
            y_styles='--',  # 虚线样式
            y_colors='gray',  # 灰色
            y_labels='y=0'  # 标签（可选）
        )

        # 为热图子图添加颜色条（仅针对该子图，匹配热图高度）
        cbar = fig1.colorbar(im, ax=axes1[1, 1], shrink=0.99, pad=0.05, aspect=30)
        cbar.set_label('Values', fontsize=10)

        # 保存多子图
        save_multi_subplot_figure(fig1, 'demo_multi_subplot', layout='2x2', purpose='publication')
        plt.close(fig1)

        # 示例2: 1x3布局的不同间距
        fig2, axes2 = create_multi_subplot_figure(
            layout='1x3',
            base_size='double_column',
            spacing='compact',
            color_scheme='science'
        )

        # 配置标题
        configure_subplot_grid(
            fig2, axes2, layout='1x3',
            title=['Plot 1', 'Plot 2', 'Plot 3'],
            suptitle='1x3 Layout with Compact Spacing'
        )

        save_multi_subplot_figure(fig2, 'demo_1x3_subplot', layout='1x3', purpose='presentation')
        plt.close(fig2)

        print("Multi-subplot demo completed successfully!")
        return True

    except Exception as e:
        warnings.warn(f"Demo failed: {e}")
        return False


# 自动配置matplotlib（当模块被导入时）
if __name__ != "__main__":
    success = configure_matplotlib_for_publication()
    if not success:
        disable_font_warnings()

# 主函数入口（用于运行示例）
if __name__ == "__main__":
    print("Running figure settings demo...")
    print(f"Available subplot layouts: {list(SUBPLOT_LAYOUTS.keys())}")
    print(f"Available spacing presets: {list(SUBPLOT_SPACING.keys())}")
    print(f"Available figure sizes: {list(FIGURE_SIZES.keys())}")
    print(f"Available color schemes: {list(JOURNAL_COLOR_SCHEMES.keys())}")
    print()

    # 运行多子图示例
    demo_multi_subplot_usage()

    print()
    print("Figure settings module loaded successfully!")
    print("Use create_multi_subplot_figure() to create multi-subplot layouts.")
    print("Use configure_subplot_grid() to configure subplot properties.")
    print("Use save_multi_subplot_figure() to save multi-subplot figures.")
