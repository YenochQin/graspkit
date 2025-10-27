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
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from cycler import cycler

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

# 自动配置matplotlib（当模块被导入时）
if __name__ != "__main__":
    success = configure_matplotlib_for_publication()
    if not success:
        disable_font_warnings()
