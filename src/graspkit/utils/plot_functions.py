"""
图表可视化工具模块

提供各种专业的数据可视化图表函数，包括：
- 双轴柱状图（用于展示占比和贡献值）
- 玫瑰图
- 其他专业图表

所有图表函数都集成了fig_settings.py中的专业发表级图表设置。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Tuple
import warnings


from .fig_settings import (
        configure_matplotlib_for_publication,
        set_figure_size,
        set_color_scheme,
        set_legend_size,
        save_figure,
        optimize_for_plot_type,
        create_multi_subplot_figure,
        add_reference_lines_to_subplots,
        configure_subplot_grid
    )


def inter_coupling_channel_bar(categories, quantity, sum_squared_ci, colors=None,
                        figsize='double_column', color_scheme='nature',
                        min_threshold=10, fontsize_scale=1.0):
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
    def format_category(cat):
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
            aggregated_count = np.sum(~mask_above_threshold)

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
    line = ax2.plot(range(len(categories)), sum_squared_ci, 'o-',
                   color='darkred', linewidth=2, markersize=6,
                   alpha=0.8, label='Contribution Value', markerfacecolor='white',
                   markeredgewidth=2, markeredgecolor='darkred')

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
    ax1.tick_params(axis='y', which='both', left=True, right=False,
                   labelleft=True, labelright=False)
    ax2.tick_params(axis='y', which='both', left=False, right=True,
                   labelleft=False, labelright=True)

    # 设置X轴标签
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=tick_fontsize)

    # 设置图例
    legend_fontsize = int(10 * fontsize_scale)
    ax1.legend(handles=line, loc='upper right', fontsize=legend_fontsize)

    # 添加网格
    ax1.grid(True, alpha=0.3, axis='y')

    # 调整布局
    plt.tight_layout()

    return fig, ax1, ax2


def auto_plot_wavefunction_comparison(
                                    data_list: list,
                                    column_names: list[str],
                                    x_col: str = 'r(a.u)',
                                    labels: list[str] | None = None,
                                    layout: str = '2x4',
                                    alpha: float = 0.75,
                                    max_x: int | None = None,
                                    xscale: str = 'symlog',
                                    linthresh: int = 1,
                                    suptitle: str = 'Wavefunction Comparison',
                                    colors  = None,
                                    linestyles: list[str] | None = None,
                                    xlabel: str | None = None,
                                    ylabel: str | None = None
                                ):
    """
    自动绘制波函数对比图的通用函数，支持多个DataFrame对比

    Args:
        data_list: 数据集列表，包含多个DataFrame [data1, data2, data3, ...]
        column_names: 要绘制的列名列表，包含成对的P和Q分量，如 ['P(4p-)', 'Q(4p-)', 'P(4p )', 'Q(4p )']
        x_col: x轴数据的列名 (默认: 'r(a.u)')
        labels: 数据集标签列表，如 ['old', 'new', 'modified'] (默认: ['Data 1', 'Data 2', ...])
        layout: 子图布局 (默认: '2x4')
        alpha: 透明度 (默认: 0.75)
        max_x: x轴最大值，如果为None则自动计算
        xscale: x轴比例类型 (默认: 'symlog')
        linthresh: symlog的线性阈值 (默认: 1)
        suptitle: 总标题 (默认: 'Wavefunction Comparison')
        colors: 线条颜色列表，如 ['blue', 'red', 'green'] (默认: 自动分配)
        linestyles: 线条样式列表，如 ['-', '--', '-.'] (默认: 全为实线)
        xlabel: x轴标签，只在最底行显示 (默认: None)
        ylabel: y轴标签，只在最左列显示 (默认: None)

    Returns:
        fig, axes: matplotlib的figure和axes对象
    """

    def extract_orbital_name(col_name):
        """
        从列名中提取轨道名称（括号内的字符）

        Args:
            col_name (str): 列名，如 'P(4p-)', 'Q(4p-)'

        Returns:
            str: 轨道名称，如 '4p-'
        """
        import re
        match = re.search(r'\((.*?)\)', col_name)
        if match:
            return match.group(1)
        return col_name

    def group_columns_by_orbital(column_names):
        """
        将列名按轨道分组，返回每组对应的P和Q列名以及轨道名称

        Args:
            column_names: 列名列表，如 ['P(4p-)', 'Q(4p-)', 'P(4p )', 'Q(4p )']

        Returns:
            list: 包含轨道信息的列表，每个元素为 {'orbital': str, 'p_col': str, 'q_col': str}
        """
        # 首先提取所有轨道名称
        orbital_map = {}
        for col_name in column_names:
            orbital = extract_orbital_name(col_name)
            if orbital not in orbital_map:
                orbital_map[orbital] = {'p_col': None, 'q_col': None}

            # 根据列名确定是P还是Q分量
            if col_name.startswith('P(') or col_name.startswith('P '):
                orbital_map[orbital]['p_col'] = col_name
            elif col_name.startswith('Q(') or col_name.startswith('Q '):
                orbital_map[orbital]['q_col'] = col_name

        # 转换为列表格式，并验证每个轨道都有P和Q分量
        orbital_groups = []
        for orbital, cols in orbital_map.items():
            if cols['p_col'] is not None and cols['q_col'] is not None:
                orbital_groups.append({
                    'orbital': orbital,
                    'p_col': cols['p_col'],
                    'q_col': cols['q_col']
                })
            else:
                warnings.warn(f"Orbital {orbital} is missing P or Q component, skipping...")

        return orbital_groups

    # 创建多子图
    fig, axes = create_multi_subplot_figure(
        layout=layout,
        base_size='single_column',
        spacing='normal',
        color_scheme='nature'
    )

    # 验证输入
    if not isinstance(data_list, (list, tuple)):
        raise ValueError("data_list must be a list or tuple of DataFrames")

    n_datasets = len(data_list)

    # 设置默认标签
    if labels is None:
        labels = [f'Data {i+1}' for i in range(n_datasets)]
    elif len(labels) != n_datasets:
        raise ValueError(f"Number of labels ({len(labels)}) must match number of datasets ({n_datasets})")

    # 设置默认颜色和线条样式
    if colors is None:
        # 扩展的颜色列表，包含足够多的颜色来支持大量数据集
        extended_colors = [
            # matplotlib默认颜色
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            # 更多蓝色系
            '#17becf', '#0080ff', '#0066cc', '#004499', '#002266', '#001133',
            # 更多红色系
            '#e377c2', '#ff1493', '#dc143c', '#b22222', '#8b0000', '#800000',
            # 更多绿色系
            '#7fff00', '#32cd32', '#228b22', '#006400', '#004000', '#002000',
            # 更多橙色/黄色系
            '#ffd700', '#ffb347', '#ff8c00', '#ff6347', '#ff4500', '#ff0000',
            # 更多紫色系
            '#9370db', '#8a2be2', '#800080', '#4b0082', '#6a0dad', '#483d8b',
            # 更多青色/青绿色系
            '#40e0d0', '#00ced1', '#008b8b', '#008080', '#20b2aa', '#5f9ea0',
            # 更多棕色系
            '#daa520', '#b8860b', '#cd853f', '#8b4513', '#a0522d', '#d2691e',
            # 更多灰色系
            '#708090', '#778899', '#696969', '#2f4f4f', '#556b2f', '#8b7355',
            # 更多粉色系
            '#ffb6c1', '#ffc0cb', '#ff69b4', '#ff1493', '#c71585', '#db7093',
            # 更多特殊颜色
            '#ffdead', '#f0e68c', '#dda0dd', '#ee82ee', '#fa8072', '#ffa07a',
            '#20b2aa', '#87ceeb', '#87cefa', '#4682b4', '#b0c4de', '#add8e6'
        ]

        # 如果数据集数量超过扩展颜色列表，则循环使用
        if n_datasets > len(extended_colors):
            colors = (extended_colors * (n_datasets // len(extended_colors) + 1))[:n_datasets]
        else:
            colors = extended_colors[:n_datasets]
    elif len(colors) < n_datasets:
        # 如果提供的颜色不够，循环使用
        colors = (colors * (n_datasets // len(colors) + 1))[:n_datasets]

    if linestyles is None:
        # 扩展的线条样式列表，提供更多样化的线条样式
        extended_linestyles = [
            '-', '--', '-.', ':',      # 基本线条样式
            (0, (3, 1, 1, 1)),        # 密集点划线
            (0, (5, 1, 1, 1)),        # 稀疏点划线
            (0, (3, 1, 3, 1, 1, 1)),  # 复杂点划线
            (0, (1, 1)),              # 密点线
            (0, (2, 2)),              # 中等点线
            (0, (5, 5)),              # 稀疏点线
        ]

        # 如果数据集数量超过线条样式列表，则循环使用
        if n_datasets > len(extended_linestyles):
            linestyles = (extended_linestyles * (n_datasets // len(extended_linestyles) + 1))[:n_datasets]
        else:
            linestyles = extended_linestyles[:n_datasets]
    elif len(linestyles) < n_datasets:
        # 如果提供的样式不够，循环使用
        linestyles = (linestyles * (n_datasets // len(linestyles) + 1))[:n_datasets]

    # 计算x轴范围
    if max_x is None:
        last_x = np.sqrt(data_list[0][x_col].iloc[-1])
        max_x = int(np.ceil(last_x / 10)) * 10

    # 获取布局信息
    nrows = int(layout.split('x')[0])
    ncols = int(layout.split('x')[1])

    # 按轨道分组列名
    orbital_groups = group_columns_by_orbital(column_names)
    n_orbitals = len(orbital_groups)

    if n_orbitals == 0:
        raise ValueError("No valid orbital groups found in column_names")

    # 自动绘制所有轨道的 P²+Q² 图
    for i, orbital_group in enumerate(orbital_groups):
        if i >= nrows * ncols:  # 超出子图数量则跳过
            break

        # 计算子图位置
        row = i // ncols
        col = i % ncols

        orbital_name = orbital_group['orbital']
        p_col = orbital_group['p_col']
        q_col = orbital_group['q_col']

        # 绘制所有数据集的 P²+Q²
        for j, data in enumerate(data_list):
            # 验证列是否存在
            if p_col not in data.columns or q_col not in data.columns:
                warnings.warn(f"Columns {p_col} or {q_col} not found in dataset {j+1}, skipping...")
                continue

            # 计算 P²+Q²
            p_squared_plus_q_squared = data[p_col]**2 + data[q_col]**2

            axes[row, col].plot(np.sqrt(data[x_col]), p_squared_plus_q_squared,
                               alpha=alpha,
                               label=labels[j],
                               color=colors[j],
                               linestyle=linestyles[j])

        # 设置x轴
        axes[row, col].set_xlim(0, max_x)
        if xscale == 'symlog':
            axes[row, col].set_xscale('symlog', linthresh=linthresh)
        elif xscale == 'log':
            axes[row, col].set_xscale('log')

        # 添加图例
        axes[row, col].legend()

    # 添加参考线
    add_reference_lines_to_subplots(
        axes, layout=layout,
        y_values=[0],
        y_styles='--',
        y_colors='gray',
        y_labels=None
    )

    # 配置子图网格，使用轨道名称作为标题
    orbital_titles = [group['orbital'] for group in orbital_groups[:nrows*ncols]]
    configure_subplot_grid(
        fig, axes, layout=layout,
        title=orbital_titles,
        suptitle=suptitle
    )

    # 智能添加坐标轴标签（只在最外层显示）
    if xlabel is not None or ylabel is not None:
        # 添加共享的x轴标签（只在最底行）
        if xlabel is not None:
            fig.text(0.5, 0.01, xlabel, ha='center', va='center', fontsize=12)

        # 添加共享的y轴标签（只在最左列）
        if ylabel is not None:
            fig.text(0.01, 0.5, ylabel, ha='center', va='center', rotation='vertical', fontsize=12)

    # 调整布局以减少留白
    if xlabel is not None or ylabel is not None:
        # 如果有坐标轴标签，预留更少的空间
        fig.subplots_adjust(top=0.92, bottom=0.05, left=0.03)
    else:
        # 如果没有坐标轴标签，使用标准调整
        fig.subplots_adjust(top=0.92)

    return fig, axes

