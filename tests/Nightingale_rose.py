import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from typing import cast
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

def rose_chart(categories, quantity, sum_squared_ci, colors=None, r_min=0.12,
                aggregate_threshold=None, aggregate_name="Others"):
    """
    categories: 类别名称列表
    quantity: 各类别占比（可不和 1 完全相等，函数会自动归一化）
    sum_squared_ci: 各类别贡献（任意正数，按最大值归一到半径）
    r_min: 基础半径，避免柱子太靠近圆心
    aggregate_threshold: 聚合阈值，占比小于此值的类别将被合并为"其他"
    aggregate_name: 聚合后类别的名称
    """
    # 转换为numpy数组
    # 处理长度不一的元组：提供几种显示选项
    def format_category(cat):
        if isinstance(cat, tuple):
            # 选项1：用连字符连接
            return '-'.join(str(item) for item in cat)
            # 选项2：用斜杠连接
            # return '/'.join(str(item) for item in cat)
            # 选项3：带括号的原始格式
            # return str(cat)
        else:
            return str(cat)

    categories = [format_category(cat) for cat in categories]
    categories = np.array(categories)
    quantity = np.array(quantity, dtype=float)
    sum_squared_ci = np.array(sum_squared_ci, dtype=float)

    # 智能聚合策略：处理"大数量小贡献"和"小数量大贡献"的情况
    if aggregate_threshold is not None:
        # 计算贡献效率（sum_squared_ci/quantity比率）来识别特殊项
        with np.errstate(divide='ignore', invalid='ignore'):
            efficiency = np.where(quantity > 0, sum_squared_ci / quantity, 0)

        # 识别特殊项：高效率（小占比大贡献）
        mean_efficiency = np.mean(efficiency)
        std_efficiency = np.std(efficiency)
        high_efficiency_threshold = mean_efficiency + 2 * std_efficiency

        mask_high_efficiency = efficiency > high_efficiency_threshold
        mask_normal_quantity = quantity >= aggregate_threshold
        mask_small_quantity = quantity < aggregate_threshold

        # 保留两类特殊项：
        # 1. 正常占比的项
        # 2. 高效率的小占比项（小占比大贡献）
        mask_keep = mask_normal_quantity | mask_high_efficiency

        if np.any(~mask_keep):
            # 计算聚合项（普通的小占比项）
            aggregated_quantity = quantity[~mask_keep].sum()
            aggregated_sum_squared_ci = sum_squared_ci[~mask_keep].sum()
            aggregated_count = np.sum(~mask_keep)

            # 保留重要项
            categories_keep = categories[mask_keep]
            quantity_keep = quantity[mask_keep]
            sum_squared_ci_keep = sum_squared_ci[mask_keep]

            # Add aggregated item with count
            if aggregated_count > 0:
                aggregate_label = f"{aggregate_name}({aggregated_count}items)"
                categories = np.append(categories_keep, aggregate_label)
                quantity = np.append(quantity_keep, aggregated_quantity)
                sum_squared_ci = np.append(sum_squared_ci_keep, aggregated_sum_squared_ci)
            else:
                categories = categories_keep
                quantity = quantity_keep
                sum_squared_ci = sum_squared_ci_keep

            # 按贡献值降序排列，突出重要项
            sort_idx = np.argsort(-sum_squared_ci)
            categories = categories[sort_idx]
            quantity = quantity[sort_idx]
            sum_squared_ci = sum_squared_ci[sort_idx]

    # 归一化占比
    quantity = quantity / quantity.sum()                  # 角度=占比
    widths = quantity * 2*np.pi

    vmax = sum_squared_ci.max()
    radii = r_min + 0.88 * (sum_squared_ci / vmax)        # 半径=贡献（归一化到 0~1）

    # 每个扇区中心角
    cum = np.cumsum(widths)
    theta = cum - widths/2

    # 直接创建极坐标图
    fig = plt.figure(figsize=(6,6), dpi=150)
    ax = cast(PolarAxes, fig.add_subplot(111, projection='polar'))
    ax.set_theta_zero_location('N')   # 0°在上方
    ax.set_theta_direction(-1)        # 顺时针
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(color='lightgray', alpha=0.6)
    ax.set_ylim(0, 1.1)

    # colors 必须在数据聚合后生成，以确保长度匹配
    if colors is None:
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(len(categories))]

    # 先画一圈淡色"占比环"（可选，让占比更明显）
    ax.bar(theta, np.full_like(radii, r_min), width=widths,
           bottom=0, color='lightgray', alpha=0.25, edgecolor='none', zorder=0)

    # 再画贡献柱（长度体现贡献）
    ax.bar(theta, radii, width=widths, bottom=0,
           color=colors, edgecolor='white', linewidth=1.0, zorder=2)

    # 文本标注：专门处理帕累托分布的情况
    for i, (t, r, name, val, pct) in enumerate(zip(theta, radii, categories, sum_squared_ci, quantity)):
        width_angle = widths[i] * 180 / np.pi  # 转换为角度

        # 计算贡献效率
        efficiency = val / pct if pct > 0 else 0

        # 高亮特殊项：小占比但高效率（小占比大贡献）
        is_special = pct < 0.05 and efficiency > np.mean(sum_squared_ci / quantity) + np.std(sum_squared_ci / quantity)

        # Check if aggregated item
        is_aggregated = aggregate_name in name and '(' in name and ')' in name

        if is_special:
            # Special item: highlight with leader line
            text_r = r + 0.2
            ax.annotate(f"★{name}\nValue:{int(val):,}\nShare:{pct*100:.2f}%\nEfficiency:{efficiency:.0f}",
                       xy=(t, r), xytext=(t, text_r),
                       ha='center', va='center',
                       arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9))
        elif is_aggregated:
            # 聚合项：显示统计信息
            ax.text(t, r + 0.08, name,
                    ha='center', va='bottom', rotation=np.degrees(-t),
                    rotation_mode='anchor', fontsize=8, style='italic',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.7))
        elif width_angle < 6:  # 极窄角度，垂直放置
            ax.text(t, r + 0.06, f"{name[:10]}...",  # 截断长名称
                    ha='center', va='bottom', rotation=90,
                    rotation_mode='anchor', fontsize=7)
        elif width_angle < 12:  # 较窄角度
            ax.text(t, r + 0.05, f"{name[:15]}",
                    ha='center', va='bottom', rotation=np.degrees(-t),
                    rotation_mode='anchor', fontsize=8)
        else:
            # 正常标注
            ax.text(t, r + 0.04, f"{name}\n{int(val):,}",
                    ha='center', va='bottom', rotation=np.degrees(-t),
                    rotation_mode='anchor', fontsize=9)

    plt.tight_layout()
    return fig, ax

def efficiency_bar_chart(categories, quantity, sum_squared_ci, colors=None, figsize=(12, 8), min_threshold=10):
    """
    Create a dual-axis bar chart showing quantity and contribution values

    Parameters:
    categories: list of category names
    quantity: list of proportions
    sum_squared_ci: list of contribution sum_squared_cis
    colors: colors for bars (optional)
    figsize: tuple, figure size
    min_threshold: minimum percentage threshold (default 10). Categories with percentage below this will be aggregated into "Others"
    """
    # Convert to numpy arrays
    categories = np.array(categories)
    quantity = np.array(quantity, dtype=float)
    sum_squared_ci = np.array(sum_squared_ci, dtype=float)


    # Handle tuple categories
    def format_category(cat):
        if isinstance(cat, tuple):
            return '-'.join(str(item) for item in cat)
        else:
            return str(cat)

    categories = [format_category(cat) for cat in categories]
    categories = np.array(categories)  # Convert back to numpy array for boolean indexing

    # Normalize quantity to 100%
    quantity_percent = quantity / quantity.sum() * 100

    # Aggregate small categories if needed
    if min_threshold > 0:
        mask_above_threshold = quantity_percent >= min_threshold

        if np.any(~mask_above_threshold):  # If there are categories below threshold
            # Calculate aggregated values
            aggregated_quantity = quantity[~mask_above_threshold].sum()
            aggregated_sum_squared_ci = sum_squared_ci[~mask_above_threshold].sum()
            aggregated_quantity_percent = quantity_percent[~mask_above_threshold].sum()
            aggregated_count = np.sum(~mask_above_threshold)

            # Keep categories above threshold
            categories_keep = categories[mask_above_threshold]
            quantity_keep = quantity[mask_above_threshold]
            sum_squared_ci_keep = sum_squared_ci[mask_above_threshold]
            quantity_percent_keep = quantity_percent[mask_above_threshold]

            # Add aggregated "Others" category
            categories = np.append(categories_keep, f"Others")
            quantity = np.append(quantity_keep, aggregated_quantity)
            sum_squared_ci = np.append(sum_squared_ci_keep, aggregated_sum_squared_ci)
            quantity_percent = np.append(quantity_percent_keep, aggregated_quantity_percent)

    # Sort by contribution value (descending) - to show highest contribution first
    # But keep "Others" category at the end
    others_mask = np.char.startswith(categories, 'Others')
    normal_mask = ~others_mask

    # Sort normal categories by contribution value
    if np.any(normal_mask):
        normal_indices = np.where(normal_mask)[0]
        others_indices = np.where(others_mask)[0]

        normal_sort_idx = normal_indices[np.argsort(-sum_squared_ci[normal_indices])]

        # Combine: normal categories first (sorted), then others categories
        final_sort_idx = np.concatenate([normal_sort_idx, others_indices])
    else:
        # If all are others categories, keep original order
        final_sort_idx = np.arange(len(categories))

    categories = categories[final_sort_idx]
    quantity = quantity[final_sort_idx]
    sum_squared_ci = sum_squared_ci[final_sort_idx]
    quantity_percent = quantity_percent[final_sort_idx]

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # Set individual colors for each category
    if colors is None:
        # Use distinct colors for different categories
        cmap = plt.colormaps.get_cmap('tab20')
        colors = [cmap(i % 20) for i in range(len(categories))]

    # Create bar chart (quantity)
    bars = ax1.bar(range(len(categories)), quantity_percent, color=colors, alpha=0.7, edgecolor='black')

    # Create line chart (contribution sum_squared_cis)
    line = ax2.plot(range(len(categories)), sum_squared_ci, 'o-', color='darkred', linewidth=2, markersize=6, alpha=0.8, label='Sum of squared ci')

    # Add labels with percentage and contribution value
    for i, (bar, pct, val, cat) in enumerate(zip(bars, quantity_percent, sum_squared_ci, categories)):
        height = bar.get_height()

        # Add label with percentage and contribution value - position above bar
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%\n{val:.3f}',
                ha='center', va='bottom', fontsize=16)

    # Format primary y-axis (quantity)
    ax1.set_ylabel('Percentage share(%)', fontsize=16, color='blue')
    ax1.set_xlabel('Intermediate coupling channel', fontsize=16)
    ax1.set_ylim(0, max(quantity_percent) * 1.2)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=14)
    # Ensure y-axis ticks are only on the left side
    ax1.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False)

    # Add complete border for primary axis
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)
    ax1.spines['left'].set_visible(True)
    ax1.spines['bottom'].set_visible(True)
    ax1.spines['top'].set_color('black')
    ax1.spines['right'].set_color('black')
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_color('black')
    ax1.spines['top'].set_linewidth(1)
    ax1.spines['right'].set_linewidth(1)
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['bottom'].set_linewidth(1)

    # Format secondary y-axis (contribution sum_squared_ci)
    ax2.set_ylabel('Contribution Value', fontsize=16, color='darkred')
    ax2.set_ylim(0, max(sum_squared_ci) * 1.1)
    ax2.tick_params(axis='y', labelcolor='darkred', labelsize=14)
    # Ensure y-axis ticks are only on the right side
    ax2.tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True)

    # Configure secondary axis spines - keep right spine visible but hide top
    ax2.spines['top'].set_visible(False)  # Hide to avoid conflict with ax1
    ax2.spines['right'].set_visible(True)
    ax2.spines['left'].set_visible(False)  # Hide to avoid conflict with ax1
    ax2.spines['bottom'].set_visible(False)  # Hide to avoid conflict with ax1
    ax2.spines['right'].set_color('black')
    ax2.spines['right'].set_linewidth(1)

    # Set x-axis labels (rotate for better readability)
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=16)

    # plt.title(f'Intermediate coupling channel',
            #   fontsize=14, fontweight='bold')

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color='darkred', linewidth=2, label='Sum of squared ci'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=14)

    # Add grid
    ax1.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig, ax1

