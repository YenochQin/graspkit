import numpy as np
import matplotlib.pyplot as plt

def rose_chart(categories, share, value, colors=None, r_min=0.12):
    """
    categories: 类别名称列表
    share: 各类别占比（可不和 1 完全相等，函数会自动归一化）
    value: 各类别贡献（任意正数，按最大值归一到半径）
    r_min: 基础半径，避免柱子太靠近圆心
    """
    share = np.array(share, dtype=float)
    share = share / share.sum()                  # 角度=占比
    widths = share * 2*np.pi

    value = np.array(value, dtype=float)
    vmax = value.max()
    radii = r_min + 0.88 * (value / vmax)        # 半径=贡献（归一化到 0~1）

    # 每个扇区中心角
    cum = np.cumsum(widths)
    theta = cum - widths/2

    fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(6,6), dpi=150)
    ax.set_theta_zero_location('N')   # 0°在上方
    ax.set_theta_direction(-1)        # 顺时针
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(color='lightgray', alpha=0.6)
    ax.set_ylim(0, 1.1)

    if colors is None:
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i) for i in range(len(categories))]

    # 先画一圈淡色“占比环”（可选，让占比更明显）
    ax.bar(theta, np.full_like(radii, r_min), width=widths,
           bottom=0, color='lightgray', alpha=0.25, edgecolor='none', zorder=0)

    # 再画贡献柱（长度体现贡献）
    ax.bar(theta, radii, width=widths, bottom=0,
           color=colors, edgecolor='white', linewidth=1.0, zorder=2)

    # 文本标注：名称 + 数值 + 占比
    for t, r, name, val, pct in zip(theta, radii, categories, value, share):
        ax.text(t, r + 0.05, f"{name}\n{int(val):,} ({pct*100:.1f}%)",
                ha='center', va='bottom', rotation=np.degrees(-t),
                rotation_mode='anchor', fontsize=9)

    plt.tight_layout()
    return fig, ax

# 演示数据
cats  = ['A','B','C','D','E','F']
share = [0.20, 0.15, 0.10, 0.25, 0.18, 0.12]   # 成分占比
value = [200,  80, 120, 350, 300, 150]        # 贡献/数量

fig, ax = rose_chart(cats, share, value)
plt.show()
