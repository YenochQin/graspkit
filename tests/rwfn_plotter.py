import graspkit as gk
import matplotlib.pyplot as plt
import numpy as np
import graspkit.utils.fig_settings as gfs


rwfn_path_e1_as1 = "D:\\PythonProjects\\RWFN_test\\GdI-evenI-vv\\eI-vvas1.w"
rwfn_path_e1_as1_2 = "D:\\PythonProjects\\RWFN_test\\GdI-evenI-vv\\eI-vvas1-new.w"


rwfn_load_e1_as1 = gk.GraspFileLoad.from_filepath(rwfn_path_e1_as1, "WAVEFUNCTION")
rwfn_data_e1_as1 = rwfn_load_e1_as1.get_radial_wavefunction()
rwfn_load_e1_as1_2 = gk.GraspFileLoad.from_filepath(rwfn_path_e1_as1_2, "WAVEFUNCTION")
rwfn_data_e1_as1_2 = rwfn_load_e1_as1_2.get_radial_wavefunction()

def auto_plot_wavefunction_comparison(data_list, column_names, x_col='r(a.u)',
                                    labels=None, layout='2x4',
                                    alpha=0.75, max_x=None, xscale='symlog',
                                    linthresh=1, suptitle='Wavefunction Comparison',
                                    colors=None, linestyles=None):
    """
    自动绘制波函数对比图的通用函数，支持多个DataFrame对比

    Args:
        data_list: 数据集列表，包含多个DataFrame [data1, data2, data3, ...]
        column_names: 要绘制的列名列表
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

    Returns:
        fig, axes: matplotlib的figure和axes对象
    """

    # 创建多子图
    fig, axes = gfs.create_multi_subplot_figure(
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
        # 使用matplotlib的默认颜色循环
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:n_datasets]
    elif len(colors) < n_datasets:
        # 如果提供的颜色不够，循环使用
        colors = (colors * (n_datasets // len(colors) + 1))[:n_datasets]

    if linestyles is None:
        linestyles = ['-'] * n_datasets
    elif len(linestyles) < n_datasets:
        # 如果提供的样式不够，循环使用
        linestyles = (linestyles * (n_datasets // len(linestyles) + 1))[:n_datasets]

    # 计算x轴范围
    if max_x is None:
        last_x = data_list[0][x_col].iloc[-1]
        max_x = int(np.ceil(last_x / 10)) * 10

    # 获取布局信息
    nrows = int(layout.split('x')[0])
    ncols = int(layout.split('x')[1])

    # 自动绘制所有子图
    for i, col_name in enumerate(column_names):
        # 计算子图位置
        row = i // ncols
        col = i % ncols

        # 绘制所有数据集
        for j, data in enumerate(data_list):
            axes[row, col].plot(np.sqrt(data[x_col]), data[col_name],
                               alpha=alpha,
                               label=labels[j],
                               color=colors[j],
                               linestyle=linestyles[j])

        # 设置x轴
        axes[row, col].set_xlim(0, max_x)
        if xscale == 'symlog':
            axes[row, col].set_xscale('symlog', linthresh=linthresh)

        # 添加图例
        axes[row, col].legend()

    # 添加参考线
    gfs.add_reference_lines_to_subplots(
        axes, layout=layout,
        y_values=[0],
        y_styles='--',
        y_colors='gray',
        y_labels=None
    )

    # 配置子图网格
    gfs.configure_subplot_grid(
        fig, axes, layout=layout,
        title=column_names,
        suptitle=suptitle
    )

    return fig, axes


# 定义要绘制的列名列表
column_names = ['P(5f )', 'Q(5f )', 'P(5f-)', 'Q(5f-)', 'P(5d )', 'Q(5d )', 'P(5d-)', 'Q(5d-)']

# 使用新的多DataFrame函数自动绘制所有子图
fig, axes = auto_plot_wavefunction_comparison(
    [rwfn_data_e1_as1, rwfn_data_e1_as1_2],  # 数据列表
    column_names,
    labels=('old', 'new'),
    suptitle='Radial Wavefunction'
)

fig.show()

# ===== 使用示例 =====

# 示例1：只绘制5f轨道的数据
fig1, axes1 = auto_plot_wavefunction_comparison(
    [rwfn_data_e1_as1, rwfn_data_e1_as1_2],
    ['P(5f )', 'Q(5f )', 'P(5f-)', 'Q(5f-)'],
    layout='1x4',
    suptitle='5f Orbital Wavefunctions'
)

# 示例2：只绘制5d轨道的数据，使用不同的标签和样式
fig2, axes2 = auto_plot_wavefunction_comparison(
    [rwfn_data_e1_as1, rwfn_data_e1_as1_2],
    ['P(5d )', 'Q(5d )', 'P(5d-)', 'Q(5d-)'],
    labels=('original', 'modified'),
    colors=['blue', 'red'],
    linestyles=['-', '--'],
    suptitle='5d Orbital Wavefunctions'
)

# 示例3：模拟多个DataFrame对比（如果有更多数据）
# 这里我们重复使用现有数据来演示多DataFrame功能
fig3, axes3 = auto_plot_wavefunction_comparison(
    [rwfn_data_e1_as1, rwfn_data_e1_as1_2, rwfn_data_e1_as1],  # 3个数据集
    ['P(5f )', 'Q(5f )'],
    labels=('calculation_A', 'calculation_B', 'calculation_C'),
    colors=['blue', 'red', 'green'],
    linestyles=['-', '--', '-.'],
    layout='1x2',
    suptitle='Three-Way Comparison'
)

# 示例4：自定义x轴范围和多个数据集
fig4, axes4 = auto_plot_wavefunction_comparison(
    [rwfn_data_e1_as1, rwfn_data_e1_as1_2],
    ['P(5f )', 'Q(5f )', 'P(5d )', 'Q(5d )'],
    max_x=50,  # 固定x轴范围
    labels=('old_method', 'new_method'),
    colors=['darkblue', 'darkred'],
    alpha=0.8,
    suptitle='Custom Range Comparison'
)

# 示例5：只比较一个子图，但显示多个数据集
fig5, axes5 = auto_plot_wavefunction_comparison(
    [rwfn_data_e1_as1, rwfn_data_e1_as1_2, rwfn_data_e1_as1],
    ['P(5f )'],  # 只绘制一个列
    labels=('baseline', 'improved', 'experimental'),
    colors=['black', 'red', 'blue'],
    linestyles=['-', '--', '-.'],
    layout='1x1',
    suptitle='Detailed P(5f) Comparison'
)

print("绘图完成！")
print("\n使用方法：")
print("1. 修改column_names列表来选择要绘制的列")
print("2. 调整layout参数改变子图布局")
print("3. 修改labels参数自定义图例标签")
print("4. 调整alpha参数改变透明度")
print("5. 设置max_x参数自定义x轴范围")
print("6. 使用colors参数自定义颜色")
print("7. 使用linestyles参数自定义线条样式")
print("8. 在data_list中传入任意数量的DataFrame")
print("\n示例：比较3个计算结果")
print("fig, axes = auto_plot_wavefunction_comparison(")
print("    [data1, data2, data3],  # 3个DataFrame")
print("    column_names,")
print("    labels=['method_A', 'method_B', 'method_C'],")
print("    colors=['blue', 'red', 'green'],")
print("    linestyles=['-', '--', '-.']")
print(")")