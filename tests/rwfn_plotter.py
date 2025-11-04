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
    Automatically plot wavefunction comparison charts, supporting multiple DataFrame comparison

    Args:
        data_list: List of datasets, containing multiple DataFrames [data1, data2, data3, ...]
        column_names: List of column names to plot
        x_col: Column name for x-axis data (default: 'r(a.u)')
        labels: List of dataset labels, e.g., ['old', 'new', 'modified'] (default: ['Data 1', 'Data 2', ...])
        layout: Subplot layout (default: '2x4')
        alpha: Transparency (default: 0.75)
        max_x: Maximum x-axis value, auto-calculated if None
        xscale: x-axis scale type (default: 'symlog')
        linthresh: Linear threshold for symlog (default: 1)
        suptitle: Main title (default: 'Wavefunction Comparison')
        colors: List of line colors, e.g., ['blue', 'red', 'green'] (default: auto-allocated)
        linestyles: List of line styles, e.g., ['-', '--', '-.'] (default: all solid lines)

    Returns:
        fig, axes: matplotlib figure and axes objects
    """

    # Create multi-subplot
    fig, axes = gfs.create_multi_subplot_figure(
        layout=layout,
        base_size='single_column',
        spacing='normal',
        color_scheme='nature'
    )

    # Validate input
    if not isinstance(data_list, (list, tuple)):
        raise ValueError("data_list must be a list or tuple of DataFrames")

    n_datasets = len(data_list)

    # Set default labels
    if labels is None:
        labels = [f'Data {i+1}' for i in range(n_datasets)]
    elif len(labels) != n_datasets:
        raise ValueError(f"Number of labels ({len(labels)}) must match number of datasets ({n_datasets})")

    # Set default colors and line styles
    if colors is None:
        # Use matplotlib's default color cycle
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:n_datasets]
    elif len(colors) < n_datasets:
        # If not enough colors provided, cycle them
        colors = (colors * (n_datasets // len(colors) + 1))[:n_datasets]

    if linestyles is None:
        linestyles = ['-'] * n_datasets
    elif len(linestyles) < n_datasets:
        # If not enough styles provided, cycle them
        linestyles = (linestyles * (n_datasets // len(linestyles) + 1))[:n_datasets]

    # Calculate x-axis range
    if max_x is None:
        last_x = data_list[0][x_col].iloc[-1]
        max_x = int(np.ceil(last_x / 10)) * 10

    # Get layout information
    nrows = int(layout.split('x')[0])
    ncols = int(layout.split('x')[1])

    # Automatically plot all subplots
    for i, col_name in enumerate(column_names):
        # Calculate subplot position
        row = i // ncols
        col = i % ncols

        # Plot all datasets
        for j, data in enumerate(data_list):
            # 使用与fig_settings中相同的辅助函数来安全访问axes
            if not isinstance(axes, np.ndarray):
                ax = axes  # 单个axes对象
            elif axes.ndim == 1:
                if axes.shape[0] == 1:  # 1x1布局
                    ax = axes[0]
                elif row == 0:  # 1xN布局
                    ax = axes[col]
                else:  # Nx1布局
                    ax = axes[row]
            else:  # MxN布局
                ax = axes[row, col]

            ax.plot(np.sqrt(data[x_col]), data[col_name],
                   alpha=alpha,
                   label=labels[j],
                   color=colors[j],
                   linestyle=linestyles[j])

        # Set x-axis
        if not isinstance(axes, np.ndarray):
            ax = axes
        elif axes.ndim == 1:
            if axes.shape[0] == 1:
                ax = axes[0]
            elif row == 0:
                ax = axes[col]
            else:
                ax = axes[row]
        else:
            ax = axes[row, col]

        ax.set_xlim(0, max_x)
        if xscale == 'symlog':
            ax.set_xscale('symlog', linthresh=linthresh)

        # Add legend
        ax.legend()

    # Add reference lines
    gfs.add_reference_lines_to_subplots(
        axes, layout=layout,
        y_values=[0],
        y_styles='--',
        y_colors='gray',
        y_labels=None
    )

    # Configure subplot grid
    gfs.configure_subplot_grid(
        fig, axes, layout=layout,
        title=column_names,
        suptitle=suptitle
    )

    return fig, axes


# Define list of column names to plot
column_names = ['P(5f )', 'Q(5f )', 'P(5f-)', 'Q(5f-)', 'P(5d )', 'Q(5d )', 'P(5d-)', 'Q(5d-)']

# Use new multi-DataFrame function to automatically plot all subplots
fig, axes = auto_plot_wavefunction_comparison(
    [rwfn_data_e1_as1, rwfn_data_e1_as1_2],  # data list
    column_names,
    labels=('old', 'new'),
    suptitle='Radial Wavefunction'
)

fig.show()

# ===== Usage Examples =====

# Example 1: Plot only 5f orbital data
fig1, axes1 = auto_plot_wavefunction_comparison(
    [rwfn_data_e1_as1, rwfn_data_e1_as1_2],
    ['P(5f )', 'Q(5f )', 'P(5f-)', 'Q(5f-)'],
    layout='1x4',
    suptitle='5f Orbital Wavefunctions'
)

# Example 2: Plot only 5d orbital data with different labels and styles
fig2, axes2 = auto_plot_wavefunction_comparison(
    [rwfn_data_e1_as1, rwfn_data_e1_as1_2],
    ['P(5d )', 'Q(5d )', 'P(5d-)', 'Q(5d-)'],
    labels=('original', 'modified'),
    colors=['blue', 'red'],
    linestyles=['-', '--'],
    suptitle='5d Orbital Wavefunctions'
)

# Example 3: Simulate multiple DataFrame comparison (if you have more data)
# Here we reuse existing data to demonstrate multi-DataFrame functionality
fig3, axes3 = auto_plot_wavefunction_comparison(
    [rwfn_data_e1_as1, rwfn_data_e1_as1_2, rwfn_data_e1_as1],  # 3 datasets
    ['P(5f )', 'Q(5f )'],
    labels=('calculation_A', 'calculation_B', 'calculation_C'),
    colors=['blue', 'red', 'green'],
    linestyles=['-', '--', '-.'],
    layout='1x2',
    suptitle='Three-Way Comparison'
)

# Example 4: Custom x-axis range and multiple datasets
fig4, axes4 = auto_plot_wavefunction_comparison(
    [rwfn_data_e1_as1, rwfn_data_e1_as1_2],
    ['P(5f )', 'Q(5f )', 'P(5d )', 'Q(5d )'],
    max_x=50,  # fixed x-axis range
    labels=('old_method', 'new_method'),
    colors=['darkblue', 'darkred'],
    alpha=0.8,
    suptitle='Custom Range Comparison'
)

# Example 5: Compare only one subplot but show multiple datasets
fig5, axes5 = auto_plot_wavefunction_comparison(
    [rwfn_data_e1_as1, rwfn_data_e1_as1_2, rwfn_data_e1_as1],
    ['P(5f )'],  # plot only one column
    labels=('baseline', 'improved', 'experimental'),
    colors=['black', 'red', 'blue'],
    linestyles=['-', '--', '-.'],
    layout='1x1',
    suptitle='Detailed P(5f) Comparison'
)

print("Plotting completed!")
print("\nUsage:")
print("1. Modify column_names list to select columns to plot")
print("2. Adjust layout parameter to change subplot layout")
print("3. Modify labels parameter to customize legend labels")
print("4. Adjust alpha parameter to change transparency")
print("5. Set max_x parameter to customize x-axis range")
print("6. Use colors parameter to customize colors")
print("7. Use linestyles parameter to customize line styles")
print("8. Pass any number of DataFrames in data_list")
print("\nExample: Compare 3 calculation results")
print("fig, axes = auto_plot_wavefunction_comparison(")
print("    [data1, data2, data3],  # 3 DataFrames")
print("    column_names,")
print("    labels=['method_A', 'method_B', 'method_C'],")
print("    colors=['blue', 'red', 'green'],")
print("    linestyles=['-', '--', '-.']")
print(")")