import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from graspkit_plot.plot_functions import rwfn_plot, rwfns_compare_plot


def _sample_rwfn_df(scale: float = 1.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "r(a.u)": [0.0, 1.0, 4.0, 9.0],
            "P(1s )": [0.0, 1.0 * scale, 2.0 * scale, 3.0 * scale],
            "Q(1s )": [0.0, 0.1 * scale, 0.2 * scale, 0.3 * scale],
            "P(2s )": [0.0, 4.0 * scale, 5.0 * scale, 6.0 * scale],
            "Q(2s )": [0.0, 0.4 * scale, 0.5 * scale, 0.6 * scale],
        }
    )


def test_rwfn_plot_defaults_to_density_mode() -> None:
    fig, axes = rwfn_plot(
        data=_sample_rwfn_df(),
        orbitals=["1s "],
        layout="1x1",
        xscale="linear",
        suptitle="",
    )

    line = axes[0, 0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(line.get_ydata(), [0.0, 1.01, 4.04, 9.09])
    assert axes[0, 0].get_title() == "1s "
    plt.close(fig)


def test_rwfn_plot_can_draw_p_or_q_component() -> None:
    df = _sample_rwfn_df()

    p_fig, p_axes = rwfn_plot(
        data=df,
        orbitals=["1s "],
        plot_mode="P",
        layout="1x1",
        x_transform="linear",
        suptitle="",
    )
    q_fig, q_axes = rwfn_plot(
        data=df,
        orbitals=["1s "],
        plot_mode="Q",
        layout="1x1",
        x_transform="linear",
        suptitle="",
    )

    np.testing.assert_allclose(p_axes[0, 0].lines[0].get_ydata(), [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(q_axes[0, 0].lines[0].get_ydata(), [0.0, 0.1, 0.2, 0.3])
    assert p_axes[0, 0].get_title() == "P(1s )"
    assert q_axes[0, 0].get_title() == "Q(1s )"
    plt.close(p_fig)
    plt.close(q_fig)


def test_rwfn_plot_components_mode_places_p_above_q_for_each_orbital() -> None:
    fig, axes = rwfn_plot(
        data=_sample_rwfn_df(),
        orbitals=["1s ", "2s "],
        plot_mode="components",
        xscale="linear",
        suptitle="",
    )

    assert axes.shape == (2, 2)
    assert axes[0, 0].get_title() == "P(1s )"
    assert axes[0, 1].get_title() == "P(2s )"
    assert axes[1, 0].get_title() == "Q(1s )"
    assert axes[1, 1].get_title() == "Q(2s )"
    np.testing.assert_allclose(axes[1, 1].lines[0].get_ydata(), [0.0, 0.4, 0.5, 0.6])
    plt.close(fig)


def test_rwfn_plot_components_mode_keeps_each_orbital_components_adjacent() -> None:
    fig, axes = rwfn_plot(
        data=_sample_rwfn_df(),
        orbitals=["1s ", "2s "],
        plot_mode="components",
        layout="2x1",
        xscale="linear",
        suptitle="",
    )

    assert axes.shape == (4, 1)
    assert axes[0, 0].get_title() == "P(1s )"
    assert axes[1, 0].get_title() == "Q(1s )"
    assert axes[2, 0].get_title() == "P(2s )"
    assert axes[3, 0].get_title() == "Q(2s )"
    plt.close(fig)


def test_rwfn_plot_components_layout_describes_orbital_groups() -> None:
    fig, axes = rwfn_plot(
        data=_sample_rwfn_df(),
        orbitals=["1s ", "2s "],
        plot_mode="components",
        layout="1x2",
        xscale="linear",
        suptitle="",
    )

    assert axes.shape == (2, 2)
    assert axes[0, 0].get_title() == "P(1s )"
    assert axes[1, 0].get_title() == "Q(1s )"
    assert axes[0, 1].get_title() == "P(2s )"
    assert axes[1, 1].get_title() == "Q(2s )"
    plt.close(fig)


def test_rwfn_plot_components_colors_are_grouped_by_orbital() -> None:
    fig, axes = rwfn_plot(
        data=_sample_rwfn_df(),
        orbitals=["1s ", "2s "],
        plot_mode="components",
        xscale="linear",
        suptitle="",
    )

    p_1s = axes[0, 0].lines[0]
    q_1s = axes[1, 0].lines[0]
    p_2s = axes[0, 1].lines[0]
    q_2s = axes[1, 1].lines[0]
    assert p_1s.get_color() == q_1s.get_color()
    assert p_2s.get_color() == q_2s.get_color()
    assert p_1s.get_color() != p_2s.get_color()
    assert p_1s.get_linestyle() == "-"
    assert q_1s.get_linestyle() == "-"
    plt.close(fig)


def test_rwfns_compare_plot_overlays_multiple_dataframes() -> None:
    old_df = _sample_rwfn_df()
    new_df = _sample_rwfn_df(scale=2.0)

    fig, axes = rwfns_compare_plot(
        data_list=[old_df, new_df],
        orbitals=["1s "],
        labels=["old", "new"],
        layout="1x1",
        x_transform="linear",
        suptitle="",
    )

    data_lines = [line for line in axes[0, 0].lines if line.get_label() in {"old", "new"}]
    assert len(data_lines) == 2
    np.testing.assert_allclose(data_lines[0].get_ydata(), [0.0, 1.01, 4.04, 9.09])
    np.testing.assert_allclose(data_lines[1].get_ydata(), [0.0, 4.04, 16.16, 36.36])
    assert axes[0, 0].get_legend() is not None
    plt.close(fig)


def test_rwfn_plot_accepts_fig_settings_size_options() -> None:
    fig, _ = rwfn_plot(
        data=_sample_rwfn_df(),
        orbitals=["1s "],
        layout="1x1",
        base_size="double_column",
        suptitle="",
    )

    np.testing.assert_allclose(fig.get_size_inches(), [7.2, 5.4])
    plt.close(fig)


def test_rwfns_compare_plot_uses_fig_settings_color_scheme() -> None:
    fig, axes = rwfns_compare_plot(
        data_list=[_sample_rwfn_df(), _sample_rwfn_df(scale=2.0)],
        orbitals=["1s "],
        labels=["old", "new"],
        layout="1x1",
        color_scheme="science",
        suptitle="",
    )

    data_lines = [line for line in axes[0, 0].lines if line.get_label() in {"old", "new"}]
    assert data_lines[0].get_color() == "#0173b2"
    assert data_lines[1].get_color() == "#de8f05"
    plt.close(fig)


def test_rwfn_plot_linear_transform_uses_raw_x_and_linear_axis() -> None:
    fig, axes = rwfn_plot(
        data=_sample_rwfn_df(),
        orbitals=["1s "],
        layout="1x1",
        x_transform="linear",
        suptitle="",
    )

    line = axes[0, 0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), [0.0, 1.0, 4.0, 9.0])
    assert axes[0, 0].get_xscale() == "linear"
    formatter = axes[0, 0].xaxis.get_major_formatter()
    assert formatter.get_useOffset() is False
    assert formatter._scientific is False
    plt.close(fig)


def test_rwfn_plot_log1p_transform_keeps_zero_x_row() -> None:
    fig, axes = rwfn_plot(
        data=_sample_rwfn_df(),
        orbitals=["1s "],
        layout="1x1",
        x_transform="log1p",
        xscale="linear",
        suptitle="",
    )

    line = axes[0, 0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), np.log1p([0.0, 1.0, 4.0, 9.0]))
    np.testing.assert_allclose(line.get_ydata(), [0.0, 1.01, 4.04, 9.09])
    plt.close(fig)


def test_rwfns_compare_plot_log1p_transform_keeps_zero_x_row() -> None:
    fig, axes = rwfns_compare_plot(
        data_list=[_sample_rwfn_df(), _sample_rwfn_df(scale=2.0)],
        orbitals=["1s "],
        labels=["old", "new"],
        layout="1x1",
        x_transform="log1p",
        xscale="linear",
        suptitle="",
    )

    data_lines = [line for line in axes[0, 0].lines if line.get_label() in {"old", "new"}]
    np.testing.assert_allclose(data_lines[0].get_xdata(), np.log1p([0.0, 1.0, 4.0, 9.0]))
    np.testing.assert_allclose(data_lines[0].get_ydata(), [0.0, 1.01, 4.04, 9.09])
    np.testing.assert_allclose(data_lines[1].get_ydata(), [0.0, 4.04, 16.16, 36.36])
    plt.close(fig)


def test_rwfn_plot_rejects_removed_log_transforms() -> None:
    for removed_transform in ["log", "ln", "log10"]:
        with pytest.raises(ValueError, match="x_transform"):
            rwfn_plot(
                data=_sample_rwfn_df(),
                orbitals=["1s "],
                x_transform=removed_transform,
            )


def test_rwfn_plot_auto_max_x_uses_last_significant_polars_value() -> None:
    data = pl.DataFrame(
        {
            "r(a.u)": [0.0, 1.0, 2.0, 3.0, 100.0],
            "P(1s )": [0.0, 1.0, 0.2, 1e-12, 0.0],
            "Q(1s )": [0.0, 0.1, 0.0, 1e-12, 0.0],
        }
    )

    fig, axes = rwfn_plot(
        data=data,
        orbitals=["1s "],
        layout="1x1",
        x_transform="linear",
        x_tail_threshold=1e-6,
        x_tail_padding=1.1,
        suptitle="",
    )

    np.testing.assert_allclose(axes[0, 0].get_xlim()[1], 2.2)
    plt.close(fig)


def test_rwfn_plot_rejects_invalid_settings_and_missing_columns() -> None:
    data = _sample_rwfn_df()

    with pytest.raises(ValueError, match="plot_mode"):
        rwfn_plot(data=data, orbitals=["1s "], plot_mode="both")

    with pytest.raises(ValueError, match="missing required columns"):
        rwfn_plot(data=data.drop("Q(1s )"), orbitals=["1s "])

    with pytest.raises(ValueError, match="x_tail_threshold"):
        rwfn_plot(data=data, orbitals=["1s "], x_tail_threshold=-1.0)

    with pytest.raises(ValueError, match="x_tail_padding"):
        rwfns_compare_plot(data_list=[data], orbitals=["1s "], x_tail_padding=0.0)
