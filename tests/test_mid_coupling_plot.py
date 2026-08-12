from pathlib import Path

import matplotlib
from matplotlib.colors import to_hex
import numpy as np
import polars as pl
import pytest
from graspkit.utils.data_modules import MixCoefficientBlock, MixCoefficientData

matplotlib.use("Agg")

from graspkit_plot import (
    MidCouplingPlotSeries,
    build_mid_coupling_plot_series,
    plot_mid_coupling_contribution,
)
from graspkit_plot.mid_coupling import _mid_coupling_bar_colors


def _mix_data() -> MixCoefficientData:
    coefficients = np.sqrt(
        np.asarray(
            [[0.60, 0.20, 0.10, 0.05, 0.05], [0.10, 0.10, 0.70, 0.05, 0.05]],
            dtype=np.float64,
        )
    )
    return MixCoefficientData(
        blocks=[
            MixCoefficientBlock(
                block_index=0,
                csf_count=5,
                level_count=2,
                j_value_location=3,
                j_value="1",
                parity=1,
                level_ids=np.asarray([4, 7], dtype=np.int64),
                base_energy=0.0,
                level_energies=np.asarray([0.0, 0.1], dtype=np.float64),
                mix_coefficients=coefficients,
            )
        ],
        sorted_level_energies=[0.0, 0.1],
    )


def _csfs_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "idx": [0, 1, 2, 3, 4],
            "block_id": [0, 0, 0, 0, 0],
            "coupling_signature": [[7, 2], [7, 2], [3, 2], [5, 2], [9, 2]],
        },
        schema_overrides={
            "idx": pl.UInt64,
            "block_id": pl.UInt32,
            "coupling_signature": pl.List(pl.Int32),
        },
    )


def test_build_mid_coupling_plot_series_sorts_and_merges_channels() -> None:
    series = build_mid_coupling_plot_series(
        _csfs_frame(),
        _mix_data(),
        block_id=0,
        asf_row_index=0,
        min_contribution=0.075,
    )

    assert series.labels == ["7/2", "3/2", "Others"]
    assert series.csf_percentages == pytest.approx([40.0, 20.0, 40.0])
    assert series.contributions == pytest.approx([0.8, 0.1, 0.1])
    assert series.level_id == 4
    assert series.parity_label == "+"


def test_plot_mid_coupling_contribution_writes_svg(tmp_path: Path) -> None:
    series = MidCouplingPlotSeries(
        block_id=0,
        asf_row_index=0,
        level_id=4,
        j_value="1",
        parity=1,
        parity_label="+",
        labels=["7/2", "5/2", "Others"],
        csf_percentages=[30.5, 23.2, 46.3],
        contributions=[0.953, 0.020, 0.027],
    )
    output = tmp_path / "e1_2J2_MCJ.svg"

    plot_mid_coupling_contribution(series, output)

    svg = output.read_text(encoding="utf-8")
    assert "Percentage Share (%)" in svg
    assert "Contribution Value" in svg
    assert "30.5%" in svg
    assert "0.953" in svg
    assert "0.020" in svg
    assert "stroke-dasharray" in svg
    assert "#222222" in svg
    assert "stroke-opacity: 0.7" in svg
    assert "J = 1" not in svg


def test_mid_coupling_bars_use_requested_positional_palette() -> None:
    colors = _mid_coupling_bar_colors(
        ["7/2", "5/2", "9/2", "3/2", "Others"]
    )

    assert [to_hex(color) for color in colors] == [
        "#56b4e9",
        "#e69f00",
        "#009e73",
        "#cc79a7",
        "#bbbbbb",
    ]


def test_mid_coupling_palette_cycles_after_four_named_channels() -> None:
    colors = _mid_coupling_bar_colors(["a", "b", "c", "d", "e"])

    assert [to_hex(color) for color in colors] == [
        "#56b4e9",
        "#e69f00",
        "#009e73",
        "#cc79a7",
        "#56b4e9",
    ]


def test_all_mid_coupling_bars_have_no_visible_border(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = MidCouplingPlotSeries(
        block_id=0,
        asf_row_index=0,
        level_id=4,
        j_value="1",
        parity=1,
        parity_label="+",
        labels=["7/2", "5/2", "Others"],
        csf_percentages=[30.5, 23.2, 46.3],
        contributions=[0.953, 0.020, 0.027],
    )

    bar_kwargs: dict[str, object] = {}
    original_bar = matplotlib.axes.Axes.bar

    def recording_bar(
        axis: matplotlib.axes.Axes,
        *args: object,
        **kwargs: object,
    ) -> object:
        bar_kwargs.update(kwargs)
        return original_bar(axis, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "bar", recording_bar)

    output = tmp_path / "bars.svg"
    plot_mid_coupling_contribution(series, output)

    assert bar_kwargs["edgecolor"] == "none"
    assert bar_kwargs["linewidth"] == 0
