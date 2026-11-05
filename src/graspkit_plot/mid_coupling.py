"""Intermediate-coupling CSF population and CI-square contribution plots."""

# Matplotlib's public plotting API intentionally exposes untyped **kwargs.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from graspkit.CSFs_processor import summarize_coupling_ci_squared
from graspkit.utils.data_modules import MixCoefficientBlock, MixCoefficientData


_CHANNEL_COLORS: tuple[str, ...] = (
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
)
_OTHERS_COLOR = "#BBBBBB"
_LINE_COLOR = "#222222"
_AXIS_COLOR = "#222222"
_TEXT_COLOR = "#222222"
_GRID_COLOR = "#D9D9D9"
_BACKGROUND_COLOR = "#FFFFFF"


@dataclass(frozen=True)
class MidCouplingPlotSeries:
    """Display-ready values for one ASF intermediate-coupling plot."""

    block_id: int
    asf_row_index: int
    level_id: int
    j_value: str
    parity: int
    parity_label: str
    labels: list[str]
    csf_percentages: list[float]
    contributions: list[float]


def _twice_j_label(value: int) -> str:
    return str(value // 2) if value % 2 == 0 else f"{value}/2"


def _coupling_label(signature: list[int]) -> str:
    labels = [_twice_j_label(value) for value in signature]
    return labels[0] if len(labels) == 1 else "(" + ", ".join(labels) + ")"


def _twice_j_value(j_value: str) -> int:
    if "/" in j_value:
        numerator, denominator = j_value.split("/", maxsplit=1)
        return 2 * int(numerator) // int(denominator)
    return 2 * int(j_value)


def mid_coupling_output_stem(*, parity: int, asf_row_index: int, j_value: str) -> str:
    """Name a plot using parity, one-based ASF row and twice J."""
    parity_prefix = {1: "e", 2: "o"}.get(parity, f"p{parity}")
    return f"{parity_prefix}{asf_row_index + 1}_2J{_twice_j_value(j_value)}_MCJ"


def _find_block(mix_data: MixCoefficientData, block_id: int) -> MixCoefficientBlock:
    for block in mix_data.blocks:
        if block.block_index == block_id:
            return block
    raise ValueError(f"unknown block_id: {block_id}")


def build_mid_coupling_plot_series(
    csfs_df: pl.DataFrame,
    mix_data: MixCoefficientData,
    *,
    block_id: int,
    asf_row_index: int,
    coupling_level: int = 1,
    min_csf_percentage: float = 0.0,
    min_contribution: float = 0.0,
    max_channels: int | None = None,
) -> MidCouplingPlotSeries:
    """Aggregate one ASF into contribution-sorted channels and an Others bin."""
    block = _find_block(mix_data, block_id)
    if asf_row_index < 0 or asf_row_index >= block.level_count:
        raise ValueError(
            f"block {block_id} asf_row_index is outside [0, {block.level_count})"
        )
    summary = summarize_coupling_ci_squared(
        csfs_df,
        mix_data,
        asf_row_indices=[
            [asf_row_index] if candidate.block_index == block_id else []
            for candidate in mix_data.blocks
        ],
        coupling_level=coupling_level,
        include_members=False,
        allow_empty_blocks=True,
    )
    return _series_from_summary(
        summary,
        block,
        asf_row_index,
        min_csf_percentage,
        min_contribution,
        max_channels,
    )


def _series_from_summary(
    summary: pl.DataFrame,
    block: MixCoefficientBlock,
    asf_row_index: int,
    min_csf_percentage: float,
    min_contribution: float,
    max_channels: int | None,
) -> MidCouplingPlotSeries:
    if not math.isfinite(min_csf_percentage) or min_csf_percentage < 0:
        raise ValueError("min_csf_percentage must be finite and non-negative")
    if not math.isfinite(min_contribution) or min_contribution < 0:
        raise ValueError("min_contribution must be finite and non-negative")
    if max_channels is not None and max_channels <= 0:
        raise ValueError("max_channels must be a positive integer or None")

    if summary.is_empty():
        raise ValueError(
            f"block {block.block_index}, ASF row {asf_row_index} has no CSFs"
        )

    rows = sorted(
        summary.iter_rows(named=True),
        key=lambda row: float(row["sum_ci"]),
        reverse=True,
    )
    total_csfs = sum(int(row["count"]) for row in rows)
    named_rows = [
        row
        for row in rows
        if 100.0 * int(row["count"]) / total_csfs >= min_csf_percentage
        and float(row["sum_ci"]) >= min_contribution
    ]
    if max_channels is not None:
        named_rows = named_rows[:max_channels]
    named_ids = {id(row) for row in named_rows}
    other_rows = [row for row in rows if id(row) not in named_ids]

    labels = [
        _coupling_label(cast(list[int], row["coupling_signature"]))
        for row in named_rows
    ]
    percentages = [100.0 * int(row["count"]) / total_csfs for row in named_rows]
    contributions = [float(row["sum_ci"]) for row in named_rows]
    if other_rows:
        labels.append("Others")
        percentages.append(
            100.0 * sum(int(row["count"]) for row in other_rows) / total_csfs
        )
        contributions.append(sum(float(row["sum_ci"]) for row in other_rows))

    parity_labels = {1: "+", 2: "-"}
    return MidCouplingPlotSeries(
        block_id=block.block_index,
        asf_row_index=asf_row_index,
        level_id=int(block.level_ids[asf_row_index]),
        j_value=block.j_value,
        parity=block.parity,
        parity_label=parity_labels.get(block.parity, str(block.parity)),
        labels=labels,
        csf_percentages=percentages,
        contributions=contributions,
    )


def _contribution_axis_upper_limit(contributions: list[float]) -> float:
    maximum = max(contributions, default=1.0)
    return max(0.1, maximum * 1.12)


def _contribution_label_layout(
    contributions: list[float], index: int
) -> tuple[int, int, str]:
    value = contributions[index]
    left_value = contributions[index - 1] if index > 0 else None
    right_value = contributions[index + 1] if index + 1 < len(contributions) else None
    steep_threshold = max(0.08, value * 3.0)
    left_is_high = left_value is not None and left_value - value > steep_threshold
    right_is_high = right_value is not None and right_value - value > steep_threshold
    if left_is_high and not right_is_high:
        return 5, 5, "left"
    if right_is_high and not left_is_high:
        return -5, 5, "right"
    return 0, 7, "center"


def _avoid_bar_label_collision(
    *,
    layout: tuple[int, int, str],
    contribution: float,
    contribution_upper_limit: float,
    csf_percentage: float,
    percentage_upper_limit: float,
) -> tuple[int, int, str]:
    if layout[0] != 0 or contribution_upper_limit <= 0 or percentage_upper_limit <= 0:
        return layout
    contribution_fraction = contribution / contribution_upper_limit
    percentage_fraction = csf_percentage / percentage_upper_limit
    if abs(contribution_fraction - percentage_fraction) >= 0.08:
        return layout
    return 0, layout[1] + 7, "center"


def _mid_coupling_bar_colors(labels: list[str]) -> list[str]:
    """Assign the requested four-color palette by displayed bar position."""
    return [
        _OTHERS_COLOR
        if label == "Others"
        else _CHANNEL_COLORS[index % len(_CHANNEL_COLORS)]
        for index, label in enumerate(labels)
    ]


def plot_mid_coupling_contribution(
    series: MidCouplingPlotSeries,
    output_path: Path,
) -> Path:
    """Render and save one adaptive dual-axis intermediate-coupling plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_positions = np.arange(len(series.labels))
    colors = _mid_coupling_bar_colors(series.labels)
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "text.color": _TEXT_COLOR,
            "axes.edgecolor": _AXIS_COLOR,
            "axes.labelcolor": _TEXT_COLOR,
            "xtick.color": _TEXT_COLOR,
            "ytick.color": _TEXT_COLOR,
            "figure.facecolor": _BACKGROUND_COLOR,
            "axes.facecolor": _BACKGROUND_COLOR,
            "savefig.facecolor": _BACKGROUND_COLOR,
        }
    ):
        figure, percentage_axis = plt.subplots(figsize=(7.0, 5.2))
        bars = percentage_axis.bar(
            x_positions,
            series.csf_percentages,
            color=colors,
            alpha=1.0,
            edgecolor="none",
            linewidth=0,
        )
        percentage_axis.set_xticks(x_positions, series.labels, rotation=45, ha="right")
        percentage_axis.set_xlabel("Intermediate coupling channel", fontsize=12)
        percentage_axis.set_ylabel("Percentage Share (%)", fontsize=12)
        percentage_axis.grid(
            axis="y",
            color=_GRID_COLOR,
            alpha=1.0,
            linewidth=0.5,
        )
        percentage_axis.set_axisbelow(True)
        percentage_upper_limit = max(series.csf_percentages, default=1.0) * 1.15
        percentage_axis.set_ylim(0.0, percentage_upper_limit)

        contribution_axis = percentage_axis.twinx()
        contribution_axis.plot(
            x_positions,
            series.contributions,
            color=_LINE_COLOR,
            linestyle="-.",
            marker="o",
            markerfacecolor=_BACKGROUND_COLOR,
            linewidth=2,
            alpha=0.75,
            label="Contribution Value",
        )
        contribution_axis.set_ylabel(
            "Contribution Value", color=_TEXT_COLOR, fontsize=12
        )
        contribution_axis.tick_params(axis="y", labelcolor=_TEXT_COLOR)
        contribution_upper_limit = _contribution_axis_upper_limit(series.contributions)
        contribution_axis.set_ylim(0.0, contribution_upper_limit)

        for index, bar in enumerate(bars):
            center = bar.get_x() + bar.get_width() / 2
            layout = _contribution_label_layout(series.contributions, index)
            x_offset, y_offset, alignment = _avoid_bar_label_collision(
                layout=layout,
                contribution=series.contributions[index],
                contribution_upper_limit=contribution_upper_limit,
                csf_percentage=series.csf_percentages[index],
                percentage_upper_limit=percentage_upper_limit,
            )
            separate_labels = y_offset > layout[1]
            percentage_axis.annotate(
                f"{series.csf_percentages[index]:.1f}%",
                (center, bar.get_height()),
                xytext=(8, 4) if separate_labels else (0, 4),
                textcoords="offset points",
                ha="left" if separate_labels else "center",
                va="bottom",
            )
            contribution_axis.annotate(
                f"{series.contributions[index]:.3f}",
                (x_positions[index], series.contributions[index]),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=alignment,
                va="bottom",
                color=_TEXT_COLOR,
            )

        contribution_axis.legend(loc="lower right", bbox_to_anchor=(1, 1), frameon=False)
        figure.tight_layout()
        figure.savefig(output_path, bbox_inches="tight")
        plt.close(figure)
    return output_path


def save_mid_coupling_plots(
    csfs_df: pl.DataFrame,
    mix_data: MixCoefficientData,
    output_dir: Path,
    *,
    asf_row_indices: list[list[int]] | None = None,
    coupling_level: int = 1,
    min_csf_percentage: float = 0.0,
    min_contribution: float = 0.0,
    max_channels: int | None = None,
    formats: tuple[str, ...] = ("svg",),
) -> list[Path]:
    """Group CSFs once and render each requested ASF in the specified formats."""
    if not formats or any(fmt not in {"svg", "png", "pdf"} for fmt in formats):
        raise ValueError("formats must contain svg, png or pdf")
    requested_rows = (
        asf_row_indices
        if asf_row_indices is not None
        else [block.asf_row_indices.tolist() for block in mix_data.blocks]
    )
    summary = summarize_coupling_ci_squared(
        csfs_df,
        mix_data,
        [list(dict.fromkeys(rows)) for rows in requested_rows],
        coupling_level,
        include_members=False,
        allow_empty_blocks=True,
    )
    by_asf = summary.partition_by(["block_id", "asf_index"], as_dict=True)
    used_stems: set[str] = set()
    paths: list[Path] = []
    for block, rows in zip(mix_data.blocks, requested_rows, strict=True):
        for row in rows:
            series = _series_from_summary(
                by_asf[(block.block_index, row)],
                block,
                row,
                min_csf_percentage,
                min_contribution,
                max_channels,
            )
            stem = mid_coupling_output_stem(
                parity=block.parity, asf_row_index=row, j_value=block.j_value
            )
            if stem in used_stems:
                stem = f"{stem}_b{block.block_index + 1}"
            used_stems.add(stem)
            for fmt in formats:
                paths.append(
                    plot_mid_coupling_contribution(series, output_dir / f"{stem}.{fmt}")
                )
    return paths


__all__ = [
    "MidCouplingPlotSeries",
    "build_mid_coupling_plot_series",
    "plot_mid_coupling_contribution",
    "mid_coupling_output_stem",
    "save_mid_coupling_plots",
]
