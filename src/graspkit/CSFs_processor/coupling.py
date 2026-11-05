# -*- encoding: utf-8 -*-
"""Polars coupling-signature analysis for rCSFs DataFrames."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
import polars as pl

from ..utils.data_modules import MixCoefficientData
from .selection import select_csf_indices_by_cumulative_weight
from .validation import (
    normalize_asf_row_indices,
    validate_coupling_level,
)

_COUPLING_SIGNATURE_COLUMN = "coupling_signature"
_SELECTED_COUPLING_COLUMN = "_selected_coupling"


def _validate_coupling_frame(csfs_df: pl.DataFrame) -> None:
    """Validate the rCSFs DataFrame contract used by coupling analysis."""
    required_columns = {"idx", "block_id", _COUPLING_SIGNATURE_COLUMN}
    missing_columns = required_columns.difference(csfs_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSF DataFrame 缺少必要列: {missing}")

    if csfs_df.schema[_COUPLING_SIGNATURE_COLUMN] != pl.List(pl.Int32):
        raise ValueError("coupling_signature 必须是 Polars List(Int32) 列")
    if not csfs_df.schema["block_id"].is_integer():
        raise ValueError("block_id 必须是整数列")
    if not csfs_df.schema["idx"].is_integer():
        raise ValueError("idx 必须是整数列")
    if csfs_df.is_empty():
        raise ValueError("CSF DataFrame 不能为空")

    for column in ("idx", "block_id", _COUPLING_SIGNATURE_COLUMN):
        if csfs_df[column].null_count() != 0:
            raise ValueError(f"{column} 不能包含 null")

    has_empty_signature = bool(
        csfs_df.select(pl.col(_COUPLING_SIGNATURE_COLUMN).list.len().eq(0).any()).item()
    )
    if has_empty_signature:
        raise ValueError("coupling_signature 不能是空列表")

    has_null_item = bool(
        csfs_df.select(
            pl.col(_COUPLING_SIGNATURE_COLUMN)
            .list.eval(pl.element().is_null())
            .list.any()
            .any()
        ).item()
    )
    if has_null_item:
        raise ValueError("coupling_signature 不能包含 null 元素")


def _with_selected_coupling(
    csfs_df: pl.DataFrame,
    coupling_level: int | None,
) -> pl.DataFrame:
    """Add block-local row indices and the requested intermediate coupling."""
    coupling_level = validate_coupling_level(coupling_level)
    selected_coupling = pl.col(_COUPLING_SIGNATURE_COLUMN)
    if coupling_level is not None:
        intermediate_coupling = selected_coupling.list.slice(
            0,
            selected_coupling.list.len() - 1,
        )
        selected_coupling = intermediate_coupling.list.tail(coupling_level)

    return csfs_df.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt64).over("block_id").alias("block_csf_idx"),
        selected_coupling.alias(_SELECTED_COUPLING_COLUMN),
    )


def collect_coupling_groups(
    csfs_df: pl.DataFrame,
    coupling_level: int | None = None,
) -> pl.DataFrame:
    """Group rCSFs coupling signatures with local and global CSF indices.

    Args:
        csfs_df: DataFrame returned by ``rcsfs.read_csfs`` with both
            ``include_block_id`` and ``include_coupling_signature`` enabled.
        coupling_level: Number of trailing intermediate-coupling ``2J`` values
            used as the grouping pattern. The signature's final total J value
            is excluded before counting. ``None`` retains the complete
            signature, including total J.

    Returns:
        One row per block and coupling pattern with columns ``block_id``,
        ``coupling_signature``, ``count``, block-local ``idxs``, and source
        ``global_idxs``.

    Raises:
        ValueError: If the DataFrame does not satisfy the rCSFs column
            contract or ``coupling_level`` is invalid.
    """
    _validate_coupling_frame(csfs_df)
    annotated = _with_selected_coupling(csfs_df, coupling_level)

    return (
        annotated.group_by(
            ["block_id", _SELECTED_COUPLING_COLUMN],
            maintain_order=True,
        )
        .agg(
            pl.len().alias("count"),
            pl.col("block_csf_idx").alias("idxs"),
            pl.col("idx").cast(pl.UInt64()).alias("global_idxs"),
        )
        .rename({_SELECTED_COUPLING_COLUMN: _COUPLING_SIGNATURE_COLUMN})
        .select(
            "block_id",
            _COUPLING_SIGNATURE_COLUMN,
            "count",
            "idxs",
            "global_idxs",
        )
    )


def _selected_block_coefficients(
    csfs_df: pl.DataFrame,
    asfs_mix_data: MixCoefficientData,
    asf_row_indices: list[list[int]] | None,
    *,
    allow_empty_blocks: bool = False,
) -> Iterator[tuple[int, list[int], NDArray[np.float64]]]:
    """Validate J/CSF alignment and return selected ASF coefficient matrices."""
    normalized_positions = normalize_asf_row_indices(
        asfs_mix_data,
        asf_row_indices,
        allow_empty_blocks=allow_empty_blocks,
    )
    dataframe_block_ids = (
        csfs_df.get_column("block_id").unique(maintain_order=True).to_list()
    )
    mix_block_ids = [block.block_index for block in asfs_mix_data.blocks]
    if dataframe_block_ids != mix_block_ids:
        raise ValueError(
            f"CSF block IDs {dataframe_block_ids} 与 rmix block IDs {mix_block_ids} 不一致"
        )

    for mix_block, mix_j_value, selected_positions in zip(
        asfs_mix_data.blocks,
        asfs_mix_data.level_J_value_list,
        normalized_positions,
        strict=True,
    ):
        coefficient_matrix = np.asarray(
            mix_block.mix_coefficients,
            dtype=np.float64,
        )
        if coefficient_matrix.ndim != 2:
            raise ValueError(
                f"Block {mix_block.block_index}: mixing coefficients 必须是二维矩阵"
            )

        signatures = csfs_df[_COUPLING_SIGNATURE_COLUMN].filter(
            csfs_df["block_id"] == mix_block.block_index
        )
        dataframe_csf_count = len(signatures)
        if coefficient_matrix.shape[1] != dataframe_csf_count:
            raise ValueError(
                f"Block {mix_block.block_index}: rmix CSF 数量 {coefficient_matrix.shape[1]} 与 CSF DataFrame {dataframe_csf_count} 不一致"
            )

        csf_two_j_values = signatures.list.last().unique(maintain_order=True).to_list()
        if len(csf_two_j_values) != 1:
            raise ValueError(
                f"Block {mix_block.block_index}: CSF DataFrame 包含多个总 J: 2J={csf_two_j_values}"
            )
        rmix_two_j = _twice_j_value(mix_j_value, block_index=mix_block.block_index)
        csf_two_j = int(csf_two_j_values[0])
        if csf_two_j != rmix_two_j:
            raise ValueError(
                f"Block {mix_block.block_index}: CSF J={_format_twice_j(csf_two_j)} 与 rmix J={mix_j_value} 不一致"
            )

        yield (
            mix_block.block_index,
            selected_positions,
            coefficient_matrix[selected_positions],
        )


def _twice_j_value(j_value: str, *, block_index: int) -> int:
    """Convert an rmix integer or half-integer J label to integral ``2J``."""
    try:
        twice_j = Fraction(j_value) * 2
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"Block {block_index}: rmix J 值格式无效: {j_value!r}"
        ) from error
    if twice_j.denominator != 1 or twice_j.numerator < 0:
        raise ValueError(f"Block {block_index}: rmix J 值无效: {j_value!r}")
    return twice_j.numerator


def _format_twice_j(two_j: int) -> str:
    """Format integral ``2J`` using the same labels as MixCoefficientData."""
    if two_j % 2 == 0:
        return str(two_j // 2)
    return f"{two_j}/2"


@dataclass(frozen=True)
class _WeightedCouplingBlock:
    block_id: int
    asf_rows: list[int]
    groups: pl.DataFrame
    ci_squared: NDArray[np.float64]
    channel_sums: NDArray[np.float64]


def _weighted_coupling_blocks(
    csfs_df: pl.DataFrame,
    mix_data: MixCoefficientData,
    asf_row_indices: list[list[int]] | None,
    coupling_level: int | None,
    *,
    allow_empty_blocks: bool = False,
) -> Iterator[_WeightedCouplingBlock]:
    """Group each CSF once, then sum each ASF over the same source-ordered members."""
    groups = collect_coupling_groups(csfs_df, coupling_level)
    for block_id, rows, coefficients in _selected_block_coefficients(
        csfs_df, mix_data, asf_row_indices, allow_empty_blocks=allow_empty_blocks
    ):
        if not rows:
            continue
        block_groups = groups.filter(pl.col("block_id") == block_id)
        members = block_groups["idxs"].explode().to_numpy()
        counts = block_groups["count"].to_numpy()
        starts = np.cumsum(counts, dtype=np.int64) - counts
        weights = np.square(coefficients)
        # Reduce one ASF at a time to bound temporary storage by the block size.
        sums = np.stack([np.add.reduceat(row[members], starts) for row in weights])
        yield _WeightedCouplingBlock(block_id, rows, block_groups, weights, sums)


def summarize_coupling_ci_squared(
    csfs_df: pl.DataFrame,
    asfs_mix_data: MixCoefficientData,
    asf_row_indices: list[list[int]] | None = None,
    coupling_level: int | None = None,
    *,
    include_members: bool = True,
    allow_empty_blocks: bool = False,
) -> pl.DataFrame:
    """Return channel counts and CI² sums in block, ASF and source-channel order.

    By default also return the block-local and source indices of each member.
    For plots, omit the member lists and allow empty ASF lists to skip blocks.
    """
    frames: list[pl.DataFrame] = []
    for block in _weighted_coupling_blocks(
        csfs_df,
        asfs_mix_data,
        asf_row_indices,
        coupling_level,
        allow_empty_blocks=allow_empty_blocks,
    ):
        groups = (
            block.groups
            if include_members
            else block.groups.drop("idxs", "global_idxs")
        )
        for asf_index, channel_sums in zip(
            block.asf_rows, block.channel_sums, strict=True
        ):
            frames.append(
                groups.with_columns(
                    pl.lit(asf_index, dtype=pl.Int64).alias("asf_index"),
                    pl.Series("sum_ci", channel_sums),
                )
            )
    schema: dict[str, type[pl.DataType] | pl.DataType] = {
        "block_id": pl.UInt32,
        "asf_index": pl.Int64,
        "coupling_signature": pl.List(pl.Int32),
        "count": pl.UInt32,
    }
    if include_members:
        schema.update(idxs=pl.List(pl.UInt64), global_idxs=pl.List(pl.UInt64))
    schema["sum_ci"] = pl.Float64
    if not frames:
        return pl.DataFrame(schema=schema)
    return pl.concat(frames).select(list(schema)).cast({"block_id": pl.UInt32})


@dataclass(frozen=True)
class CumulativeCouplingSelection:
    """Source-ordered selection and the statistics used to construct it."""

    selected_csfs: pl.DataFrame
    prefix_counts: dict[int, int]
    dominant_channels: pl.DataFrame


def select_csfs_by_cumulative_coupling(
    csfs_df: pl.DataFrame,
    mix_data: MixCoefficientData,
    *,
    cumulative_threshold: float,
    coupling_level: int,
    asf_row_indices: list[list[int]] | None = None,
) -> CumulativeCouplingSelection:
    """Union each selected ASF's normalized CI² prefix and its dominant channel.

    Prefix ties follow select_csf_indices_by_cumulative_weight. Equal channel
    sums select the first channel in CSF source order, as in ML presampling.
    """
    selected = np.zeros(csfs_df.height, dtype=np.bool_)
    prefix_counts: dict[int, int] = {}
    dominant_frames: list[pl.DataFrame] = []
    for block in _weighted_coupling_blocks(
        csfs_df, mix_data, asf_row_indices, coupling_level
    ):
        prefix = select_csf_indices_by_cumulative_weight(
            block.ci_squared, cumulative_threshold=cumulative_threshold
        )
        prefix_counts[block.block_id] = len(prefix.indices)
        winners = np.argmax(block.channel_sums, axis=1)
        block_mask = np.zeros(block.ci_squared.shape[1], dtype=np.bool_)
        block_mask[prefix.indices] = True
        theme_members = (
            block.groups["idxs"].gather(np.unique(winners)).explode().to_numpy()
        )
        block_mask[theme_members] = True
        selected[csfs_df["block_id"].to_numpy() == block.block_id] = block_mask
        dominant_frames.append(
            block.groups.select("block_id", "coupling_signature", "count")[
                winners
            ].with_columns(
                pl.Series("asf_index", block.asf_rows, dtype=pl.Int64),
                pl.Series(
                    "sum_ci", block.channel_sums[np.arange(len(winners)), winners]
                ),
            )
        )
    return CumulativeCouplingSelection(
        csfs_df.filter(selected), prefix_counts, pl.concat(dominant_frames)
    )


def select_csfs_by_coupling_theme(
    csfs_df: pl.DataFrame,
    asfs_mix_data: MixCoefficientData,
    *,
    asf_row_indices: list[list[int]] | None = None,
    ci_squared_cutoff: float,
    coupling_level: int,
) -> pl.DataFrame:
    """Select rCSFs rows by CI-square threshold and dominant coupling theme.

    The input must be returned by :func:`rcsfs.read_csfs` with both
    ``include_block_id`` and ``include_coupling_signature`` enabled. For every
    block, the result is the source-ordered union of CSFs whose selected-ASF
    CI-square exceeds ``ci_squared_cutoff`` and every member of the dominant coupling
    group for each selected ASF.

    ``coupling_level`` is the positive number of trailing intermediate-coupling
    integer ``2J`` values retained from the rCSFs fixed-width coupling
    signature. The final total J value is excluded before counting.
    """
    validate_coupling_level(coupling_level)
    if type(ci_squared_cutoff) not in (int, float):
        raise ValueError("ci_squared_cutoff must be a finite non-negative number")
    cutoff = float(ci_squared_cutoff)
    if not np.isfinite(cutoff) or cutoff < 0:
        raise ValueError("ci_squared_cutoff must be a finite non-negative number")

    coupling_groups = collect_coupling_groups(csfs_df, coupling_level)
    groups_by_block: dict[int, list[NDArray[np.int64]]] = {}
    for block_id, idxs in coupling_groups.select("block_id", "idxs").iter_rows():
        groups_by_block.setdefault(int(block_id), []).append(
            np.asarray(idxs, dtype=np.int64)
        )

    block_ids = csfs_df.get_column("block_id").to_numpy()
    selected_row_positions: list[int] = []
    for block_id, _, selected_coefficients in _selected_block_coefficients(
        csfs_df,
        asfs_mix_data,
        asf_row_indices,
    ):
        scores = np.square(selected_coefficients, dtype=np.float64)
        block_groups = groups_by_block.get(block_id)
        if not block_groups:
            raise ValueError(f"CSF block {block_id} contains no coupling themes")

        selected_local = set(
            np.flatnonzero(np.any(scores > cutoff, axis=0)).astype(int).tolist()
        )
        group_sums = np.column_stack(
            [np.sum(scores[:, idxs], axis=1) for idxs in block_groups]
        )
        for dominant_group_index in np.argmax(group_sums, axis=1):
            selected_local.update(block_groups[int(dominant_group_index)].tolist())

        block_row_positions = np.flatnonzero(block_ids == block_id)
        selected_row_positions.extend(
            block_row_positions[sorted(selected_local)].astype(int).tolist()
        )

    return csfs_df[np.asarray(sorted(selected_row_positions), dtype=np.int64)]
