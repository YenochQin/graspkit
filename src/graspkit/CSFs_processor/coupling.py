# -*- encoding: utf-8 -*-
"""Polars coupling-signature analysis for rCSFs DataFrames."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from numpy.typing import NDArray
import polars as pl

from ..utils.data_modules import MixCoefficientData
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
) -> list[tuple[int, list[int], NDArray[np.float64]]]:
    """Validate J/CSF alignment and return selected ASF coefficient matrices."""
    normalized_positions = normalize_asf_row_indices(
        asfs_mix_data,
        asf_row_indices,
    )
    dataframe_block_ids = (
        csfs_df.get_column("block_id").unique(maintain_order=True).to_list()
    )
    mix_block_ids = [block.block_index for block in asfs_mix_data.blocks]
    if dataframe_block_ids != mix_block_ids:
        raise ValueError(
            f"CSF block IDs {dataframe_block_ids} 与 rmix block IDs {mix_block_ids} 不一致"
        )

    selected_blocks: list[tuple[int, list[int], NDArray[np.float64]]] = []
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

        dataframe_block = csfs_df.filter(
            pl.col("block_id") == mix_block.block_index
        )
        dataframe_csf_count = dataframe_block.height
        if coefficient_matrix.shape[1] != dataframe_csf_count:
            raise ValueError(
                f"Block {mix_block.block_index}: rmix CSF 数量 {coefficient_matrix.shape[1]} 与 CSF DataFrame {dataframe_csf_count} 不一致"
            )

        csf_two_j_values = (
            dataframe_block.get_column(_COUPLING_SIGNATURE_COLUMN)
            .list.last()
            .unique(maintain_order=True)
            .to_list()
        )
        if len(csf_two_j_values) != 1:
            raise ValueError(
                f"Block {mix_block.block_index}: CSF DataFrame 包含多个总 J: "
                f"2J={csf_two_j_values}"
            )
        rmix_two_j = _twice_j_value(mix_j_value, block_index=mix_block.block_index)
        csf_two_j = int(csf_two_j_values[0])
        if csf_two_j != rmix_two_j:
            raise ValueError(
                f"Block {mix_block.block_index}: CSF J={_format_twice_j(csf_two_j)} "
                f"与 rmix J={mix_j_value} 不一致"
            )

        selected_blocks.append(
            (
                mix_block.block_index,
                selected_positions,
                coefficient_matrix[selected_positions],
            )
        )
    return selected_blocks


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


def _mixing_coefficients_to_long_frame(
    csfs_df: pl.DataFrame,
    asfs_mix_data: MixCoefficientData,
    asf_row_indices: list[list[int]] | None,
) -> pl.DataFrame:
    """Convert selected ASF-by-CSF matrices to one Polars long frame."""
    long_frames: list[pl.DataFrame] = []
    for (
        block_index,
        selected_positions,
        selected_coefficients,
    ) in _selected_block_coefficients(csfs_df, asfs_mix_data, asf_row_indices):
        selected_count = len(selected_positions)
        csf_count = selected_coefficients.shape[1]
        long_frames.append(
            pl.DataFrame(
                {
                    "block_id": np.full(
                        selected_count * csf_count,
                        block_index,
                        dtype=np.uint32,
                    ),
                    "asf_index": np.repeat(
                        np.asarray(selected_positions, dtype=np.int64),
                        csf_count,
                    ),
                    "block_csf_idx": np.tile(
                        np.arange(csf_count, dtype=np.uint64),
                        selected_count,
                    ),
                    "ci_squared": np.square(
                        selected_coefficients,
                        dtype=np.float64,
                    ).reshape(-1),
                },
                schema={
                    "block_id": pl.UInt32,
                    "asf_index": pl.Int64,
                    "block_csf_idx": pl.UInt64,
                    "ci_squared": pl.Float64,
                },
            )
        )

    if not long_frames:
        raise ValueError("没有可汇总的 mixing coefficient block")
    return pl.concat(long_frames, how="vertical")


def summarize_coupling_ci_squared(
    csfs_df: pl.DataFrame,
    asfs_mix_data: MixCoefficientData,
    asf_row_indices: list[list[int]] | None = None,
    coupling_level: int | None = None,
) -> pl.DataFrame:
    """Summarize CI-square contributions by block, ASF, and coupling pattern.

    The returned frame is in long form: one row represents one selected ASF
    and one coupling signature in one block. This keeps the ASF identity
    explicit instead of encoding aligned sums in nested dictionaries.
    """
    _validate_coupling_frame(csfs_df)
    annotated = _with_selected_coupling(csfs_df, coupling_level).select(
        "block_id",
        "block_csf_idx",
        pl.col("idx").cast(pl.UInt64()),
        _SELECTED_COUPLING_COLUMN,
    )
    coefficient_frame = _mixing_coefficients_to_long_frame(
        csfs_df,
        asfs_mix_data,
        asf_row_indices,
    )
    joined = coefficient_frame.join(
        annotated,
        on=["block_id", "block_csf_idx"],
        how="inner",
        validate="m:1",
        maintain_order="left",
    )
    if joined.height != coefficient_frame.height:
        raise ValueError("mixing coefficient 行与 CSF DataFrame 未能完整对齐")

    return (
        joined.group_by(
            ["block_id", "asf_index", _SELECTED_COUPLING_COLUMN],
            maintain_order=True,
        )
        .agg(
            pl.len().alias("count"),
            pl.col("block_csf_idx").alias("idxs"),
            pl.col("idx").alias("global_idxs"),
            pl.col("ci_squared").sum().alias("sum_ci"),
        )
        .rename({_SELECTED_COUPLING_COLUMN: _COUPLING_SIGNATURE_COLUMN})
        .select(
            "block_id",
            "asf_index",
            _COUPLING_SIGNATURE_COLUMN,
            "count",
            "idxs",
            "global_idxs",
            "sum_ci",
        )
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
