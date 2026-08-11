# -*- encoding: utf-8 -*-
"""Polars coupling-signature analysis with legacy raw-line compatibility."""

from __future__ import annotations

import logging
from collections import Counter
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
import polars as pl

from ..utils.data_modules import MixCoefficientData
from .validation import (
    normalize_asf_positions,
    validate_coupling_level,
    validate_csf_records,
)

logger = logging.getLogger(__name__)

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
    """Add block-local row indices and the requested trailing signature."""
    coupling_level = validate_coupling_level(coupling_level)
    selected_coupling = pl.col(_COUPLING_SIGNATURE_COLUMN)
    if coupling_level is not None:
        selected_coupling = selected_coupling.list.slice(-coupling_level)

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
        coupling_level: Number of trailing coupling ``2J`` values used as the
            grouping pattern. ``None`` retains the complete signature.

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
    asfs_position: list[list[int]] | None,
) -> list[tuple[int, list[int], NDArray[np.float64]]]:
    """Validate block alignment and return selected ASF coefficient matrices."""
    normalized_positions = normalize_asf_positions(
        asfs_mix_data,
        asfs_position,
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
    for mix_block, selected_positions in zip(
        asfs_mix_data.blocks,
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

        dataframe_csf_count = csfs_df.filter(
            pl.col("block_id") == mix_block.block_index
        ).height
        if coefficient_matrix.shape[1] != dataframe_csf_count:
            raise ValueError(
                f"Block {mix_block.block_index}: rmix CSF 数量 {coefficient_matrix.shape[1]} 与 CSF DataFrame {dataframe_csf_count} 不一致"
            )

        selected_blocks.append(
            (
                mix_block.block_index,
                selected_positions,
                coefficient_matrix[selected_positions],
            )
        )
    return selected_blocks


def _mixing_coefficients_to_long_frame(
    csfs_df: pl.DataFrame,
    asfs_mix_data: MixCoefficientData,
    asfs_position: list[list[int]] | None,
) -> pl.DataFrame:
    """Convert selected ASF-by-CSF matrices to one Polars long frame."""
    long_frames: list[pl.DataFrame] = []
    for (
        block_index,
        selected_positions,
        selected_coefficients,
    ) in _selected_block_coefficients(csfs_df, asfs_mix_data, asfs_position):
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
    asfs_position: list[list[int]] | None = None,
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
        asfs_position,
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


class CouplingJInfo(TypedDict):
    """Summary for CSFs sharing the same final coupling-J pattern.

    Attributes:
        count: Number of CSFs that match the coupling pattern.
        idxs: CSF indices in the source block.
    """

    count: int
    idxs: list[int]


class CouplingJInfoWithSumCi(TypedDict):
    """Coupling-J summary with one accumulated CI-square contribution.

    Attributes:
        count: Number of CSFs that match the coupling pattern.
        idxs: CSF indices in the source block.
        sum_ci: Sum of squared CI coefficients for the matching CSFs.
    """

    count: int
    idxs: list[int]
    sum_ci: float


class CouplingJInfoWithSumCiList(TypedDict):
    """Coupling-J summary with CI-square contributions for multiple ASFs.

    Attributes:
        count: Number of CSFs that match the coupling pattern.
        idxs: CSF indices in the source block.
        sum_ci: Per-selected-ASF sums of squared CI coefficients.
    """

    count: int
    idxs: list[int]
    sum_ci: list[float]


def select_csfs_by_coupling_theme(
    csfs_df: pl.DataFrame,
    asfs_mix_data: MixCoefficientData,
    *,
    asfs_position: list[list[int]] | None = None,
    ci_squared_cutoff: float,
    coupling_level: int,
) -> pl.DataFrame:
    """Select rCSFs rows by CI-square threshold and dominant coupling theme.

    The input must be returned by :func:`rcsfs.read_csfs` with both
    ``include_block_id`` and ``include_coupling_signature`` enabled. For every
    block, the result is the source-ordered union of CSFs whose selected-ASF
    CI-square exceeds ``ci_squared_cutoff`` and every member of the dominant coupling
    group for each selected ASF.

    ``coupling_level`` is the positive number of trailing integer ``2J`` values
    retained from the rCSFs fixed-width coupling signature.
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
        asfs_position,
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


def single_block_csfs_final_coupling_J_collector(
    block_csfs: list[list[str]], coupling_level: int | None = None
) -> dict[tuple[str, ...], CouplingJInfo]:
    """Collect whitespace tokens from legacy raw ``line3`` CSF records.

    Prefer :func:`collect_coupling_groups` for DataFrames produced by rCSFs.
    This compatibility function does not perform GRASP fixed-width coupling
    parsing and therefore cannot recover the complete coupling signature.

    Args:
        block_csfs: CSF records whose third line contains whitespace-separated
            coupling tokens.
        coupling_level: Number of trailing coupling tokens to keep. If None,
            all tokens are used. Shorter CSF records fall back to all tokens.

    Returns:
        Mapping from coupling-token pattern to its occurrence count and CSF
        indices.

    Raises:
        ValueError: If ``coupling_level`` is not ``None`` or positive, or a
            CSF record does not have exactly 3 lines.
    """
    coupling_level = validate_coupling_level(coupling_level)
    validate_csf_records(block_csfs)
    selected_tokens = [
        tuple(csf[2].lstrip().split())[-coupling_level:]
        if coupling_level is not None
        else tuple(csf[2].lstrip().split())
        for csf in block_csfs
    ]
    counts = Counter(selected_tokens)
    groups: dict[tuple[str, ...], CouplingJInfo] = {
        pattern: {"count": count, "idxs": []} for pattern, count in counts.items()
    }
    for index, pattern in enumerate(selected_tokens):
        groups[pattern]["idxs"].append(index)
    return groups


def batch_blocks_csfs_final_coupling_J_collection(
    blocks_csfs_list: list[list[list[str]]], coupling_level: int | None = None
) -> dict[int, dict[tuple[str, ...], CouplingJInfo]]:
    """Collect final coupling-J patterns for every CSF block.

    Args:
        blocks_csfs_list: CSF blocks, where each block contains three-line CSF
            records.
        coupling_level: Number of trailing coupling tokens to keep for each
            CSF pattern.

    Returns:
        Dictionary keyed by block index with per-pattern coupling summaries.
    """
    blocks_coupling_J_collection: dict[int, dict[tuple[str, ...], CouplingJInfo]] = {}
    for block, block_csfs in enumerate(blocks_csfs_list):
        logger.info(f"Block {block + 1}: 包含 {len(block_csfs)} 个 CSF")
        block_coupling_J_collection = single_block_csfs_final_coupling_J_collector(
            block_csfs, coupling_level
        )
        blocks_coupling_J_collection[block] = block_coupling_J_collection
    return blocks_coupling_J_collection


def single_asf_csfs_final_coupling_J_mix_coefficient_sum(
    block_csfs_coupling_J_collection_dict: dict[tuple[str, ...], CouplingJInfo],
    mix_coefficient_list: list[float] | np.ndarray,
) -> dict[tuple[str, ...], CouplingJInfoWithSumCi]:
    """Sum squared CI coefficients for each coupling-J pattern in one ASF.

    Args:
        block_csfs_coupling_J_collection_dict: Coupling summaries for a CSF
            block.
        mix_coefficient_list: CI coefficients aligned with the CSFs in the
            same block.

    Returns:
        Coupling summaries augmented with the summed squared CI contribution.
    """
    coeff_array = np.asarray(mix_coefficient_list)
    result: dict[tuple[str, ...], CouplingJInfoWithSumCi] = {}
    for pattern, info in block_csfs_coupling_J_collection_dict.items():
        idxs = info["idxs"]
        sum_ci = float(np.sum(coeff_array[idxs] ** 2))
        logger.debug(f"{pattern=}  count={info['count']}  sum_ci={sum_ci}")
        result[pattern] = {"count": info["count"], "idxs": idxs, "sum_ci": sum_ci}
    return result


def single_block_batch_asfs_CSFs_final_coupling_J_collection(
    block_CSFs: list[list[str]],
    block_asfs_mix_coefficient_list: list[np.ndarray] | np.ndarray,
    block_asfs_position: list[int] | np.ndarray | None = None,
    coupling_level: int | None = None,
) -> dict[tuple[str, ...], CouplingJInfoWithSumCiList]:
    """Collect coupling-J contributions for selected ASFs in one CSF block.

    Args:
        block_CSFs: CSF records for a single block.
        block_asfs_mix_coefficient_list: Matrix-like ASF by CSF CI
            coefficients for the block.
        block_asfs_position: ASF indices to include. If None, all ASFs are
            included.
        coupling_level: Number of trailing coupling tokens used to define each
            coupling pattern.

    Returns:
        Mapping from coupling pattern to counts, CSF indices, and
        per-selected-ASF summed squared CI coefficients.
    """
    normalized_positions: NDArray[np.int64]
    if block_asfs_position is None:
        normalized_positions = np.arange(
            len(block_asfs_mix_coefficient_list), dtype=np.int64
        )
    else:
        normalized_positions = np.asarray(block_asfs_position, dtype=np.int64)

    coeff_matrix: NDArray[np.float64] = np.asarray(
        block_asfs_mix_coefficient_list, dtype=np.float64
    )

    base_coupling_dict = single_block_csfs_final_coupling_J_collector(
        block_CSFs, coupling_level
    )
    result: dict[tuple[str, ...], CouplingJInfoWithSumCiList] = {}
    for pattern, info in base_coupling_dict.items():
        idxs: NDArray[np.int64] = np.asarray(info["idxs"], dtype=np.int64)
        sum_ci_list: list[float] = []
        for asf_idx in normalized_positions:
            asf_coeff = coeff_matrix[int(asf_idx)]
            sum_ci_list.append(float(np.sum(asf_coeff[idxs] ** 2, dtype=np.float64)))
        result[pattern] = {
            "count": info["count"],
            "idxs": info["idxs"],
            "sum_ci": sum_ci_list,
        }

    return result


def batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum(
    blocks_CSFs_list: list[list[list[str]]],
    asfs_mix_data: MixCoefficientData,
    asfs_position: list[list[int]] | None = None,
    coupling_level: int | None = None,
) -> dict[int, dict[tuple[str, ...], CouplingJInfoWithSumCiList]]:
    """Collect coupling-J CI-square summaries for all CSF blocks.

    Args:
        blocks_CSFs_list: CSF blocks aligned with ``asfs_mix_data``.
        asfs_mix_data: Parsed ASF mixing-coefficient data.
        asfs_position: Optional selected ASF indices for each block. Defaults
            to each block's ``level_indices``.
        coupling_level: Number of trailing coupling tokens used to define each
            coupling pattern.

    Returns:
        Nested dictionary keyed first by block index and then by coupling
        pattern.

    Raises:
        ValueError: If ASF positions are inconsistent with the mixing data or
            CSF and coefficient lengths do not match.
    """
    if asfs_position is None:
        normalized_positions: list[list[int]] = [
            block.level_indices.astype(np.int64).tolist()
            for block in asfs_mix_data.blocks
        ]
    else:
        normalized_positions = [
            [int(pos) for pos in block_positions] for block_positions in asfs_position
        ]

    if len(normalized_positions) != len(asfs_mix_data.blocks):
        raise ValueError(
            f"asfs_position 第一层长度 {len(normalized_positions)} 与 blocks {len(asfs_mix_data.blocks)} 不一致。"
        )

    if len(blocks_CSFs_list) != len(asfs_mix_data.blocks):
        raise ValueError(
            f"blocks_CSFs_list 长度 {len(blocks_CSFs_list)} 与 mix blocks {len(asfs_mix_data.blocks)} 不一致。"
        )

    blocks_asfs_coupling_J_sum_ci: dict[
        int, dict[tuple[str, ...], CouplingJInfoWithSumCiList]
    ] = {}
    for block_offset, (block_csfs, mix_block, selected_positions) in enumerate(
        zip(blocks_CSFs_list, asfs_mix_data.blocks, normalized_positions, strict=True)
    ):
        allowed_positions = set(mix_block.level_indices.astype(np.int64).tolist())
        selected_set = {int(pos) for pos in selected_positions}
        if not selected_set.issubset(allowed_positions):
            raise ValueError(
                f"asfs_position 第 {block_offset} 层元素 {sorted(selected_set)} 不是 block.level_indices 对应层 {sorted(allowed_positions)} 的子集。"
            )

        logger.info(
            f"Block {mix_block.block_index + 1}: 包含 {len(mix_block.mix_coefficients)} 个 ASF"
        )
        if any(
            len(asf_mix) != len(block_csfs) for asf_mix in mix_block.mix_coefficients
        ):
            raise ValueError(
                f"Block {mix_block.block_index}: block_CSFs 长度 {len(block_csfs)} 与 block_asfs_mix_coefficient 长度不匹配。"
            )

        blocks_asfs_coupling_J_sum_ci[mix_block.block_index] = (
            single_block_batch_asfs_CSFs_final_coupling_J_collection(
                block_CSFs=block_csfs,
                block_asfs_mix_coefficient_list=mix_block.mix_coefficients,
                block_asfs_position=selected_positions,
                coupling_level=coupling_level,
            )
        )

    return blocks_asfs_coupling_J_sum_ci
