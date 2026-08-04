# -*- encoding: utf-8 -*-
"""Deterministic and random CSF selection primitives."""

from __future__ import annotations

import math
import random
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import polars as pl

from ..grasp_data_extractor.rmix_data_processor import (
    RmixCiSquaredData,
    ci_squared,
    filter_ci_scores_by_threshold,
    sort_ci_scores,
)
from ..utils.data_modules import MixCoefficientData
from .validation import validate_selection_idxs


def single_asf_mix_square_above_threshold(
    asf_mix_data_array: np.ndarray, threshold: float = 0.1
) -> list[tuple[int]]:
    """
    获取一维数组中平方值超过阈值的元素的索引，并按绝对值降序排序

    Args:
        asf_mix_data_array: 一维输入数组
        threshold: 阈值(平方值比较)，默认0.1

    Returns:
        包含索引的元组列表，如[(idx1,), (idx2,), ...]，按绝对值降序排列
        如果数组为空或没有元素超过阈值，返回空列表
    """
    if asf_mix_data_array.ndim != 1:
        raise ValueError("输入数组必须是一维的")

    squared_scores = ci_squared(asf_mix_data_array)
    idxs = filter_ci_scores_by_threshold(squared_scores, threshold)

    if len(idxs) == 0:
        return []

    values = np.asarray(asf_mix_data_array)[idxs]
    sorted_idxs = idxs[np.argsort(-np.abs(values))]

    return [(int(idx),) for idx in sorted_idxs]


def batch_asfs_mix_square_above_threshold(
    asfs_mix_data: MixCoefficientData,
    asfs_position: list[list[int]] | None = None,
    threshold: float = 0.1,
) -> dict[int, np.ndarray]:
    """
    批量处理多个块的混合系数数据，找出每个块中所有层级中超过阈值的系数索引

    Args:
        asfs_mix_data: Parsed block-based ASF mixing-coefficient data.
        asfs_position: Optional selected ASF row indices for each block. Defaults
            to each block's ``level_indices``.
        threshold: 阈值(平方值比较)，默认0.1

    Returns:
        字典，键是block编号，值是该块中所有超过阈值的系数索引(已去重)
        如果没有满足条件的索引，对应的值为空数组
    """
    result: dict[int, NDArray[np.int64]] = {}

    if not asfs_position:
        normalized_positions: list[list[int]] = [
            block.level_indices.astype(np.int64).tolist()
            for block in asfs_mix_data.blocks
        ]
    else:
        normalized_positions = [
            [int(pos) for pos in block_positions]
            for block_positions in asfs_position
        ]

    if len(normalized_positions) != len(asfs_mix_data.blocks):
        raise ValueError(
            f"asfs_position 第一层长度({len(normalized_positions)}) "
            f"与 blocks({len(asfs_mix_data.blocks)}) 不一致。"
        )

    for block, selected_positions in zip(
        asfs_mix_data.blocks, normalized_positions, strict=True
    ):
        allowed_positions = set(block.level_indices.astype(np.int64).tolist())
        selected_set = {int(pos) for pos in selected_positions}
        if not selected_set.issubset(allowed_positions):
            raise ValueError(
                f"asfs_position 元素 {sorted(selected_set)} "
                f"不是 block.level_indices {sorted(allowed_positions)} 的子集。"
            )

        block_data = block.mix_coefficients[selected_positions]

        squared_scores = ci_squared(block_data)
        score_matrix = (
            squared_scores
            if squared_scores.ndim == 2
            else np.atleast_2d(squared_scores)
        )

        above_threshold_mask = np.any(score_matrix > threshold, axis=0)
        result[block.block_index] = np.where(above_threshold_mask)[0]

    return result


def CSFs_block_get_CSF(
    CSFs_block: list[list[str]], CSf_idx: list[int] | np.ndarray
) -> list[list[str]]:
    """Select CSF records from a block by index.

    Args:
        CSFs_block: CSF records in one block.
        CSf_idx: Indices of the CSF records to select.

    Returns:
        Selected CSF records in the same order as ``CSf_idx``.
    """
    return [CSFs_block[i] for i in CSf_idx]


def union_lists_with_order(*lists: list[int | str]) -> list[int | str]:
    """Return the ordered union of multiple lists.

    Args:
        *lists: Lists whose elements should be merged.

    Returns:
        De-duplicated list that preserves the first occurrence order.
    """
    all_elements: list[int | str] = []
    for lst in lists:
        all_elements.extend(lst)
    return list(dict.fromkeys(all_elements))


def CSFs_sort_by_mix_coefficient(
    CSFs_block: list[list[str]],
    mix_coefficients: np.ndarray,
    threshold: float | None = None
) -> list[list[str]]:
    """Sort CSFs by the summed square of their mixing coefficients.

    Args:
        CSFs_block: CSF records in one block.
        mix_coefficients: 1D (single ASF) or 2D (multiple ASFs) coefficient
            array whose last axis matches ``CSFs_block``.
        threshold: Optional coefficient cutoff. When provided, only CSFs whose
            combined squared coefficient is above ``threshold ** 2`` are kept.

    Returns:
        CSF records sorted by descending combined squared coefficient.

    Raises:
        ValueError: If the CSF block or coefficient array is empty, or if their
            lengths do not match.
    """
    if len(CSFs_block) == 0 or len(mix_coefficients) == 0:
        raise ValueError("CSFs_block和mix_coefficients不能为空")

    coeff_array = np.atleast_2d(np.asarray(mix_coefficients))
    if coeff_array.shape[-1] != len(CSFs_block):
        raise ValueError("mix_coefficients长度必须与CSFs_block匹配")

    squared_coefficients = ci_squared(coeff_array)
    combined_coeff: NDArray[np.float64] = np.asarray(
        np.sum(squared_coefficients, axis=0, dtype=np.float64),
        dtype=np.float64,
    )
    sorted_idxs, _ = sort_ci_scores(combined_coeff)

    if threshold is not None:
        threshold_idxs = filter_ci_scores_by_threshold(combined_coeff, threshold**2)
        sorted_idxs = sorted_idxs[np.isin(sorted_idxs, threshold_idxs)]

    return [CSFs_block[int(idx)] for idx in sorted_idxs]


def generate_unique_random_numbers(max_num: int, count: int) -> list[int]:
    """Generate unique random positive integers.

    Args:
        max_num: Inclusive upper bound for generated numbers.
        count: Number of unique values to generate.

    Returns:
        List of unique sampled integers in the range ``[1, max_num]``.
    """
    number: list[int] = random.sample(range(1, max_num + 1), count)
    return number


def radom_choose_csfs(
    block_csfs_list: list[list[str]],
    method: Literal["ratio", "quality"],
    ratio_or_quality: float,
    selected_csfs_idxs: list[int] | None = None,
) -> tuple[list[list[str]], NDArray[np.int64], NDArray[np.int64]]:
    """Randomly choose additional CSFs from a block.

    Args:
        block_csfs_list: Candidate CSFs in one block.
        method: Selection mode. ``"ratio"`` treats ``ratio_or_quality`` as a
            fraction of the block size; ``"quality"`` treats it as an absolute
            target count.
        ratio_or_quality: Ratio or target count, depending on ``method``.
        selected_csfs_idxs: Indices that have already been selected.

    Returns:
        Tuple of selected CSF records, selected indices, and unselected indices.

    Raises:
        ValueError: If ``method`` is not ``"ratio"`` or ``"quality"``.
    """
    if selected_csfs_idxs is None:
        selected_csfs_idxs = []

    block_csfs_num = len(block_csfs_list)
    selected_csfs_num = len(selected_csfs_idxs)
    if method == "ratio":
        total_needed = math.ceil(block_csfs_num * ratio_or_quality)
    elif method == "quality":
        total_needed = math.ceil(ratio_or_quality)
    else:
        raise ValueError(f"method 必须是 'ratio' 或 'quality': {method!r}")
    choose_csfs_num = max(0, total_needed - selected_csfs_num)

    all_idxs = np.arange(block_csfs_num, dtype=np.int64)

    if selected_csfs_num > 0:
        selected_set = set(selected_csfs_idxs)
        unselected_mask = ~np.isin(all_idxs, list(selected_set))
        unselected_idxs = all_idxs[unselected_mask]

        if choose_csfs_num > 0:
            random_idxs = np.random.choice(
                unselected_idxs, size=choose_csfs_num, replace=False
            )
            chosen_csfs_idxs = np.concatenate(
                [np.asarray(selected_csfs_idxs, dtype=np.int64), random_idxs]
            )
        else:
            chosen_csfs_idxs = np.array(selected_csfs_idxs, dtype=np.int64)
    else:
        chosen_csfs_idxs = np.random.choice(
            all_idxs, size=total_needed, replace=False
        )

    unselected_idxs = np.setdiff1d(all_idxs, chosen_csfs_idxs)

    chosen_csfs: list[list[str]] = [block_csfs_list[idx] for idx in chosen_csfs_idxs]

    return chosen_csfs, chosen_csfs_idxs, unselected_idxs


def select_csfs_rows(
    csfs_df: pl.DataFrame,
    idxs: np.ndarray,
    *,
    allow_duplicates: bool = False,
) -> pl.DataFrame:
    """Select CSF rows from a DataFrame under the strict index contract.

    Args:
        csfs_df: DataFrame with one row per CSF.
        idxs: Row indices to select.
        allow_duplicates: If False (default), duplicate indices are rejected.

    Returns:
        The selected rows, in ``idxs`` order.

    Raises:
        ValueError: If ``idxs`` fails :func:`validation.validate_selection_idxs`.
    """
    valid_idxs = validate_selection_idxs(
        idxs,
        row_count=csfs_df.height,
        allow_duplicates=allow_duplicates,
    )
    return csfs_df[valid_idxs]


def _unique_preserve_order(idxs: NDArray[np.int64]) -> NDArray[np.int64]:
    seen: set[int] = set()
    ordered: list[int] = []
    for idx in idxs:
        idx_int = int(idx)
        if idx_int in seen:
            continue
        seen.add(idx_int)
        ordered.append(idx_int)
    return np.array(ordered, dtype=np.int64)


def rmix_cumulative_selected_row_idxs(
    rmix_ci_squared: RmixCiSquaredData,
    cumulative_threshold: float,
) -> NDArray[np.int64]:
    """Select CSF row indices whose cumulative CI-square contribution is kept.

    Selected per-ASF CSF indices are unioned across all selected ASFs and all
    blocks, then offset by each block's CSF count so the result indexes
    directly into a concatenated (all-blocks) CSF row space. Order is
    preserved from the descending CI-square ranking (most important CSF
    first), not sorted by row index.

    Args:
        rmix_ci_squared: Parsed rmix CI-square data, one 2D array per block
            (ASF rows by CSF columns).
        cumulative_threshold: Cumulative CI-square fraction to keep, in
            ``(0, 1]``.

    Returns:
        Duplicate-free row indices into the concatenated CSF space, ordered
        by descending CI-square importance.

    Raises:
        ValueError: If no block/ASF selects any CSF.
    """
    selected_by_block = rmix_ci_squared.filter_sorted_ci_scores_by_cumulative(
        cumulative_threshold
    )

    mapped_by_block: list[NDArray[np.int64]] = []
    block_offset = 0
    for block_ci_indices, block_ci_squared in zip(
        selected_by_block.csf_indices_list,
        rmix_ci_squared.ci_squared_list,
        strict=True,
    ):
        block_selected = [
            np.asarray(asf_ci_indices, dtype=np.int64)
            for asf_ci_indices in block_ci_indices
            if len(asf_ci_indices) > 0
        ]
        if block_selected:
            block_union = _unique_preserve_order(np.concatenate(block_selected))
            mapped_by_block.append(block_union + block_offset)

        block_offset += int(block_ci_squared.shape[1])

    if not mapped_by_block:
        raise ValueError("rmix 累计贡献筛选没有选出任何 CSF")
    return _unique_preserve_order(np.concatenate(mapped_by_block))
