# -*- encoding: utf-8 -*-
"""Deterministic and random CSF selection primitives."""

from __future__ import annotations

import math
import random

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
from .validation import normalize_asf_row_indices, validate_selection_idxs


def select_csf_indices_above_ci_squared_cutoff(
    coefficients: np.ndarray,
    *,
    ci_squared_cutoff: float = 0.1,
) -> NDArray[np.int64]:
    """Return CSF indices above a CI-square cutoff, highest score first."""
    coefficient_array = np.asarray(coefficients, dtype=np.float64)
    if coefficient_array.ndim != 1:
        raise ValueError("coefficients must be a one-dimensional array")

    squared_scores = ci_squared(coefficient_array)
    indices = filter_ci_scores_by_threshold(
        squared_scores,
        threshold=ci_squared_cutoff,
    )
    if indices.size == 0:
        return np.array([], dtype=np.int64)
    return indices[np.argsort(-squared_scores[indices])].astype(np.int64, copy=False)


def select_csf_indices_by_ci_squared_cutoff(
    asfs_mix_data: MixCoefficientData,
    *,
    asf_row_indices: list[list[int]] | None = None,
    ci_squared_cutoff: float = 0.1,
) -> dict[int, NDArray[np.int64]]:
    """Select the union of CSF columns above a CI-square cutoff per block."""
    result: dict[int, NDArray[np.int64]] = {}

    normalized_rows = normalize_asf_row_indices(asfs_mix_data, asf_row_indices)

    for block, selected_rows in zip(
        asfs_mix_data.blocks, normalized_rows, strict=True
    ):
        block_data = block.mix_coefficients[selected_rows]

        squared_scores = ci_squared(block_data)
        score_matrix = (
            squared_scores
            if squared_scores.ndim == 2
            else np.atleast_2d(squared_scores)
        )

        above_cutoff_mask = np.any(score_matrix > ci_squared_cutoff, axis=0)
        result[block.block_index] = np.flatnonzero(above_cutoff_mask).astype(
            np.int64, copy=False
        )

    return result


def sort_csfs_by_mix_coefficient(
    csfs_block: list[list[str]],
    mix_coefficients: np.ndarray,
    ci_coefficient_cutoff: float | None = None,
) -> list[list[str]]:
    """Sort CSFs by the summed square of their mixing coefficients.

    Args:
        csfs_block: CSF records in one block.
        mix_coefficients: 1D (single ASF) or 2D (multiple ASFs) coefficient
            array whose last axis matches ``csfs_block``.
        ci_coefficient_cutoff: Optional CI coefficient cutoff. When provided,
            only CSFs whose combined CI-square is above its square are kept.

    Returns:
        CSF records sorted by descending combined squared coefficient.

    Raises:
        ValueError: If the CSF block or coefficient array is empty, or if their
            lengths do not match.
    """
    if len(csfs_block) == 0 or len(mix_coefficients) == 0:
        raise ValueError("CSFs_block和mix_coefficients不能为空")

    coeff_array = np.atleast_2d(np.asarray(mix_coefficients))
    if coeff_array.shape[-1] != len(csfs_block):
        raise ValueError("mix_coefficients长度必须与CSFs_block匹配")

    squared_coefficients = ci_squared(coeff_array)
    combined_coeff: NDArray[np.float64] = np.asarray(
        np.sum(squared_coefficients, axis=0, dtype=np.float64),
        dtype=np.float64,
    )
    sorted_idxs, _ = sort_ci_scores(combined_coeff)

    if ci_coefficient_cutoff is not None:
        threshold_idxs = filter_ci_scores_by_threshold(
            combined_coeff,
            ci_coefficient_cutoff**2,
        )
        sorted_idxs = sorted_idxs[np.isin(sorted_idxs, threshold_idxs)]

    return [csfs_block[int(idx)] for idx in sorted_idxs]


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


def random_choose_csfs(
    block_csfs_list: list[list[str]],
    *,
    ratio: float | None = None,
    target_count: int | None = None,
    selected_csf_indices: list[int] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[list[str]], NDArray[np.int64], NDArray[np.int64]]:
    """Randomly choose additional CSFs from a block.

    Args:
        block_csfs_list: Candidate CSFs in one block.
        ratio: Fraction of the block to select, in ``[0, 1]``.
        target_count: Absolute number of CSFs to select.
        selected_csf_indices: Indices that have already been selected.
        rng: Optional NumPy generator for reproducible sampling.

    Returns:
        Tuple of selected CSF records, selected indices, and unselected indices.

    Raises:
        ValueError: If exactly one target is not supplied or an input is invalid.
    """
    if (ratio is None) == (target_count is None):
        raise ValueError("exactly one of ratio or target_count must be provided")
    generator = rng or np.random.default_rng()

    block_csfs_num = len(block_csfs_list)
    if ratio is not None:
        if not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("ratio must be finite and within [0, 1]")
        total_needed = math.ceil(block_csfs_num * ratio)
    else:
        if type(target_count) is not int or target_count < 0:
            raise ValueError("target_count must be a non-negative integer")
        total_needed = target_count
    if total_needed > block_csfs_num:
        raise ValueError("selection target cannot exceed the CSF block size")

    selected_indices = validate_selection_idxs(
        np.asarray(selected_csf_indices or [], dtype=np.int64),
        row_count=block_csfs_num,
    )
    selected_csfs_num = int(selected_indices.size)
    if selected_csfs_num > total_needed:
        raise ValueError("preselected CSF count exceeds the selection target")

    choose_csfs_num = max(0, total_needed - selected_csfs_num)

    all_idxs = np.arange(block_csfs_num, dtype=np.int64)

    if selected_csfs_num > 0:
        selected_set = set(selected_indices.tolist())
        unselected_mask = ~np.isin(all_idxs, list(selected_set))
        unselected_idxs = all_idxs[unselected_mask]

        if choose_csfs_num > 0:
            random_idxs = generator.choice(
                unselected_idxs, size=choose_csfs_num, replace=False
            )
            chosen_csfs_idxs = np.concatenate(
                [selected_indices, random_idxs]
            )
        else:
            chosen_csfs_idxs = selected_indices
    else:
        chosen_csfs_idxs = generator.choice(
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
