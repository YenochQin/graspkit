# -*- encoding: utf-8 -*-
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..data_IO.loaders.mix_coef_loader import MixCoefLoader

AggregationMethod = Literal["sum", "max", "mean"]


@dataclass(frozen=True)
class RmixAsfSelection:
    """CI-square selection result for one ASF in an rmix symmetry block."""

    asf_index: int
    selected_csf_indices: NDArray[np.int64]
    scores: NDArray[np.float64]
    selected_scores: NDArray[np.float64]
    cumulative_scores: NDArray[np.float64]
    selected_cumulative_scores: NDArray[np.float64]


@dataclass(frozen=True)
class RmixBlockSelection:
    """CI-square selection result for selected ASFs in one rmix block."""

    block_index: int
    selected_asf_indices: NDArray[np.int64]
    asf_selections: list[RmixAsfSelection]
    ci_squared: NDArray[np.float64]


def _as_1d_or_2d_float_array(
    values: NDArray[np.float64],
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if array.ndim not in (1, 2):
        raise ValueError(f"{name} must be a 1D or 2D array")
    return array


def ci_squared(coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return squared CI coefficients as a float64 array."""
    coefficient_array = _as_1d_or_2d_float_array(coefficients, "coefficients")
    return np.square(coefficient_array, dtype=np.float64)


def aggregate_ci_squared(
    ci_squared_values: NDArray[np.float64],
    method: AggregationMethod = "sum",
) -> NDArray[np.float64]:
    """Aggregate CI-square values per CSF across ASFs."""
    ci_array = _as_1d_or_2d_float_array(ci_squared_values, "ci_squared_values")
    if ci_array.ndim == 1:
        return ci_array.astype(np.float64, copy=False)
    if method == "sum":
        return np.sum(ci_array, axis=0, dtype=np.float64)
    if method == "max":
        return np.max(ci_array, axis=0)
    if method == "mean":
        return np.mean(ci_array, axis=0, dtype=np.float64)
    raise ValueError(f"Unsupported aggregation method: {method}")


def _as_1d_score_array(
    values: NDArray[np.float64],
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return array


def sort_ci_scores(
    scores: NDArray[np.float64],
    descending: bool = True,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Sort per-CSF CI-square scores and return original indexes."""
    score_array = _as_1d_score_array(scores, "scores")
    order = np.argsort(score_array)
    if descending:
        order = order[::-1]
    sorted_indices = order.astype(np.int64, copy=False)
    return sorted_indices, score_array[sorted_indices]


def filter_ci_scores_by_threshold(
    scores: NDArray[np.float64],
    threshold: float,
    inclusive: bool = False,
) -> NDArray[np.int64]:
    """Return original indexes whose CI-square scores pass a direct cutoff."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    score_array = _as_1d_score_array(scores, "scores")
    if inclusive:
        return np.where(score_array >= threshold)[0].astype(np.int64, copy=False)
    return np.where(score_array > threshold)[0].astype(np.int64, copy=False)


def filter_sorted_ci_scores_by_cumulative(
    sorted_scores: NDArray[np.float64],
    cumulative_threshold: float,
) -> NDArray[np.int64]:
    """Return sorted-array positions needed to reach a cumulative contribution."""
    if not 0 < cumulative_threshold <= 1:
        raise ValueError("cumulative_threshold must be > 0 and <= 1")
    score_array = _as_1d_score_array(sorted_scores, "sorted_scores")
    total = float(np.sum(score_array))
    if total <= 0:
        return np.array([], dtype=np.int64)
    cumulative = np.cumsum(score_array) / total
    count = int(np.searchsorted(cumulative, cumulative_threshold, side="left")) + 1
    return np.arange(count, dtype=np.int64)


def _validate_top_limits(top_k: int | None, top_ratio: float | None) -> None:
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_ratio is not None and not 0 < top_ratio <= 1:
        raise ValueError("top_ratio must be > 0 and <= 1")


def _normalized_cumulative(sorted_scores: NDArray[np.float64]) -> NDArray[np.float64]:
    total = float(np.sum(sorted_scores))
    if total <= 0:
        return np.zeros_like(sorted_scores, dtype=np.float64)
    return np.cumsum(sorted_scores) / total


def _normalize_asf_indices(
    coefficients: NDArray[np.float64],
    select_asfs: Sequence[int] | NDArray[np.integer] | None,
) -> NDArray[np.int64]:
    if select_asfs is None or len(select_asfs) == 0:
        requested_asfs = np.array([], dtype=np.int64)
    else:
        requested_asfs = np.asarray(select_asfs, dtype=np.int64)

    if coefficients.ndim == 1:
        if requested_asfs.size != 0 and (
            requested_asfs.size != 1 or int(requested_asfs[0]) != 0
        ):
            raise ValueError("1D coefficient arrays only support ASF index 0")
        return np.array([0], dtype=np.int64)

    if requested_asfs.size == 0:
        return np.arange(coefficients.shape[0], dtype=np.int64)

    if requested_asfs.ndim != 1:
        raise ValueError("select_asfs must be a 1D sequence of ASF indices")
    if np.any(requested_asfs < 0) or np.any(requested_asfs >= coefficients.shape[0]):
        raise ValueError("select_asfs contains an ASF index outside this block")
    return requested_asfs


def _select_single_asf_ci_scores(
    scores: NDArray[np.float64],
    asf_index: int,
    score_threshold: float | None = None,
    cumulative_threshold: float | None = None,
    top_k: int | None = None,
    top_ratio: float | None = None,
) -> RmixAsfSelection:
    sorted_indices, sorted_scores = sort_ci_scores(scores)
    cumulative_scores = _normalized_cumulative(sorted_scores)

    keep_positions = np.arange(sorted_indices.size, dtype=np.int64)

    if score_threshold is not None:
        threshold_indices = filter_ci_scores_by_threshold(scores, score_threshold)
        threshold_mask = np.isin(sorted_indices, threshold_indices)
        keep_positions = keep_positions[threshold_mask[keep_positions]]

    if cumulative_threshold is not None:
        cumulative_positions = filter_sorted_ci_scores_by_cumulative(
            sorted_scores,
            cumulative_threshold,
        )
        keep_positions = np.intersect1d(
            keep_positions,
            cumulative_positions,
            assume_unique=True,
        )

    if top_k is not None:
        top_k_positions = np.arange(min(top_k, sorted_indices.size), dtype=np.int64)
        keep_positions = np.intersect1d(
            keep_positions,
            top_k_positions,
            assume_unique=True,
        )

    if top_ratio is not None:
        ratio_count = int(np.ceil(sorted_indices.size * top_ratio))
        top_ratio_positions = np.arange(ratio_count, dtype=np.int64)
        keep_positions = np.intersect1d(
            keep_positions,
            top_ratio_positions,
            assume_unique=True,
        )

    selected_indices = sorted_indices[keep_positions]
    selected_scores = sorted_scores[keep_positions]
    selected_cumulative_scores = cumulative_scores[keep_positions]

    return RmixAsfSelection(
        asf_index=asf_index,
        selected_csf_indices=selected_indices,
        scores=sorted_scores,
        selected_scores=selected_scores,
        cumulative_scores=cumulative_scores,
        selected_cumulative_scores=selected_cumulative_scores,
    )


def select_block_ci_scores(
    coefficients: NDArray[np.float64],
    block_index: int = 0,
    select_asfs: Sequence[int] | NDArray[np.integer] | None = None,
    score_threshold: float | None = None,
    cumulative_threshold: float | None = None,
    top_k: int | None = None,
    top_ratio: float | None = None,
) -> RmixBlockSelection:
    """Analyze and select CSFs for each selected ASF in one rmix block."""
    _validate_top_limits(top_k, top_ratio)

    coefficient_array = _as_1d_or_2d_float_array(coefficients, "coefficients")
    selected_asf_indices = _normalize_asf_indices(coefficient_array, select_asfs)
    if coefficient_array.ndim == 1:
        selected_coefficients = coefficient_array[np.newaxis, :]
    else:
        selected_coefficients = coefficient_array[selected_asf_indices]
    block_ci_squared = ci_squared(selected_coefficients)

    asf_selections = [
        _select_single_asf_ci_scores(
            scores=asf_scores,
            asf_index=int(asf_index),
            score_threshold=score_threshold,
            cumulative_threshold=cumulative_threshold,
            top_k=top_k,
            top_ratio=top_ratio,
        )
        for asf_index, asf_scores in zip(
            selected_asf_indices,
            block_ci_squared,
            strict=True,
        )
    ]

    return RmixBlockSelection(
        block_index=block_index,
        selected_asf_indices=selected_asf_indices,
        asf_selections=asf_selections,
        ci_squared=block_ci_squared,
    )


def analyze_rmix_file(
    rmix_path: str | Path,
    select_asfs: Sequence[Sequence[int] | NDArray[np.integer]] | None = None,
    score_threshold: float | None = None,
    cumulative_threshold: float | None = None,
    top_k: int | None = None,
    top_ratio: float | None = None,
) -> list[RmixBlockSelection]:
    """Load an rmix file and analyze CI-square scores for every block."""
    mix_data = MixCoefLoader(Path(rmix_path)).load()
    if select_asfs is not None and len(select_asfs) not in (0, mix_data.block_num):
        raise ValueError("select_asfs length must match the number of rmix blocks")
    return [
        select_block_ci_scores(
            coefficients=block_coefficients,
            block_index=block_index,
            select_asfs=(
                select_asfs[block_index]
                if select_asfs is not None and len(select_asfs) > 0
                else None
            ),
            score_threshold=score_threshold,
            cumulative_threshold=cumulative_threshold,
            top_k=top_k,
            top_ratio=top_ratio,
        )
        for block_index, block_coefficients in enumerate(mix_data.mix_coefficient_list)
    ]
