# -*- encoding: utf-8 -*-
from typing import Literal

import numpy as np
from numpy.typing import NDArray

AggregationMethod = Literal["sum", "max", "mean"]


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
