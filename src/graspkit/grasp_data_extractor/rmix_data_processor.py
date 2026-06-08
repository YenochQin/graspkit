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
