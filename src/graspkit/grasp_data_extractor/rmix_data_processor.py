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
class RmixCsfIndexSelection:
    """保存按 block 和 ASF 分组的 CSF 索引选择结果。

    Attributes
    ----------
    block_indices : list[int]
        rmix 文件中的 block 索引列表。
    selected_asfs : list[list[int]]
        每个 block 中参与处理的 ASF 索引列表。
    csf_indices_list : list[list[list[int]]]
        每个 block、每个 ASF 对应的 CSF 索引列表。
    """

    block_indices: list[int]
    selected_asfs: list[list[int]]
    csf_indices_list: list[list[list[int]]]


@dataclass(frozen=True)
class RmixCiSquaredData:
    """保存按 block 和 ASF 分组的 CI 系数平方数据。

    Attributes
    ----------
    block_indices : list[int]
        rmix 文件中的 block 索引列表。
    selected_asfs : list[list[int]]
        每个 block 中参与处理的 ASF 索引列表。
    ci_squared_list : list[NDArray[np.float64]]
        每个 block 对应的 CI 系数平方矩阵列表，形状为
        ``(n_selected_asf, n_csf)``。
    """

    block_indices: list[int]
    selected_asfs: list[list[int]]
    ci_squared_list: list[NDArray[np.float64]]

    def sort_ci_scores(self, descending: bool = True) -> RmixCsfIndexSelection:
        """返回每个 ASF 中按 CI 系数平方排序后的 CSF 索引。

        Parameters
        ----------
        descending : bool
            是否按 CI 系数平方降序排序。

        Returns
        -------
        RmixCsfIndexSelection
            按 block 和 ASF 分组的排序后 CSF 索引。
        """
        return RmixCsfIndexSelection(
            block_indices=self.block_indices,
            selected_asfs=self.selected_asfs,
            csf_indices_list=[
                [
                    sort_ci_scores(asf_scores, descending=descending)[0].tolist()
                    for asf_scores in block_ci_squared
                ]
                for block_ci_squared in self.ci_squared_list
            ],
        )

    def filter_ci_scores_by_threshold(
        self,
        threshold: float,
        inclusive: bool = False,
    ) -> RmixCsfIndexSelection:
        """返回每个 ASF 中通过 CI 系数平方阈值筛选的 CSF 索引。

        Parameters
        ----------
        threshold : float
            CI 系数平方的筛选阈值。
        inclusive : bool
            是否包含等于阈值的 CSF。

        Returns
        -------
        RmixCsfIndexSelection
            按 block 和 ASF 分组的筛选后 CSF 索引。
        """
        return RmixCsfIndexSelection(
            block_indices=self.block_indices,
            selected_asfs=self.selected_asfs,
            csf_indices_list=[
                [
                    filter_ci_scores_by_threshold(
                        asf_scores,
                        threshold=threshold,
                        inclusive=inclusive,
                    ).tolist()
                    for asf_scores in block_ci_squared
                ]
                for block_ci_squared in self.ci_squared_list
            ],
        )

    def filter_sorted_ci_scores_by_cumulative(
        self,
        cumulative_threshold: float,
    ) -> RmixCsfIndexSelection:
        """返回每个 ASF 中达到累计贡献阈值的排序后 CSF 索引。

        Parameters
        ----------
        cumulative_threshold : float
            归一化累计贡献阈值，取值范围为 ``(0, 1]``。

        Returns
        -------
        RmixCsfIndexSelection
            按 block 和 ASF 分组的累计截断后 CSF 索引。
        """
        block_csf_indices: list[list[list[int]]] = []
        for block_ci_squared in self.ci_squared_list:
            block_indices: list[list[int]] = []
            for asf_scores in block_ci_squared:
                sorted_indices, sorted_scores = sort_ci_scores(asf_scores)
                cumulative_positions = filter_sorted_ci_scores_by_cumulative(
                    sorted_scores,
                    cumulative_threshold=cumulative_threshold,
                )
                block_indices.append(sorted_indices[cumulative_positions].tolist())
            block_csf_indices.append(block_indices)

        return RmixCsfIndexSelection(
            block_indices=self.block_indices,
            selected_asfs=self.selected_asfs,
            csf_indices_list=block_csf_indices,
        )


def _as_1d_or_2d_float_array(
    values: NDArray[np.float64],
    name: str,
) -> NDArray[np.float64]:
    """将输入转换为非空的一维或二维 float64 数组。

    Parameters
    ----------
    values : NDArray[np.float64]
        待转换和校验的输入数组。
    name : str
        用于错误信息的输入名称。

    Returns
    -------
    NDArray[np.float64]
        转换后的 float64 数组。
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if array.ndim not in (1, 2):
        raise ValueError(f"{name} must be a 1D or 2D array")
    return array


def ci_squared(coefficients: NDArray[np.float64]) -> NDArray[np.float64]:
    """返回 CI 系数平方。

    Parameters
    ----------
    coefficients : NDArray[np.float64]
        一维或二维 CI 系数数组。

    Returns
    -------
    NDArray[np.float64]
        与输入形状一致的 CI 系数平方数组，dtype 为 float64。
    """
    coefficient_array = _as_1d_or_2d_float_array(coefficients, "coefficients")
    return np.square(coefficient_array, dtype=np.float64)


def aggregate_ci_squared(
    ci_squared_values: NDArray[np.float64],
    method: AggregationMethod = "sum",
) -> NDArray[np.float64]:
    """按 CSF 聚合多个 ASF 的 CI 系数平方。

    Parameters
    ----------
    ci_squared_values : NDArray[np.float64]
        一维 ``(n_csf,)`` 或二维 ``(n_asf, n_csf)`` 的 CI 系数平方数组。
    method : AggregationMethod
        二维输入的聚合方式，可选 ``"sum"``、``"max"`` 或 ``"mean"``。

    Returns
    -------
    NDArray[np.float64]
        每个 CSF 对应的聚合后 CI 系数平方。
    """
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
    """将输入转换为非空的一维 float64 分数数组。

    Parameters
    ----------
    values : NDArray[np.float64]
        待转换和校验的分数数组。
    name : str
        用于错误信息的输入名称。

    Returns
    -------
    NDArray[np.float64]
        转换后的一维 float64 分数数组。
    """
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
    """按 CI 系数平方排序 CSF。

    Parameters
    ----------
    scores : NDArray[np.float64]
        一维 CI 系数平方分数数组。
    descending : bool
        是否按降序排序。

    Returns
    -------
    tuple[NDArray[np.int64], NDArray[np.float64]]
        原始 CSF 索引的排序结果，以及对应的排序后 CI 系数平方。
    """
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
    """按 CI 系数平方阈值筛选 CSF 索引。

    Parameters
    ----------
    scores : NDArray[np.float64]
        一维 CI 系数平方分数数组。
    threshold : float
        CI 系数平方筛选阈值。
    inclusive : bool
        是否包含等于阈值的 CSF。

    Returns
    -------
    NDArray[np.int64]
        通过阈值筛选的原始 CSF 索引。
    """
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
    """按归一化累计贡献截断已排序的 CI 系数平方。

    Parameters
    ----------
    sorted_scores : NDArray[np.float64]
        已排序的一维 CI 系数平方分数数组。
    cumulative_threshold : float
        归一化累计贡献阈值，取值范围为 ``(0, 1]``。

    Returns
    -------
    NDArray[np.int64]
        在已排序数组中的保留位置索引。
    """
    if not 0 < cumulative_threshold <= 1:
        raise ValueError("cumulative_threshold must be > 0 and <= 1")
    score_array = _as_1d_score_array(sorted_scores, "sorted_scores")
    total = float(np.sum(score_array))
    if total <= 0:
        return np.array([], dtype=np.int64)
    cumulative = np.cumsum(score_array) / total
    count = int(np.searchsorted(cumulative, cumulative_threshold, side="left")) + 1
    return np.arange(count, dtype=np.int64)


def _normalize_asf_indices(
    coefficients: NDArray[np.float64],
    select_asfs: Sequence[int] | NDArray[np.integer] | None,
) -> NDArray[np.int64]:
    """规范化一个 block 中需要处理的 ASF 索引。

    Parameters
    ----------
    coefficients : NDArray[np.float64]
        一个 block 的 CI 系数数组，形状为 ``(n_csf,)`` 或
        ``(n_asf, n_csf)``。
    select_asfs : Sequence[int] | NDArray[np.integer] | None
        用户指定的 ASF 索引；为空时表示选择该 block 中全部 ASF。

    Returns
    -------
    NDArray[np.int64]
        规范化后的一维 ASF 索引数组。
    """
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


def load_rmix_ci_squared(
    rmix_path: str | Path,
    select_asfs: list[list[int]] | None = None,
) -> RmixCiSquaredData:
    """读取 rmix 文件并返回选中 ASF 的 CI 系数平方。

    Parameters
    ----------
    rmix_path : str | Path
        需要读取的 rmix 文件路径。
    select_asfs : list[list[int]] | None
        每个 block 中需要处理的 ASF 索引列表；为空或 None 时表示每个
        block 中全部 ASF 都参与处理。

    Returns
    -------
    RmixCiSquaredData
        按 block 分组的选中 ASF 的 CI 系数平方数据。
    """
    mix_data = MixCoefLoader(Path(rmix_path)).load()
    if select_asfs is not None and len(select_asfs) not in (0, mix_data.block_num):
        raise ValueError("select_asfs length must match the number of rmix blocks")

    selected_asfs: list[list[int]] = []
    ci_squared_list: list[NDArray[np.float64]] = []
    for block_index, block_coefficients in enumerate(mix_data.mix_coefficient_list):
        coefficient_array = _as_1d_or_2d_float_array(block_coefficients, "coefficients")
        block_select_asfs = (
            select_asfs[block_index]
            if select_asfs is not None and len(select_asfs) > 0
            else None
        )
        selected_asf_indices = _normalize_asf_indices(
            coefficient_array,
            block_select_asfs,
        )
        selected_coefficients = (
            coefficient_array[np.newaxis, :]
            if coefficient_array.ndim == 1
            else coefficient_array[selected_asf_indices]
        )
        selected_asfs.append(selected_asf_indices.tolist())
        ci_squared_list.append(ci_squared(selected_coefficients))

    return RmixCiSquaredData(
        block_indices=list(mix_data.block_idx_list),
        selected_asfs=selected_asfs,
        ci_squared_list=ci_squared_list,
    )
