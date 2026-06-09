# -*- encoding: utf-8 -*-

import logging
import random
import math
from typing import Literal, TypedDict
from collections import Counter

logger = logging.getLogger(__name__)

import numpy as np
from numpy.typing import NDArray

from ..grasp_data_extractor.rmix_data_processor import (
    aggregate_ci_squared,
    ci_squared,
    filter_ci_scores_by_threshold,
    sort_ci_scores,
)
from ..utils.tool_function import *
from ..utils.data_modules import MixCoefficientData

"""
    csfs data dictionary:
    {
        'CSFs_block_data': 
        list[
            blocks[
                    block_csfs[CSF_item[csf_1], CSF_item[csf_2], ...]]
                    ]
            ],
        'CSFs_block_j_value',
        'CSFs_block_length': list[length of each block],
        'parity',
        'subshell_info_raw'
    }

    rmix data dictionary:
    {
        'block_num': list[length of each block],
        'block_energy_count_list': list[levels of each block],
        'block_energy_list': list[energy of each block],
        'block_idx_list': list[idx of each block],
        'block_level_energy_list': list[
                                        block[level energy]
                                    ],
        'block_levels_idx_list': list[
                                        block[level idx]
                                    ],
        'j_value_location_list': list[location of j value],
        'mix_coefficient_list': list[
                                    block numpy.ndarray[
                                                        level numpy.ndarray[mix coefficient]
                                    ]
                                ],
        'parity_list': list[parity of each block],
    }
"""


#######################################################################
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
    # 检查输入是否为一维数组
    if asf_mix_data_array.ndim != 1:
        raise ValueError("输入数组必须是一维的")

    squared_scores = ci_squared(asf_mix_data_array)
    idxs = filter_ci_scores_by_threshold(squared_scores, threshold)

    # 如果没有满足条件的元素，返回空列表
    if len(idxs) == 0:
        return []

    # 获取对应的值
    values = np.asarray(asf_mix_data_array)[idxs]

    # 按绝对值降序排序索引
    sorted_idxs = idxs[np.argsort(-np.abs(values))]

    # 返回索引元组列表
    return [(int(idx),) for idx in sorted_idxs]


# 测试使用新的流程
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

    for block_offset, (block, selected_positions) in enumerate(
        zip(asfs_mix_data.blocks, normalized_positions)
    ):
        allowed_positions = set(block.level_indices.astype(np.int64).tolist())
        selected_set = {int(pos) for pos in selected_positions}
        if not selected_set.issubset(allowed_positions):
            raise ValueError(
                f"asfs_position 第 {block_offset} 层元素 {sorted(selected_set)} "
                f"不是 block.level_indices 对应层 {sorted(allowed_positions)} 的子集。"
            )

        block_data = block.mix_coefficients[selected_positions]

        squared_scores = ci_squared(block_data)
        score_matrix = (
            squared_scores
            if squared_scores.ndim == 2
            else np.atleast_2d(squared_scores)
        )

        # 按列求逻辑或：只要任意层级超过阈值，就保留该系数索引
        above_threshold_mask = np.any(score_matrix > threshold, axis=0)

        # 获取超过阈值的系数索引
        result[block.block_index] = np.where(above_threshold_mask)[0]

    return result


#######################################################################


def CSFs_block_get_CSF(CSFs_block: list[list[str]], CSf_idx: list[int] | np.ndarray) -> list[list[str]]:
    """Select CSF records from a block by index.

    Args:
        CSFs_block: CSF records in one block.
        CSf_idx: Indices of the CSF records to select.

    Returns:
        Selected CSF records in the same order as ``CSf_idx``.
    """
    selected_data: list[list[str]] = [CSFs_block[i] for i in CSf_idx]

    return selected_data


#######################################################################

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


def single_block_csfs_final_coupling_J_collector(
    block_csfs: list[list[str]], coupling_level: int | None = None
) -> dict[tuple[str, ...], CouplingJInfo]:
    """Collect final coupling-J patterns from one CSF block.

    Args:
        block_csfs: CSF records whose third line contains whitespace-separated
            coupling tokens.
        coupling_level: Number of trailing coupling tokens to keep. If None,
            all tokens are used. Shorter CSF records fall back to all tokens.

    Returns:
        Mapping from coupling-token pattern to its occurrence count and CSF
        indices.
    """
    all_tokens: list[tuple[str, ...]] = [
        tuple(csf[2].lstrip().split()) for csf in block_csfs
    ]

    if coupling_level is None:
        selected_tokens = all_tokens
    else:
        selected_tokens = [
            tokens[-coupling_level:] if len(tokens) >= coupling_level else tokens
            for tokens in all_tokens
        ]

    coupling_J_counts = Counter(selected_tokens)
    coupling_J_collection: dict[tuple[str, ...], CouplingJInfo] = {
        pattern: {"count": cnt, "idxs": []}
        for pattern, cnt in coupling_J_counts.items()
    }
    for idx, pattern in enumerate(selected_tokens):
        coupling_J_collection[pattern]["idxs"].append(idx)

    return coupling_J_collection


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
        logger.info("Block {block + 1}: 包含 {len(block_csfs)} 个 CSF")
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
        logger.debug(f"{pattern=}  count={info["count"]}  sum_ci={sum_ci}")
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
    # 修复可变默认参数
    if block_asfs_position is None:
        block_asfs_position = list(range(len(block_asfs_mix_coefficient_list)))

    # 用 set 将 O(n) 查找降为 O(1)
    position_set: set[int] = {int(i) for i in block_asfs_position}

    # 将系数统一转成 numpy 数组以便向量化
    coeff_matrix = np.asarray(block_asfs_mix_coefficient_list)

    base_coupling_dict = single_block_csfs_final_coupling_J_collector(
        block_CSFs, coupling_level
    )
    result: dict[tuple[str, ...], CouplingJInfoWithSumCiList] = {}
    for pattern, info in base_coupling_dict.items():
        idxs = np.asarray(info["idxs"])
        sum_ci_list: list[float] = []
        for asf_idx in block_asfs_position:
            if asf_idx in position_set:
                asf_coeff = coeff_matrix[int(asf_idx)]
                sum_ci_list.append(float(np.sum(asf_coeff[idxs] ** 2)))
        result[pattern] = {"count": info["count"], "idxs": info["idxs"], "sum_ci": sum_ci_list}

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
            [int(pos) for pos in block_positions]
            for block_positions in asfs_position
        ]

    if len(normalized_positions) != len(asfs_mix_data.blocks):
        raise ValueError(
            f"asfs_position 第一层长度 {len(normalized_positions)} "
            f"与 blocks {len(asfs_mix_data.blocks)} 不一致。"
        )

    if len(blocks_CSFs_list) != len(asfs_mix_data.blocks):
        raise ValueError(
            f"blocks_CSFs_list 长度 {len(blocks_CSFs_list)} "
            f"与 mix blocks {len(asfs_mix_data.blocks)} 不一致。"
        )

    blocks_asfs_coupling_J_sum_ci: dict[int, dict[tuple[str, ...], CouplingJInfoWithSumCiList]] = {}
    for block_offset, (block_csfs, mix_block, selected_positions) in enumerate(
        zip(blocks_CSFs_list, asfs_mix_data.blocks, normalized_positions)
    ):
        allowed_positions = set(mix_block.level_indices.astype(np.int64).tolist())
        selected_set = {int(pos) for pos in selected_positions}
        if not selected_set.issubset(allowed_positions):
            raise ValueError(
                f"asfs_position 第 {block_offset} 层元素 {sorted(selected_set)} "
                f"不是 block.level_indices 对应层 {sorted(allowed_positions)} 的子集。"
            )

        logger.info(f"Block {mix_block.block_index + 1}: 包含 {len(mix_block.mix_coefficients)} 个 ASF")
        if any(len(asf_mix) != len(block_csfs) for asf_mix in mix_block.mix_coefficients):
            raise ValueError(
                f"Block {mix_block.block_index}: block_CSFs 长度 {len(block_csfs)} 与 "
                f"block_asfs_mix_coefficient 长度不匹配。"
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


#######################################################################


def union_lists_with_order(*lists: list[int | str]) -> list[int | str]:
    """Return the ordered union of multiple lists.

    Args:
        *lists: Lists whose elements should be merged.

    Returns:
        De-duplicated list that preserves the first occurrence order.
    """
    # 使用 dict.fromkeys 保留元素顺序并去重
    all_elements: list[int | str] = []
    for lst in lists:
        all_elements.extend(lst)
    return list(dict.fromkeys(all_elements))


#######################################################################


def CSFs_sort_by_mix_coefficient(
    CSFs_block: list[list[str]],
    mix_coefficients: np.ndarray,
    threshold: float | None = None
) -> list[list[str]]:
    """Sort CSFs by the summed square of their mixing coefficients.

    Args:
        CSFs_block: CSF records in one block.
        mix_coefficients: Coefficient array whose length matches
            ``CSFs_block``.
        threshold: Optional coefficient cutoff. When provided, only CSFs whose
            combined squared coefficient is above ``threshold ** 2`` are kept.

    Returns:
        CSF records sorted by descending combined squared coefficient.

    Raises:
        ValueError: If the CSF block or coefficient array is empty, or if their
            lengths do not match.
    """
    # 检查输入参数的有效性
    if len(CSFs_block) == 0 or len(mix_coefficients) == 0:
        raise ValueError("CSFs_block和mix_coefficients不能为空")

    # 检查所有系数数组长度一致
    coeff_lengths = {len(coeff) for coeff in mix_coefficients}
    if len(coeff_lengths) > 1:
        raise ValueError("所有mix_coefficients数组长度必须相同")
    if len(CSFs_block) != next(iter(coeff_lengths)):
        raise ValueError("mix_coefficients长度必须与CSFs_block匹配")

    # TODO: 该旧排序逻辑仍依赖跨 ASF 聚合；后续应改为使用
    # RmixCiSquaredData/load_rmix_ci_squared 风格的新接口提供 CSF 索引。
    combined_coeff = aggregate_ci_squared(ci_squared(np.asarray(mix_coefficients)))
    sorted_idxs, _ = sort_ci_scores(combined_coeff)

    if threshold is not None:
        threshold_idxs = filter_ci_scores_by_threshold(combined_coeff, threshold**2)
        sorted_idxs = sorted_idxs[np.isin(sorted_idxs, threshold_idxs)]

    return [CSFs_block[int(idx)] for idx in sorted_idxs]


#######################################################################
# random select csfs from block_csfs_list
#######################################################################


def generate_unique_random_numbers(max_num: int, count: int) -> list[int]:
    """Generate unique random positive integers.

    Args:
        max_num: Inclusive upper bound for generated numbers.
        count: Number of unique values to generate.

    Returns:
        List of unique sampled integers in the range ``[1, max_num]``.
    """
    number: list[int]= random.sample(range(1, max_num + 1), count)
    return number


def radom_choose_csfs(
    block_csfs_list: list[list[str]],
    method: Literal["ratio", "quality"],
    ratio_or_quality: float,
    selected_csfs_idxs: list[list[str]] = [],
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
    """
    block_csfs_num = len(block_csfs_list)
    selected_csfs_num = len(selected_csfs_idxs)
    if method == "ratio":
        # 计算需要选择的总数
        total_needed = math.ceil(block_csfs_num * ratio_or_quality)
    elif method == "quality":
        # 计算需要选择的总数
        total_needed = math.ceil(ratio_or_quality)
    # 计算还需要补充的数量
    choose_csfs_num = max(0, total_needed - selected_csfs_num)

    # 使用numpy数组加速操作
    all_idxs = np.arange(block_csfs_num, dtype=np.int64)

    if selected_csfs_num > 0:
        # 使用numpy的set操作
        selected_set = set(selected_csfs_idxs)
        unselected_mask = ~np.isin(all_idxs, list(selected_set))
        unselected_idxs = all_idxs[unselected_mask]

        if choose_csfs_num > 0:
            # 使用numpy的随机选择
            random_idxs = np.random.choice(
                unselected_idxs, size=choose_csfs_num, replace=False
            )
            chosen_csfs_idxs = np.concatenate(
                [selected_csfs_idxs, random_idxs]
            )
        else:
            chosen_csfs_idxs = np.array(selected_csfs_idxs)
    else:
        # 直接随机选择
        chosen_csfs_idxs = np.random.choice(
            all_idxs, size=total_needed, replace=False
        )

    # 获取未选择的索引
    unselected_idxs = np.setdiff1d(all_idxs, chosen_csfs_idxs)

    # 使用列表推导式获取CSF数据
    chosen_csfs:list[list[str]] = [block_csfs_list[idx] for idx in chosen_csfs_idxs]

    return chosen_csfs, chosen_csfs_idxs, unselected_idxs
