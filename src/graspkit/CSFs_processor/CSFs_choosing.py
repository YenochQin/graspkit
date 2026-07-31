import math
import logging
import random
import sys
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

logger = logging.getLogger(__name__)

import numpy as np
from numpy.typing import NDArray
import polars as pl
import rtoml

from ..grasp_data_extractor.rmix_data_processor import (
    ci_squared,
    filter_ci_scores_by_threshold,
    load_rmix_ci_squared,
    sort_ci_scores,
)
from ..utils.tool_function import *
from ..utils.data_modules import MixCoefficientData

# This module consumes the current data_IO containers:
# - CSF records are grouped as CSFs.CSFs_block_data:
#   list[block][csf][line], where each CSF is its original three text lines.
# - Mixing coefficients are provided as MixCoefficientData.blocks, with one
#   MixCoefficientBlock per symmetry block.


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

    squared_coefficients = ci_squared(np.asarray(mix_coefficients))
    score_matrix = (
        squared_coefficients
        if squared_coefficients.ndim == 2
        else np.atleast_2d(squared_coefficients)
    )
    combined_coeff: NDArray[np.float64] = np.asarray(
        np.sum(score_matrix, axis=0, dtype=np.float64),
        dtype=np.float64,
    )
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


CUSTOM_CONFIG_EXAMPLE = """
Custom multi-J config example:

cumulative_threshold = 0.9

[output]
csfs_file = "chosen.c"

[[csfs_units]]
j = "0"
idx = "j0_idxs.npy"
csfs = "j0.parquet"
rmix_file = "j0.cm"
select_asfs = [0]

[[csfs_units]]
j = "2"
idx = "j2_idxs.npy"
csfs = "j2.c"
rmix_file = "j2.cm"
select_asfs = [[0, 1]]
"""


@dataclass(frozen=True)
class CsfsSelectionUnit:
    """One independent CSF source and its selected row indexes."""

    idx_file: Path
    csfs_file: Path
    label: str | None = None
    rmix_file: Path | None = None
    select_asfs: list[list[int]] | None = None
    cumulative_threshold: float = 0.9


@dataclass(frozen=True)
class SelectedCsfsBlock:
    """Selected CSF rows plus their source header."""

    header_lines: list[str]
    csfs_df: pl.DataFrame
    source_path: Path
    idx_file: Path
    label: str | None = None


def convert_csfs(
    input_path: Path,
    output_path: Path,
    *,
    num_workers: int | None = None,
) -> dict[str, object]:
    """Convert a GRASP .c CSF file to parquet via optional rcsfs dependency."""

    from rcsfs import convert_csfs as _convert_csfs

    return cast(
        dict[str, object],
        _convert_csfs(
            input_path=input_path,
            output_path=output_path,
            num_workers=num_workers,
        ),
    )


def _header_path_for_parquet(parquet_path: Path) -> Path:
    return parquet_path.with_name(f"{parquet_path.stem}_header.toml")


def _resolve_relative_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_selection_idxs(idx_file: Path) -> NDArray[np.int64]:
    loaded = np.load(idx_file, allow_pickle=False)
    idxs = cast(NDArray[np.int64], loaded).astype(np.int64, copy=False)
    if idxs.ndim != 1:
        raise ValueError(f"idx 文件必须是一维数组: {idx_file}")
    return idxs


def _valid_row_idxs(row_count: int, idxs: NDArray[np.int64]) -> NDArray[np.int64]:
    return idxs[(idxs >= 0) & (idxs < row_count)]


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


def _rmix_selected_local_ci_idxs(unit: CsfsSelectionUnit) -> NDArray[np.int64]:
    if unit.rmix_file is None:
        raise ValueError("rmix_file 未设置")

    rmix_ci_squared = load_rmix_ci_squared(
        unit.rmix_file,
        select_asfs=unit.select_asfs,
    )
    selected_by_block = rmix_ci_squared.filter_sorted_ci_scores_by_cumulative(
        unit.cumulative_threshold
    )

    mapped_by_block: list[NDArray[np.int64]] = []
    block_offset = 0
    for block_ci_indices, block_ci_squared in zip(
        selected_by_block.csf_indices_list,
        rmix_ci_squared.ci_squared_list,
    ):
        block_selected = [
            np.asarray(asf_ci_indices, dtype=np.int64)
            for asf_ci_indices in block_ci_indices
            if len(asf_ci_indices) > 0
        ]
        if block_selected:
            mapped_by_block.append(
                _unique_preserve_order(np.concatenate(block_selected)) + block_offset
            )

        block_offset += int(block_ci_squared.shape[1])

    if not mapped_by_block:
        raise RuntimeError(f"rmix 累计贡献筛选没有选出 CSFs: {unit.rmix_file}")
    return _unique_preserve_order(np.concatenate(mapped_by_block))


def _load_unit_raw_row_idxs(unit: CsfsSelectionUnit) -> NDArray[np.int64]:
    raw_idx_map = _load_selection_idxs(unit.idx_file)
    if unit.rmix_file is None:
        return raw_idx_map

    local_ci_idxs = _rmix_selected_local_ci_idxs(unit)
    if np.any(local_ci_idxs < 0) or np.any(local_ci_idxs >= raw_idx_map.shape[0]):
        raise ValueError(
            f"rmix 选出的 ci_idx 超出 idx 映射范围: {unit.rmix_file}, {unit.idx_file}"
        )
    return raw_idx_map[local_ci_idxs]


def _ensure_parquet_from_csfs(
    csfs_file: Path,
    *,
    num_workers: int | None,
) -> Path:
    if not csfs_file.is_file():
        raise FileNotFoundError(f"原始 CSFs 文件不存在: {csfs_file}")

    if csfs_file.suffix == ".parquet":
        return csfs_file

    if csfs_file.suffix != ".c":
        raise ValueError(f"-csfs 只支持 .parquet 或 .c 文件: {csfs_file}")

    parquet_path = csfs_file.with_suffix(".parquet")
    convert_csfs(
        input_path=csfs_file,
        output_path=parquet_path,
        num_workers=num_workers,
    )
    return parquet_path


def _load_header_lines(header_path: Path) -> list[str]:
    if not header_path.is_file():
        raise FileNotFoundError(f"CSFs header TOML 文件不存在: {header_path}")

    header = rtoml.load(header_path)
    header_info = header.get("header_info")
    if not isinstance(header_info, dict):
        raise ValueError(f"header TOML 缺少 [header_info]: {header_path}")
    header_lines = header_info.get("header_lines")
    if not isinstance(header_lines, list) or not all(
        isinstance(line, str) for line in header_lines
    ):
        raise ValueError(f"header TOML 缺少 header_info.header_lines: {header_path}")
    if len(header_lines) != 5:
        raise ValueError(f"CSFs header 必须是 5 行: {header_path}")
    return list(header_lines)


def _select_unit_csfs(
    unit: CsfsSelectionUnit,
    *,
    convert_workers: int | None,
) -> SelectedCsfsBlock:
    if not unit.idx_file.is_file():
        raise FileNotFoundError(f"idx 文件不存在: {unit.idx_file}")

    parquet_path = _ensure_parquet_from_csfs(
        unit.csfs_file,
        num_workers=convert_workers,
    )
    header_lines = _load_header_lines(_header_path_for_parquet(parquet_path))
    raw_csfs_df = pl.read_parquet(parquet_path)
    idxs = _load_unit_raw_row_idxs(unit)
    valid_idxs = _valid_row_idxs(raw_csfs_df.height, idxs)
    if valid_idxs.shape[0] == 0:
        raise RuntimeError(f"没有可用于提取的 idx: {unit.idx_file}")

    return SelectedCsfsBlock(
        header_lines=header_lines,
        csfs_df=raw_csfs_df[valid_idxs],
        source_path=unit.csfs_file,
        idx_file=unit.idx_file,
        label=unit.label,
    )


def _write_csfs_blocks_to_cfile(
    header_lines: list[str],
    blocks: Sequence[pl.DataFrame],
    output_file: Path,
) -> None:
    if len(header_lines) != 5:
        raise ValueError("CSFs file header info error!")
    if not blocks:
        raise ValueError("没有可写入的 CSFs block")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        for line in header_lines:
            file.write(f"{line}\n")

        for block_idx, csfs_df in enumerate(blocks):
            for row in csfs_df.select(["line1", "line2", "line3"]).iter_rows():
                file.write("\n".join(cast(tuple[str, str, str], row)))
                file.write("\n")
            if block_idx != len(blocks) - 1:
                file.write(" *\n")


def extract_csfs_units(
    units: Sequence[CsfsSelectionUnit],
    output_file: Path,
    *,
    workers: int | None = None,
) -> list[SelectedCsfsBlock]:
    """Extract selected CSFs from one or more independent units."""

    if not units:
        raise ValueError("至少需要一个 CSFs 提取单元")

    if len(units) == 1 or workers is None or workers <= 1:
        selected_blocks = [
            _select_unit_csfs(unit, convert_workers=workers) for unit in units
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            selected_blocks = list(
                executor.map(
                    lambda unit: _select_unit_csfs(
                        unit,
                        convert_workers=workers,
                    ),
                    units,
                )
            )

    _write_csfs_blocks_to_cfile(
        selected_blocks[0].header_lines,
        [block.csfs_df for block in selected_blocks],
        output_file,
    )
    return selected_blocks


def _output_path_from_cli(
    output_file: str | None,
    *,
    default_dir: Path,
    default_name: str = "chosen.c",
) -> Path:
    if output_file is None:
        return (default_dir / default_name).resolve()
    path = Path(output_file)
    if path.suffix == "":
        path = path.with_suffix(".c")
    if not path.is_absolute():
        path = (default_dir / path).resolve()
    return path


def _value_from_unit(
    unit: dict[str, Any],
    keys: Sequence[str],
    *,
    config_path: Path,
) -> str:
    for key in keys:
        value = unit.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"{config_path} 的 csfs_units 单元缺少字段: {keys}")


def _j_sort_key(label: str | None, original_index: int) -> tuple[int, Fraction, int]:
    if label is None:
        return (1, Fraction(0), original_index)
    try:
        return (0, Fraction(label.strip()), original_index)
    except ValueError as err:
        raise ValueError(
            f"csfs_units[{original_index}] 的 j 值无法解析: {label}"
        ) from err


def _parse_select_asfs(
    value: object, config_path: Path, unit_index: int
) -> list[list[int]] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(
            f"{config_path} 的 csfs_units[{unit_index}].select_asfs 必须是列表"
        )
    if not value:
        return []
    if all(isinstance(item, int) for item in value):
        return [[int(item) for item in value]]
    if all(isinstance(item, list) for item in value):
        nested: list[list[int]] = []
        for block_idx, block_value in enumerate(value):
            if not all(isinstance(item, int) for item in block_value):
                raise ValueError(
                    f"{config_path} 的 csfs_units[{unit_index}].select_asfs[{block_idx}] 必须是整数列表"
                )
            nested.append([int(item) for item in block_value])
        return nested
    raise ValueError(
        f"{config_path} 的 csfs_units[{unit_index}].select_asfs 必须是一维或二维整数列表"
    )


def _parse_cumulative_threshold(raw_config: dict[str, Any]) -> float:
    threshold_value = raw_config.get(
        "cumulative_threshold",
        raw_config.get("cumulative_ratio", 0.9),
    )
    threshold = float(threshold_value)
    if not 0 < threshold <= 1:
        raise ValueError("cumulative_threshold 必须在 (0, 1] 范围内")
    return threshold


def _units_from_custom_config(
    raw_config: dict[str, Any], config_path: Path
) -> list[CsfsSelectionUnit]:
    raw_units = raw_config.get("csfs_units")
    if not isinstance(raw_units, list):
        raise ValueError(f"{config_path} 缺少 [[csfs_units]] 配置")

    cumulative_threshold = _parse_cumulative_threshold(raw_config)
    indexed_units: list[tuple[int, CsfsSelectionUnit]] = []
    for idx, raw_unit in enumerate(raw_units):
        if not isinstance(raw_unit, dict):
            raise ValueError(f"{config_path} 的 csfs_units[{idx}] 必须是表")
        idx_value = _value_from_unit(
            raw_unit,
            ("idx", "idxs", "idx_file", "idxs_file"),
            config_path=config_path,
        )
        csfs_value = _value_from_unit(
            raw_unit,
            ("csfs", "csfs_file", "raw_csfs", "raw_csfs_file"),
            config_path=config_path,
        )
        label = raw_unit.get("j", raw_unit.get("label"))
        rmix_value = raw_unit.get("rmix_file")
        unit = CsfsSelectionUnit(
            idx_file=_resolve_relative_path(idx_value, config_path.parent),
            csfs_file=_resolve_relative_path(csfs_value, config_path.parent),
            label=str(label) if label is not None else None,
            rmix_file=(
                _resolve_relative_path(rmix_value, config_path.parent)
                if isinstance(rmix_value, str) and rmix_value
                else None
            ),
            select_asfs=_parse_select_asfs(
                raw_unit.get("select_asfs"),
                config_path,
                idx,
            ),
            cumulative_threshold=cumulative_threshold,
        )
        indexed_units.append((idx, unit))

    return [
        unit
        for _idx, unit in sorted(
            indexed_units,
            key=lambda indexed_unit: _j_sort_key(
                indexed_unit[1].label,
                indexed_unit[0],
            ),
        )
    ]


def _output_from_custom_config(
    raw_config: dict[str, Any],
    config_path: Path,
    cli_output_file: str | None,
) -> Path:
    if cli_output_file is not None:
        return _output_path_from_cli(cli_output_file, default_dir=Path.cwd())

    output = raw_config.get("output")
    if isinstance(output, dict):
        output_value = output.get("csfs_file", output.get("file"))
        if isinstance(output_value, str) and output_value:
            return _resolve_relative_path(output_value, config_path.parent)

    output_value = raw_config.get("output_file")
    if isinstance(output_value, str) and output_value:
        return _resolve_relative_path(output_value, config_path.parent)

    return (config_path.parent / "chosen.c").resolve()


def _unit_from_pipeline_config(config: Any) -> CsfsSelectionUnit:
    if config.cal_settings.cal_loop_num > 1:
        if config.cal_path.ml_results_path is None:
            raise ValueError("cal_path.ml_results_path 未设置")
        idx_file = config.cal_path.ml_results_path.with_suffix(".npy")
    else:
        idx_file = (
            Path(config.cal_settings.root_path)
            / f"{config.target.conf}_presampled_idxs.npy"
        )

    parquet_path = config.cal_path.full_CSFs_set_parquet_path
    csfs_file = (
        parquet_path
        if parquet_path.is_file()
        else config.cal_path.full_CSFs_set_file_path
    )
    return CsfsSelectionUnit(
        idx_file=idx_file,
        csfs_file=csfs_file,
        label=config.target.conf,
    )


def _pipeline_paths_from_raw_config(
    raw_config: dict[str, Any],
    config_path: Path,
) -> tuple[CsfsSelectionUnit, Path]:
    """Resolve the legacy Tools pipeline paths without importing Tools models."""
    target = raw_config.get("target")
    cal_settings = raw_config.get("cal_settings")
    if not isinstance(target, dict) or not isinstance(cal_settings, dict):
        raise ValueError(f"{config_path} 缺少 target 或 cal_settings 配置")

    conf = str(target["conf"])
    root_path = _resolve_relative_path(
        str(cal_settings["root_path"]),
        config_path.parent,
    )
    loop_num = int(cal_settings["cal_loop_num"])
    full_csfs_path = _resolve_relative_path(
        str(target["full_CSFs_set_file"]),
        root_path,
    )
    parquet_path = full_csfs_path.with_suffix(".parquet")
    idx_file = (
        root_path / "results" / f"{conf}_{loop_num - 1}_final_sampled_idxs.npy"
        if loop_num > 1
        else root_path / f"{conf}_presampled_idxs.npy"
    )
    unit = CsfsSelectionUnit(
        idx_file=idx_file,
        csfs_file=parquet_path if parquet_path.is_file() else full_csfs_path,
        label=conf,
    )
    output = root_path / f"{conf}_{loop_num}" / f"{conf}_{loop_num}.c"
    return unit, output


def _output_from_pipeline_config(
    config: Any,
    cli_output_file: str | None,
) -> Path:
    if cli_output_file is not None:
        return _output_path_from_cli(
            cli_output_file,
            default_dir=Path(config.cal_settings.root_path),
        )
    return Path(config.cal_path.cal_loop_path) / (
        f"{config.target.conf}_{config.cal_settings.cal_loop_num}.c"
    )


def extract_from_config(
    config_path: Path,
    *,
    output_file: str | None = None,
    workers: int | None = None,
) -> list[SelectedCsfsBlock]:
    """Extract selected CSFs using either custom multi-unit or pipeline TOML."""

    config_path = config_path.resolve()
    raw_config = rtoml.load(config_path)
    if "csfs_units" in raw_config:
        units = _units_from_custom_config(raw_config, config_path)
        resolved_output = _output_from_custom_config(
            raw_config,
            config_path,
            output_file,
        )
    else:
        unit, default_output = _pipeline_paths_from_raw_config(raw_config, config_path)
        units = [unit]
        resolved_output = (
            _output_path_from_cli(output_file, default_dir=default_output.parent)
            if output_file is not None
            else default_output
        )

    return extract_csfs_units(units, resolved_output, workers=workers)


def build_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Extract selected CSFs from idxs and raw CSFs data.",
        formatter_class=RawDescriptionHelpFormatter,
        epilog=CUSTOM_CONFIG_EXAMPLE,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="TOML config file. Supports pipeline config or [[csfs_units]].",
    )
    parser.add_argument(
        "-idx",
        "--idx-file",
        type=Path,
        help="Direct mode: .npy file containing selected CSF indexes.",
    )
    parser.add_argument(
        "-csfs",
        "--csfs-file",
        type=Path,
        help="Direct mode: raw CSFs .parquet or .c file.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        help="Output .c file path. Defaults to chosen.c in direct/custom mode.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count for independent units and rcsfs conversion.",
    )
    return parser


def _direct_args_requested(args: Namespace) -> bool:
    return args.idx_file is not None or args.csfs_file is not None


def run_from_cli(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.config is not None and _direct_args_requested(args):
        parser.error("-c 不能与 -idx/-csfs 同时使用")

    if args.config is not None:
        extract_from_config(
            args.config,
            output_file=args.output_file,
            workers=args.workers,
        )
        return 0

    if _direct_args_requested(args):
        if args.idx_file is None or args.csfs_file is None:
            parser.error("direct mode requires both -idx and -csfs")
        output_file = _output_path_from_cli(
            args.output_file,
            default_dir=Path.cwd(),
        )
        extract_csfs_units(
            [
                CsfsSelectionUnit(
                    idx_file=args.idx_file.resolve(),
                    csfs_file=args.csfs_file.resolve(),
                )
            ],
            output_file,
            workers=args.workers,
        )
        return 0

    parser.error("需要指定 -c，或同时指定 -idx 和 -csfs")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(run_from_cli())
    except Exception as err:
        print(f"程序执行失败: {err}", file=sys.stderr)
        raise SystemExit(1)
