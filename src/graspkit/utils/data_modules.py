# -*- encoding: utf-8 -*-
"""
@Id :data_modules.py
@date :2025/04/09 17:04:00
@author :YenochQin (秦毅)
"""

from dataclasses import dataclass
from typing import TypedDict
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MixCoefficientData:
    block_num: int
    block_idx_list: list[int]
    block_CSFs_nums: list[int]
    block_energy_count_list: list[int]
    level_J_value_list: list[str]
    parity_list: list[int]
    block_levels_idx_list: list[NDArray[np.int32]]
    block_energy_list: list[float]
    block_level_energy_list: list[NDArray[np.float64]]
    mix_coefficient_list: list[NDArray[np.float64]]
    level_list: list[float]


class CSFsDict(TypedDict, total=False):
    subshell_info_raw: list[str]
    CSFs_block_j_value: list[str]
    parity: str
    CSFs_block_data: list[list[list[str]]]
    CSFs_block_length: list[int]
    block_num: int

@dataclass
class CSFs:
    subshell_info_raw: list[str]
    CSFs_block_j_value: list[str]
    parity: str
    CSFs_block_data: list[
        list[list[str]]
    ]  # list of blocks, each block is list of CSFs (3 lines each)
    CSFs_block_length: list[int]
    block_num: int

    @classmethod
    def from_dict(cls, data: CSFsDict) -> "CSFs":
        "从字典创建CSFs实例（自动处理NumPy数组转换）"
        return cls(
            subshell_info_raw=data.get("subshell_info_raw", []),
            CSFs_block_j_value=data.get("CSFs_block_j_value", []),
            parity=data.get("parity", ""),
            CSFs_block_data=data.get("CSFs_block_data", []),
            CSFs_block_length=data.get("CSFs_block_length", []),
            block_num=data.get("block_num", 0),
        )


@dataclass
class MLDataCounts:
    total_csfs_count: int
    cal_csfs_count: int

    important_csfs_count: int | None = None
    ml_sampled_count: int | None = None
    ml_predicted_count: int | None = None
    final_sampled_count: int | None = None

    # 重要组态留存率 (important Retention Rate)：本轮计算的重要组态/上一轮计算的重要组态
    important_retention_rate: float | None = None
    # 验证留存率 (Screening Retention Rate) / 良品率:经过实际计算（或仿真/实验）后，有多少数据被认为是“好”的并保留下来。
    screening_retention_rate: float | None = None
    # ML 预测留存率 (ML Selection Retention Rate) : 进行预测并截断时产生的留存率,模型对未知空间的探索力度。
    ml_retention_rate: float | None = None
    # 迭代增长率 (Iteration Growth/Retention Rate): 下一次计算的规模相对于这一次的变化, 控制计算成本。如果 $>1$，计算量在发散；如果 $<1$，计算量在收敛。
    iteration_retention_rate: float | None = None
