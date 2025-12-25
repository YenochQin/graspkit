#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :data_modules.py
@date :2025/04/09 17:04:00
@author :YenochQin (秦毅)
'''
import numpy as np
from dataclasses import dataclass
from typing import Optional
from numpy.typing import NDArray

@dataclass(frozen=True)
class MixCoefficientData:
    block_num: int
    block_idx_list: list
    block_CSFs_nums: list
    block_energy_count_list: list 
    level_J_value_list: list
    parity_list: list
    block_levels_idx_list: list
    block_energy_list: list
    block_level_energy_list: list
    mix_coefficient_list: list
    level_list: list

@dataclass
class CSFs:
    subshell_info_raw: list[str]
    CSFs_block_j_value: list[str]
    parity: str
    CSFs_block_data: list
    CSFs_block_length: list[int] | NDArray[np.integer]  # 兼容列表或ndarray
    block_num: int

    @classmethod
    def from_dict(cls, data: dict) -> 'CSFs':
        """从字典创建CSFs实例（自动处理NumPy数组转换）"""
        return cls(
            subshell_info_raw=data.get('subshell_info_raw', []),
            CSFs_block_j_value=data.get('CSFs_block_j_value', []),
            parity=data.get('parity', ''),
            CSFs_block_data=data.get('CSFs_block_data', []),
            CSFs_block_length=np.array(data['CSFs_block_length']) 
                if isinstance(data.get('CSFs_block_length', []), list) 
                else data.get('CSFs_block_length', np.array([])),
            block_num=data.get('block_num', 0)
        )

@dataclass
class MLDataCounts:
    total_csfs_count: int
    cal_csfs_count: int

    import_csfs_count: Optional[int] = None
    ml_sampled_count: Optional[int] = None
    ml_new_count: Optional[int] = None
    ml_predicted_count: Optional[int] = None
    final_sampled_count: Optional[int] = None

