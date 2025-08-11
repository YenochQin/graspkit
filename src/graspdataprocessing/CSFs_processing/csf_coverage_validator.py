#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :csf_coverage_validator.py
@date :2025/08/11 16:45:06
@author :Claude Code
'''
import numpy as np
from typing import List, Tuple


def validate_csf_descriptors_coverage(descriptors: np.ndarray, 
                                    n_orbitals: int = None, 
                                    with_subshell_info: bool = False) -> Tuple[bool, List[int]]:
    """
    验证选取的CSFs描述符子集是否满足覆盖条件：
    对于每个轨道，至少有一个CSF在其对应的电子填充数位置不为零
    
    Args:
        descriptors (np.ndarray): 选取出的CSFs描述符数组，形状为 (n_csfs, n_features)
        n_orbitals (int, optional): 轨道数量。如果为None，则从描述符结构推断
        with_subshell_info (bool): 是否包含子壳层信息
            - False: 使用parse_csf_2_descriptor生成的描述符（每个轨道3个值）
            - True: 使用parse_csf_2_descriptor_with_subshell生成的描述符（每个轨道5个值）
    
    Returns:
        Tuple[bool, List[int]]: (是否满足覆盖条件, 未覆盖的轨道索引列表)
    """
    # 检查输入参数
    if descriptors.size == 0:
        if n_orbitals is None:
            return False, []
        return False, list(range(n_orbitals))
    
    # 确定每个轨道的电子填充位置索引
    if with_subshell_info:
        # 对于parse_csf_2_descriptor_with_subshell：每个轨道5个值，电子数在索引2, 7, 12, ...位置
        values_per_orbital = 5
        electron_index_in_orbital = 2
    else:
        # 对于parse_csf_2_descriptor：每个轨道3个值，电子数在索引0, 3, 6, ...位置
        values_per_orbital = 3
        electron_index_in_orbital = 0
    
    # 从描述符结构推断轨道数量
    inferred_n_orbitals = descriptors.shape[1] // values_per_orbital
    
    # 如果提供了n_orbitals，验证一致性
    if n_orbitals is not None and n_orbitals != inferred_n_orbitals:
        raise ValueError(f"轨道数量不匹配：期望 {n_orbitals}，实际 {inferred_n_orbitals}")
    
    actual_n_orbitals = inferred_n_orbitals
    
    # 直接通过切片获取每个轨道的电子填充
    electron_indices = np.arange(electron_index_in_orbital, 
                                actual_n_orbitals * values_per_orbital, 
                                values_per_orbital)
    
    # 提取所有CSF的电子数信息
    electron_counts = descriptors[:, electron_indices]  # 形状为 (n_csfs, actual_n_orbitals)
    
    # 检查每个轨道是否至少有一个CSF的电子数不为零
    has_nonzero_electrons = np.any(electron_counts > 0, axis=0)  # 形状为 (actual_n_orbitals,)
    
    # 找出未覆盖的轨道索引
    uncovered_orbitals = np.where(~has_nonzero_electrons)[0].tolist()
    
    # 返回验证结果
    is_covered = len(uncovered_orbitals) == 0
    return is_covered, uncovered_orbitals

def select_csfs_for_coverage(descriptors: np.ndarray,
                            uncovered_orbitals: List[int],
                            full_descriptors: np.ndarray,
                            with_subshell_info: bool = False) -> Tuple[np.ndarray, List[int]]:
    """
    当覆盖验证失败时，从给定的完整描述符中按顺序选取包含缺少轨道的CSF描述符
    
    Args:
        descriptors (np.ndarray): 当前的CSFs描述符数组，形状为 (n_csfs, n_features)
        uncovered_orbitals (List[int]): 未覆盖的轨道索引列表
        full_descriptors (np.ndarray): 完整的CSFs描述符数组，形状为 (n_full_csfs, n_features)
        with_subshell_info (bool): 是否包含子壳层信息
    
    Returns:
        Tuple[np.ndarray, List[int]]: (更新后的描述符数组, 选取的CSF索引列表)
            - 更新后的描述符数组包含原有描述符和新选取的描述符
            - 选取的CSF索引列表对应于full_descriptors中的索引
    """
    if not uncovered_orbitals:
        return descriptors, []
    
    # 确定每个轨道的电子填充位置索引
    if with_subshell_info:
        values_per_orbital = 5
        electron_index_in_orbital = 2
    else:
        values_per_orbital = 3
        electron_index_in_orbital = 0
    
    # 获取每个轨道的电子填充位置索引
    n_orbitals = full_descriptors.shape[1] // values_per_orbital
    electron_indices = np.arange(electron_index_in_orbital, 
                                n_orbitals * values_per_orbital, 
                                values_per_orbital)
    
    # 提取完整描述符中的电子数信息
    full_electron_counts = full_descriptors[:, electron_indices]
    
    # 找出当前描述符中已包含的CSF索引（避免重复选择）
    current_csfs_set = set(range(len(descriptors))) if descriptors.size > 0 else set()
    
    selected_indices = []
    remaining_uncovered = set(uncovered_orbitals)
    
    # 按顺序遍历完整描述符
    for idx in range(len(full_descriptors)):
        if idx in current_csfs_set:
            continue  # 跳过已包含的CSF
            
        # 检查当前CSF是否包含任何剩余未覆盖的轨道
        csf_electrons = full_electron_counts[idx]
        covers_orbitals = [orb for orb in remaining_uncovered if csf_electrons[orb] > 0]
        
        if covers_orbitals:
            selected_indices.append(idx)
            remaining_uncovered -= set(covers_orbitals)
            
            # 如果所有轨道都已覆盖，提前退出
            if not remaining_uncovered:
                break
    
    if not selected_indices:
        return descriptors, []
    
    # 构建更新后的描述符数组
    new_descriptors = full_descriptors[selected_indices]
    
    if descriptors.size == 0:
        updated_descriptors = new_descriptors
    else:
        updated_descriptors = np.vstack([descriptors, new_descriptors])
    
    return updated_descriptors, selected_indices


# 示例用法
if __name__ == "__main__":
    # 示例1：使用parse_csf_2_descriptor生成的描述符
    # 假设有3个轨道，每个CSF有9个特征值(3*3)
    descriptors1 = np.array([
        [2, 0, 4, 0, 0, 0, 1, 0, 2],  # 轨道0有2个电子，轨道2有1个电子
        [0, 0, 0, 4, 3, 2, 0, 0, 0],  # 轨道1有4个电子
    ], dtype=np.float32)
    
    # 完整描述符（包含更多CSF）
    full_descriptors1 = np.array([
        [2, 0, 4, 0, 0, 0, 1, 0, 2],  # 轨道0,2有电子
        [0, 0, 0, 4, 3, 2, 0, 0, 0],  # 轨道1有电子
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 轨道2有电子
        [1, 0, 1, 0, 0, 0, 0, 0, 0],  # 轨道0有电子
        [0, 0, 0, 1, 0, 1, 0, 0, 0],  # 轨道1有电子
    ], dtype=np.float32)
    
    # 验证覆盖
    is_covered1, uncovered1 = validate_csf_descriptors_coverage(descriptors1, with_subshell_info=False)
    print(f"示例1 - 是否满足覆盖条件: {is_covered1}")
    print(f"示例1 - 未覆盖的轨道索引: {uncovered1}")
    
    if not is_covered1:
        updated_descriptors, selected_indices = select_csfs_for_coverage(
            descriptors1, uncovered1, full_descriptors1, with_subshell_info=False)
        print(f"示例1 - 选取的CSF索引: {selected_indices}")
        print(f"示例1 - 更新后的描述符形状: {updated_descriptors.shape}")
        
        # 验证更新后的覆盖
        is_covered_after, uncovered_after = validate_csf_descriptors_coverage(updated_descriptors, with_subshell_info=False)
        print(f"示例1 - 更新后是否满足覆盖条件: {is_covered_after}")
    
    # 示例2：使用parse_csf_2_descriptor_with_subshell生成的描述符
    # 假设有2个轨道，每个CSF有10个特征值(5*2)
    descriptors2 = np.array([
        [5, -1, 2, 0, 4, 4, 2, 0, 0, 0],  # 轨道0有2个电子
    ], dtype=np.float32)
    
    full_descriptors2 = np.array([
        [5, -1, 2, 0, 4, 4, 2, 0, 0, 0],  # 轨道0有电子
        [5, -1, 0, 0, 0, 4, 2, 3, 2, 4],  # 轨道1有电子
        [5, -1, 1, 0, 1, 4, 2, 1, 0, 2],  # 轨道0,1都有电子
    ], dtype=np.float32)
    
    is_covered2, uncovered2 = validate_csf_descriptors_coverage(descriptors2, with_subshell_info=True)
    print(f"\n示例2 - 是否满足覆盖条件: {is_covered2}")
    print(f"示例2 - 未覆盖的轨道索引: {uncovered2}")
    
    if not is_covered2:
        updated_descriptors2, selected_indices2 = select_csfs_for_coverage(
            descriptors2, uncovered2, full_descriptors2, with_subshell_info=True)
        print(f"示例2 - 选取的CSF索引: {selected_indices2}")
        print(f"示例2 - 更新后的描述符形状: {updated_descriptors2.shape}")
        
        is_covered_after2, uncovered_after2 = validate_csf_descriptors_coverage(updated_descriptors2, with_subshell_info=True)
        print(f"示例2 - 更新后是否满足覆盖条件: {is_covered_after2}")