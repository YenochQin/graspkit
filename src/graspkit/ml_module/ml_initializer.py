#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :machine_learning_initialization.py
@date :2025/06/09 15:13:58
@author :YenochQin (秦毅)
'''

import logging
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

from ..data_IO import GraspFileLoad, load_csfs_binary, load_descriptors, csfs_index_load, save_descriptors, load_descriptors_with_multi_block
from ..data_IO.h5_descriptor_loader import load_hdf5_descriptors
from ..grasp_data_extractor.ASF_data_collection import LevelsEnergyData
from ..CSFs_processor import batch_asfs_mix_square_above_threshold
from ..utils.data_modules import MixCoefficientData
from ..utils.environment_config import get_environment_config

def setup_logging(config):
    """配置日志系统，支持环境感知"""
    env_config = get_environment_config()
    log_config = env_config.get_logging_config()
    
    # 创建日志目录
    log_dir = config.root_path / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 配置日志级别
    log_level = getattr(logging, log_config['level'])
    
    # 创建处理器列表
    handlers = []
    handlers.append(logging.FileHandler(log_dir / "machine_learning_training.log", encoding='utf-8'))
    
    # 在调试模式下添加控制台输出
    if not env_config.is_production_mode:
        handlers.append(logging.StreamHandler())
    
    # 配置日志
    logging.basicConfig(
        level=log_level,
        format=log_config['format'],
        handlers=handlers,
        force=True  # 强制重新配置
    )
    
    logger = logging.getLogger(__name__)
    
    # 输出环境信息
    env_info = env_config.get_environment_info()
    logger.info(f"🔧 环境配置 - SLURM: {env_info['is_slurm']}, 调试模式: {env_info['is_debug']}, 生产模式: {env_info['is_production']}")
    
    if env_info['slurm_job_id']:
        logger.info(f"🔧 SLURM作业ID: {env_info['slurm_job_id']}")
    
    return logger

def setup_directories(config):
    """创建必要的目录结构"""
    
    directories = ["models", "descripotors", "test_data", "roc_curves", "results"]
    
    for directory in directories:
        (config.root_path / directory).mkdir(parents=True, exist_ok=True)
    
    return '目录创建成功'

def initialize_iteration_results_csv(config, logger=None):
    """
    初始化迭代结果CSV文件的表头
    
    Args:
        config: 配置对象
        logger: 日志记录器
    """
    results_file = Path(config.root_path) / 'results' / 'iteration_results.csv'
    
    # 如果文件已存在，不重新创建表头
    if results_file.exists():
        if logger:
            logger.info(f"迭代结果文件已存在: {results_file}")
        return
    
    # 创建目录
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入表头
    headers = [
        'training_time', 'eval_time', 'execution_time', 'total_time',
        'test_f1', 'test_roc_auc', 'test_accuracy', 'test_precision', 'test_recall',
        'Es_term', 'import_count', 'stay_count', 'MLsampling_ratio', 'chosen_count', 'weight',
        'train_f1', 'train_roc_auc', 'train_accuracy', 'train_precision', 'train_recall'
    ]
    
    with open(results_file, mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    
    if logger:
        logger.info(f"初始化迭代结果CSV文件: {results_file}")

def validate_initial_files(config, logger) -> None:
    """验证初始文件的存在和有效性"""
    # 验证目标总组态文件
    target_pool_file_path = config.root_path / config.target_pool_file
    try:
        if not target_pool_file_path.is_file():
            logger.error(f"目标总组态文件无效或不存在: {target_pool_file_path}")
            raise FileNotFoundError(f"目标总组态文件无效或不存在: {target_pool_file_path}")
        logger.info(f"成功加载目标总组态文件: {target_pool_file_path}")
    except PermissionError as e:
        logger.error(f"无权限访问目标总组态文件: {target_pool_file_path}")
        raise
    except Exception as e:
        logger.error(f"加载目标总组态文件时发生未知错误: {str(e)}")
        raise


def load_data_files(config, logger) -> tuple:
    """加载数据文件
    
    Args:
        config: 配置对象
        logger: 日志记录器
        use_cpp: 是否使用C++生成的HDF5文件格式
    
    Returns:
        tuple: (energy_level_data_pd, rmix_file_data, raw_csfs_descriptors, cal_csfs_data, caled_csfs_indices_dict)
    """
    # config.yaml文件读取时已经处理好root_path和config.scf_cal_path路径
    
    # 加载能级文件
    energy_level_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.level'
    energy_level_file_load = LevelsEnergyData.from_filepath(str(energy_level_file_path), 'LEVEL')
    energy_level_data_pd = energy_level_file_load.energy_level_2_pd()
    logger.info(f"加载能级数据: {energy_level_file_path}")
    
    # 加载rmix文件
    # 根据计算轮次确定文件后缀
    if config.cal_method == 'rmcdhf':
        rmix_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.m'
    elif config.cal_method == 'rci':
        rmix_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.cm'
    else:
        raise ValueError(f"不支持的计算方法: {config.cal_method}")
    
    rmix_file_load = GraspFileLoad.from_filepath(str(rmix_file_path), 'mix')
    rmix_file_data = rmix_file_load.data_file_process()
    logger.info(f"加载 mix coefficient 文件数据: {rmix_file_path}")
    
    # 加载初始 CSFs 描述符文件
    target_pool_file_path = config.root_path / f'{config.conf}'
    use_cpp = config.ml_config.get('use_cpp_descriptor_generator', False)
    if use_cpp:
        # 使用C++生成的HDF5文件
        hdf5_file_path = target_pool_file_path.with_suffix('.h5')
        try:
            hdf5_data = load_hdf5_descriptors(str(hdf5_file_path))
            raw_csfs_descriptors = hdf5_data['descriptors']
            raw_csfs_indices = hdf5_data.get('labels', None)
            logger.info(f"使用C++ HDF5文件加载初始 CSFs 描述符: {hdf5_file_path}")
        except Exception as e:
            logger.warning(f"C++ HDF5文件加载失败: {e}")
            raise FileNotFoundError(f"无法加载初始 CSFs 描述符文件: {target_pool_file_path}")
    else:
        # 使用传统文件格式
        result = load_descriptors_with_multi_block(target_pool_file_path, 'npy')
        if result is None:
            raise FileNotFoundError(f"无法加载初始 CSFs 描述符文件: {target_pool_file_path}")
        raw_csfs_descriptors, raw_csfs_indices = result
        logger.info(f"加载初始 CSFs 描述符文件: {target_pool_file_path}")
    
    # 加载本轮计算CSFs文件
    cal_csfs_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}.c'
    cal_csfs_file_laod = GraspFileLoad.from_filepath(str(cal_csfs_file_path), 'CSFs')
    cal_csfs_data = cal_csfs_file_laod.data_file_process()
    logger.info(f"加载本轮计算 CSFs 文件: {cal_csfs_file_path}")
    
    # 加载本轮选择的CSFs的索引文件
    caled_csfs_indices_file_path = config.scf_cal_path / f'{config.conf}_{config.cal_loop_num}_chosen_indices.pkl'
    caled_csfs_indices_dict = csfs_index_load(caled_csfs_indices_file_path)
    logger.info(f"加载本轮选择的 CSFs 的索引文件: {caled_csfs_indices_file_path}")
    
    return energy_level_data_pd, rmix_file_data, raw_csfs_descriptors, cal_csfs_data, caled_csfs_indices_dict

def check_configuration_coupling(config, energy_level_data_pd, logger):
    """检查组态耦合是否正确"""
    cal_configuration_list = energy_level_data_pd['configuration'].tolist()
    
    # 统计config.spectral_term中每个谱项的出现次数
    spectral_term_counts = {}
    for term in config.spectral_term:
        spectral_term_counts[term] = spectral_term_counts.get(term, 0) + 1
    
    # 检查每个光谱项的出现次数是否与配置中的要求一致，并记录位置
    spectral_term_positions = []
    all_found_correctly = True
    
    for term in set(config.spectral_term):  # 使用set去重，避免重复检查
        expected_count = spectral_term_counts[term]
        actual_count = cal_configuration_list.count(term)
        
        if actual_count == expected_count:
            # 找到所有出现位置
            positions = [i for i, x in enumerate(cal_configuration_list) if x == term]
            spectral_term_positions.extend(positions)
            if expected_count == 1:
                logger.info(f"光谱项 '{term}' 在位置 {positions[0]} 找到")
            else:
                logger.info(f"光谱项 '{term}' 在位置 {positions} 找到（期望 {expected_count} 次，实际 {actual_count} 次）")
        elif actual_count == 0:
            logger.error(f"光谱项 '{term}' 未找到")
            all_found_correctly = False
        else:
            logger.error(f"光谱项 '{term}' 出现 {actual_count} 次，期望 {expected_count} 次")
            all_found_correctly = False
    
    if all_found_correctly:
        # 按位置排序，保持一致的输出顺序
        spectral_term_positions.sort()
        logger.info(f"cal_loop {config.cal_loop_num} 组态耦合正确，位置索引: {spectral_term_positions}")
        return True, spectral_term_positions
    else:
        logger.error(f"cal_loop {config.cal_loop_num} 组态耦合错误")
        return False, []


def check_energy_convergence(
                        config, 
                        logger, 
                        current_energy_data: pd.DataFrame, 
                        convergence_threshold: float = 0.001
                        ) -> bool:
    """
    检查能量收敛性：比较当前轮与上一轮的能量差异
    
    Args:
        config: 配置对象
        logger: 日志记录器
        current_energy_data: 当前轮的能量数据DataFrame
        convergence_threshold: 收敛阈值，默认为0.001（绝对能量差）
    
    Returns:
        bool: True表示继续计算，False表示需要回退到上一轮重算
    """
    
    try:
        # 获取上一轮的能量数据文件路径
        previous_energy_path = config.root_path / f'{config.conf}_{config.loop_num}' / f'{config.conf}_{config.cal_loop_num-1}_correct_levels.csv'
        
        if not previous_energy_path.exists():
            logger.warning(f"未找到上一轮能量数据: {previous_energy_path}")
            return True
        
        # 加载上一轮能量数据
        previous_energy_data = pd.read_csv(previous_energy_path)
        
        # 检查数据一致性
        if len(current_energy_data) != len(previous_energy_data):
            logger.warning(f"当前轮与能量数据数量不一致: 当前{len(current_energy_data)} vs 上轮{len(previous_energy_data)}")
            return True
        
        # 获取能量列
        current_energies = np.asarray(current_energy_data['EnergyTotal'], dtype=np.float64)
        previous_energies = np.asarray(previous_energy_data['EnergyTotal'], dtype=np.float64)
        
        # 计算绝对能量差异
        energy_diffs = np.abs(current_energies - previous_energies)
        max_diff = np.max(energy_diffs)
        
        # 检查是否超过阈值
        if max_diff > convergence_threshold:
            max_diff_idx = np.argmax(energy_diffs)
            logger.warning(f"检测到能量不收敛: 第{max_diff_idx+1}个能级能量差异{max_diff:.6f} (阈值: {convergence_threshold:.6f})")
            logger.warning(f"当前能量: {current_energies[max_diff_idx]:.7f}")
            logger.warning(f"上轮能量: {previous_energies[max_diff_idx]:.7f}")
            return False
        
        logger.info("能量收敛检查通过，继续计算")
        return True
        
    except Exception as e:
        logger.error(f"能量收敛检查过程中发生错误: {str(e)}")
        return True  # 发生错误时继续计算，避免阻塞

def evaluate_calculation_convergence(config, logger, current_calculation_csfs=None):
    """
    检查GRASP计算的收敛性
    
    使用最近三次计算结果的能级数据和组态数量数据，通过计算：
    1. 每个能级的标准差
    2. 组态数量的相对标准差
    来判定收敛。
    
    Args:
        config: 配置对象
        logger: 日志记录器
        current_calculation_csfs: 当前轮的CSFs数量（可选，如果提供则不从文件读取）
        
    Returns:
        bool: True表示继续计算，False表示已收敛停止计算
    """
    try:
        # 读取最近3次计算的能级数据
        energy_data_list = []
        
        for i in range(3):
            loop_num = config.cal_loop_num - 2 + i  # 前3次：当前-2, 当前-1, 当前
            csv_path = config.root_path / f'{config.conf}_{loop_num}' / f'{config.conf}_{loop_num}_correct_levels.csv'
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                energy_data_list.append(df)
                logger.info(f"读取第{loop_num}轮能级数据: {csv_path}")
            else:
                logger.warning(f"未找到第{loop_num}轮能级数据文件: {csv_path}")
                return True  # 文件不存在，继续计算
        
        if len(energy_data_list) < 3:
            logger.warning("无法读取完整的3轮能级数据，继续计算")
            return True
        
        # 读取组态数量数据
        iteration_results_path = config.root_path / 'results' / 'iteration_results.csv'
        csfs_num = []  # 存储每轮的组态数量
        
        # 如果提供了当前轮的CSFs数量，则优先使用
        if current_calculation_csfs is not None:
            # 从iteration_results.csv读取前两轮的数据，使用传入的当前轮数据
            if iteration_results_path.exists():
                try:
                    iteration_df = pd.read_csv(iteration_results_path)
                    # 获取前两轮的组态数量
                    for i in range(2):  # 只读取前两轮
                        loop_num = config.cal_loop_num - 2 + i
                        # 查找对应轮次的数据
                        loop_data = iteration_df[iteration_df['iteration'] == loop_num]
                        if not loop_data.empty:
                            current_count = loop_data['current_calculation_count'].iloc[0]
                            csfs_num.append(current_count)
                            logger.info(f"读取第{loop_num}轮组态数量: {current_count}")
                        else:
                            logger.warning(f"在iteration_results.csv中未找到第{loop_num}轮的数据")
                            return True  # 数据不完整，继续计算
                    
                    # 添加当前轮的CSFs数量
                    csfs_num.append(current_calculation_csfs)
                    logger.info(f"读取第{config.cal_loop_num}轮组态数量: {current_calculation_csfs}")
                    
                    if len(csfs_num) < 3:
                        logger.warning("无法读取完整的3轮组态数量数据，继续计算")
                        return True
                        
                except Exception as e:
                    logger.warning(f"读取iteration_results.csv文件出错: {e}")
                    return True
            else:
                logger.warning(f"未找到iteration_results.csv文件: {iteration_results_path}")
                return True
        else:
            # 原有逻辑：从iteration_results.csv读取所有3轮数据
            if iteration_results_path.exists():
                try:
                    iteration_df = pd.read_csv(iteration_results_path)
                    # 获取最近3轮的组态数量
                    for i in range(3):
                        loop_num = config.cal_loop_num - 2 + i
                        # 查找对应轮次的数据
                        loop_data = iteration_df[iteration_df['iteration'] == loop_num]
                        if not loop_data.empty:
                            current_count = loop_data['current_calculation_count'].iloc[0]
                            csfs_num.append(current_count)
                            logger.info(f"读取第{loop_num}轮组态数量: {current_count}")
                        else:
                            logger.warning(f"在iteration_results.csv中未找到第{loop_num}轮的数据")
                            return True  # 数据不完整，继续计算
                    
                    if len(csfs_num) < 3:
                        logger.warning("无法读取完整的3轮组态数量数据，继续计算")
                        return True
                        
                except Exception as e:
                    logger.warning(f"读取iteration_results.csv文件出错: {e}")
                    return True
            else:
                logger.warning(f"未找到iteration_results.csv文件: {iteration_results_path}")
                return True
        
        # === 1. 能级标准差计算 ===
        # 获取所有configuration
        configurations = energy_data_list[0]['configuration'].tolist()
        
        # 存储每个能级的标准差
        std_deviations = []
        
        for config_name in configurations:
            # 获取该configuration在3轮计算中的能级值
            energy_values = []
            for df in energy_data_list:
                if config_name in df['configuration'].values:
                    energy = df[df['configuration'] == config_name]['EnergyTotal'].iloc[0]
                    energy_values.append(energy)
                else:
                    logger.warning(f"在第{len(energy_values)+1}轮数据中未找到configuration: {config_name}")
                    return True  # 数据不完整，继续计算
            
            if len(energy_values) == 3:
                # 计算标准差
                energy_std = np.std(energy_values)
                std_deviations.append(energy_std)
                
                logger.debug(f"Configuration {config_name}: "
                            f"能级值={energy_values}, "
                            f"标准差={energy_std:.5e}")
        
        # 计算所有能级的平均标准差
        avg_energy_std = np.mean(std_deviations)
        
        # === 2. 组态数量相对标准差计算 ===
        # 计算组态数量的标准差和相对标准差
        csfs_num_std = np.std(csfs_num)
        csfs_num_mean = np.mean(csfs_num)
        
        if csfs_num_mean > 0:
            csfs_num_relative_std = csfs_num_std / csfs_num_mean
        else:
            csfs_num_relative_std = csfs_num_std  # 如果平均值为零，直接使用标准差
        
        # 从配置文件读取收敛阈值（如果没有设置则使用默认值）
        energy_std_threshold = getattr(config, 'energy_std_threshold', 1e-5)  # 能级标准差阈值
        csfs_num_relative_std_threshold = getattr(config, 'csfs_num_relative_std_threshold', 1e-3)  # 组态数量相对标准差阈值（5%）
        
        logger.info(f"收敛性统计:")
        logger.info(f"  最近3轮组态数量: {csfs_num}")
        logger.info(f"  组态数量平均值: {csfs_num_mean:.1f}")
        logger.info(f"  组态数量标准差: {csfs_num_std:.2f}")
        logger.info(f"  组态数量相对标准差: {csfs_num_relative_std:.4f} (阈值: {csfs_num_relative_std_threshold:.4f})")
        logger.info(f"  能级平均标准差: {avg_energy_std:.5e} (阈值: {energy_std_threshold:.5e})")
        logger.info(f"  能级标准差收敛: {avg_energy_std < energy_std_threshold}")
        logger.info(f"  组态数量相对标准差收敛: {csfs_num_relative_std < csfs_num_relative_std_threshold}")
        
        # 判断收敛性：两个条件都满足才算收敛
        energy_converged = avg_energy_std < energy_std_threshold
        csfs_num_converged = csfs_num_relative_std < csfs_num_relative_std_threshold
        is_converged = energy_converged and csfs_num_converged
        
        if is_converged:
            logger.info("能级和组态数量都已收敛，停止计算")
            return False
        else:
            if not energy_converged:
                logger.info("能级未完全收敛，继续计算")
            if not csfs_num_converged:
                logger.info("组态数量未稳定收敛，继续计算")
            return True
            
    except Exception as e:
        logger.error(f"收敛检查过程中出错: {e}")
        return True  # 出错时继续计算


def generate_chosen_csfs_descriptors(
                                    config, 
                                    chosen_csfs_indices_dict: Dict, 
                                    raw_csfs_descriptors: np.ndarray, 
                                    rmix_file_data: MixCoefficientData, asfs_position: List[int], 
                                    logger, 
                                    include_wrong_level_negatives: bool = False) -> np.ndarray:
    """
    生成用于机器学习训练的CSFs描述符数据
    
    Args:
        config: 配置对象
        chosen_csfs_indices_dict: 选中的CSFs索引字典
        raw_csfs_descriptors: 原始CSFs描述符数组
        rmix_file_data: 混合系数数据
        asfs_position: 正确能级位置索引列表
        logger: 日志记录器
        include_wrong_level_negatives: 是否包含错误能级的组态作为负样本
        
    Returns:
        np.ndarray: 包含描述符和标签的训练数据
    """
    
    # 验证字典并安全获取block 0的索引
    if not chosen_csfs_indices_dict:
        raise ValueError("chosen_csfs_indices_dict为空，无法获取选中的CSFs索引")
    
    if 0 not in chosen_csfs_indices_dict:
        raise KeyError(f"chosen_csfs_indices_dict中缺少键0，可用键: {list(chosen_csfs_indices_dict.keys())}")
    
    selected_indices = np.array(chosen_csfs_indices_dict[0])
    selected_csfs_descriptors = raw_csfs_descriptors[selected_indices]
    
    # 获取所有能级的混合系数数据
    cal_mix_coeffs = rmix_file_data.mix_coefficient_List[0]  # shape: (n_levels, n_csfs)
    
    ## 方案1：仅使用正确能级位置的数据（原有方案）
    if not include_wrong_level_negatives:
        # 获取正确能级位置的混合系数平方和
        correct_mix_coeff_squared_sum = np.sum(cal_mix_coeffs[asfs_position]**2, axis=0)
        
        # 生成标签（只基于正确能级位置的混合系数）
        csf_mix_coeff_descriptors = correct_mix_coeff_squared_sum >= np.float64(config.cutoff_value)
        
        logger.info(f"使用原有方案（仅正确能级）")
        logger.info(f"正确能级位置: {asfs_position}")
        logger.info(f"本轮计算CSFs数量: {len(selected_csfs_descriptors)}")
        logger.info(f"超过阈值的CSFs数量: {np.sum(csf_mix_coeff_descriptors)}")
        logger.info(f"正样本比例: {np.sum(csf_mix_coeff_descriptors)/len(csf_mix_coeff_descriptors):.4f}")
        
        caled_csfs_descriptors = np.column_stack([selected_csfs_descriptors, csf_mix_coeff_descriptors])
    
    ## 方案2：增强方案 - 包含错误能级的组态作为负样本
    else:
        # 重要：cal_mix_coeffs的shape是(n_levels, cal_csfs_num)
        # 其中cal_csfs_num是本轮计算的CSFs数量
        
        # 获取正确能级位置的混合系数平方和
        correct_mix_coeff_squared_sum = np.sum(cal_mix_coeffs[asfs_position]**2, axis=0)
        
        # 获取错误能级位置的索引
        wrong_level_indices = list(set(range(cal_mix_coeffs.shape[0])) - set(asfs_position))
        
        # 获取错误能级位置的混合系数平方和
        if len(wrong_level_indices) > 0:
            wrong_mix_coeff_squared_sum = np.sum(cal_mix_coeffs[wrong_level_indices]**2, axis=0)
        else:
            wrong_mix_coeff_squared_sum = np.zeros(cal_mix_coeffs.shape[1])
        
        # 生成增强标签
        cutoff_value = np.float64(config.cutoff_value)
        
        # 正样本：在正确能级位置有较高混合系数
        positive_mask = correct_mix_coeff_squared_sum >= cutoff_value
        
        # 负样本包括：
        # 1. 在正确能级位置混合系数较低的CSFs
        # 2. 在错误能级位置有较高混合系数但在正确能级位置较低的CSFs（这些是"坏"组态）
        negative_mask_low_correct = correct_mix_coeff_squared_sum < cutoff_value
        negative_mask_high_wrong = (wrong_mix_coeff_squared_sum >= cutoff_value * 100) & (correct_mix_coeff_squared_sum < cutoff_value)
        
        # 最终标签：正样本为True，负样本为False
        csf_mix_coeff_descriptors = positive_mask
        
        # 统计信息
        n_positive = np.sum(positive_mask)
        n_negative_low_correct = np.sum(negative_mask_low_correct & ~negative_mask_high_wrong)
        n_negative_high_wrong = np.sum(negative_mask_high_wrong)
        n_total = len(selected_csfs_descriptors)
        
        logger.info(f"使用增强方案（包含错误能级负样本）")
        logger.info(f"正确能级位置: {asfs_position}")
        logger.info(f"错误能级位置: {wrong_level_indices}")
        logger.info(f"本轮计算CSFs数量: {n_total}")
        logger.info(f"正样本数量: {n_positive} (在正确能级位置混合系数 ≥ {cutoff_value})")
        logger.info(f"负样本数量: {n_total - n_positive}")
        logger.info(f"  - 正确能级位置低混合系数: {n_negative_low_correct}")
        logger.info(f"  - 错误能级位置高混合系数: {n_negative_high_wrong}")
        logger.info(f"正样本比例: {n_positive/n_total:.4f}")
        logger.info(f"错误能级高混合系数比例: {n_negative_high_wrong/n_total:.4f}")
        
        # 如果有错误能级的高混合系数组态，说明这些是"坏"组态
        if n_negative_high_wrong > 0:
            logger.info(f"发现 {n_negative_high_wrong} 个在错误能级有高混合系数的组态，这些将作为负样本帮助模型学习识别错误组态")
        
        caled_csfs_descriptors = np.column_stack([selected_csfs_descriptors, csf_mix_coeff_descriptors])
    
    # 保存描述符文件
    cal_path = config.root_path / f'{config.conf}_{config.cal_loop_num}'
    save_descriptors(caled_csfs_descriptors, f'{cal_path}/{config.conf}_{config.cal_loop_num}', 'npy')
    logger.info(f"保存本轮选择的 CSFs 的描述符文件: {cal_path}/{config.conf}_{config.cal_loop_num}.npy")

    return caled_csfs_descriptors


def get_unselected_descriptors(raw_csfs_descriptors: np.ndarray, chosen_csfs_indices_dict: Dict[int, List[int]]) -> np.ndarray:
    """
    找出不在chosen_csfs_indices_dict索引中的描述符
    
    Args:
        raw_csfs_descriptors: 原始CSFs描述符数组
        chosen_csfs_indices_dict: 已选择的CSFs索引字典，格式为{block_index: [indices]}
        
    Returns:
        np.ndarray: 不在chosen_csfs_indices_dict中的描述符数组
    """
    # 获取所有已选择的索引
    chosen_indices = []
    for block_indices in chosen_csfs_indices_dict.values():
        chosen_indices.extend(block_indices)
    chosen_indices = set(chosen_indices)
    
    # 获取所有可能的索引
    all_indices = set(range(len(raw_csfs_descriptors)))
    
    # 找出不在chosen_indices中的索引
    unselected_indices = list(all_indices - chosen_indices)
    
    # 返回对应的描述符
    return raw_csfs_descriptors[unselected_indices]


def get_stay_descriptors(raw_csfs_descriptors: np.ndarray, chosen_csfs_indices_dict: Dict[int, List[int]]) -> np.ndarray:
    """
    找出不在chosen_csfs_indices_dict索引中的描述符
    
    Args:
        raw_csfs_descriptors: 原始CSFs描述符数组
        chosen_csfs_indices_dict: 已选择的CSFs索引字典，格式为{block_index: [indices]}
        
    Returns:
        np.ndarray: 不在chosen_csfs_indices_dict中的描述符数组
    """
    # 验证字典并安全获取所有已选择的索引
    if not chosen_csfs_indices_dict:
        raise ValueError("chosen_csfs_indices_dict为空，无法获取选中的CSFs索引")
    
    if 0 not in chosen_csfs_indices_dict:
        raise KeyError(f"chosen_csfs_indices_dict中缺少键0，可用键: {list(chosen_csfs_indices_dict.keys())}")
    
    chosen_indices = set(chosen_csfs_indices_dict[0])
    
    # 获取所有可能的索引
    all_indices = set(range(len(raw_csfs_descriptors)))
    
    # 找出不在chosen_indices中的索引
    stay_indices = list(all_indices - chosen_indices)
    
    # 返回对应的描述符
    return raw_csfs_descriptors[stay_indices]
