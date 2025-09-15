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

from ..data_IO import GraspFileLoad, pkl_storange, pkl_loader, save_descriptors, load_descriptors_with_multi_block
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
    caled_csfs_indices_dict = pkl_loader(caled_csfs_indices_file_path)
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

def merge_historical_ci_data(previous_indices_ci_dict, current_indices_ci_dict, logger) -> Tuple[np.ndarray, np.ndarray]:
    """
    合并历史CI系数数据，取索引并集并比较共有索引的CI系数大小
    
    合并规则：
    1. 取indices的并集
    2. 两个字典indices中的交集对应的ci_squared取较大值
    
    Args:
        previous_indices_ci_dict: 历史CI数据字典，格式为 {0: {"indices": [...], "ci_squared": [...]}}
        current_indices_ci_dict: 当前CI数据字典，格式为 {0: {"indices": [...], "ci_squared": [...]}}
        logger: 日志记录器
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: 合并后的索引数组和CI系数平方数组
    """
    
    # 获取历史数据和当前数据
    previous_indices = np.array(previous_indices_ci_dict[0]["indices"])
    previous_ci_squared = np.array(previous_indices_ci_dict[0]["ci_squared"])
    
    current_indices = np.array(current_indices_ci_dict[0]["indices"])
    current_ci_squared = np.array(current_indices_ci_dict[0]["ci_squared"])
    
    # 创建索引到CI系数的映射
    previous_dict = dict(zip(previous_indices, previous_ci_squared))
    current_dict = dict(zip(current_indices, current_ci_squared))
    
    # 获取索引的并集
    all_indices = set(previous_indices) | set(current_indices)
    
    # 合并CI系数：对于交集索引，取较大值
    merged_indices = []
    merged_ci_squared = []
    
    for idx in sorted(all_indices):
        merged_indices.append(idx)
        
        # 如果索引在两个字典中都存在，取较大的CI系数
        if idx in previous_dict and idx in current_dict:
            max_ci = max(previous_dict[idx], current_dict[idx])
            merged_ci_squared.append(max_ci)
        # 如果只在历史数据中存在
        elif idx in previous_dict:
            merged_ci_squared.append(previous_dict[idx])
        # 如果只在当前数据中存在
        else:  # idx in current_dict
            merged_ci_squared.append(current_dict[idx])
    
    merged_indices = np.array(merged_indices)
    merged_ci_squared = np.array(merged_ci_squared)
    
    logger.info(f"历史数据合并统计:")
    logger.info(f"  历史数据CSFs数量: {len(previous_indices)}")
    logger.info(f"  当前数据CSFs数量: {len(current_indices)}")
    logger.info(f"  合并后CSFs数量: {len(merged_indices)}")
    logger.info(f"  新增CSFs数量: {len(all_indices - set(previous_indices))}")
    logger.info(f"  重复CSFs数量: {len(set(previous_indices) & set(current_indices))}")
    
    return merged_indices, merged_ci_squared
        

def generate_train_csfs_descriptors(
                                    config,
                                    raw_csfs_descriptors: np.ndarray,
                                    logger) -> np.ndarray:
    """
    生成用于机器学习训练的CSFs描述符数据
    基于历次迭代的CI系数数据，取索引并集并比较共有索引的CI系数大小

    Args:
        config: 配置对象
        raw_csfs_descriptors: 原始CSFs描述符数组
        logger: 日志记录器

    Returns:
        np.ndarray: 包含描述符和标签的训练数据
    """

    # 加载当前轮次的CSF索引和CI系数数据（从保存的文件中读取）
    logger.info("加载当前轮次保存的CSF索引和CI系数数据")

    # 加载当前轮次选择的CSF索引
    current_indices_ci_path = config.root_path / 'results' / f'{config.conf}_{config.cal_loop_num}_ci_squared.pkl'
    if not current_indices_ci_path.exists():
        raise FileNotFoundError(f"当前轮次CSF索引文件不存在: {current_indices_ci_path}")

    current_indices_ci_dict = pkl_loader(current_indices_ci_path)
    try:
        assert 0 in current_indices_ci_dict, "缺少主键 0"
        assert "indices" in current_indices_ci_dict[0], "缺少子键 indices"
        assert "ci_squared" in current_indices_ci_dict[0], "缺少子键 ci_squared"
        assert current_indices_ci_dict[0]["indices"] is not None, "indices 值为空"
        assert current_indices_ci_dict[0]["ci_squared"] is not None, "ci_squared 值为空"
        
        print("✅ 所有键值验证通过")
        
    except AssertionError as e:
        print(f"❌ 验证失败: {e}")

    current_selected_indices = np.array(current_indices_ci_dict[0]["indices"])

    if config.cal_loop_num > 1:
        # 读取历次迭代保存的CI系数数据
        previous_indices_ci_path = config.root_path / 'results' / f'{config.conf}_previous_ci_squared.pkl'
        if not previous_indices_ci_path.exists():
            raise FileNotFoundError(f"当前轮次CSF索引文件不存在: {previous_indices_ci_path}")

        previous_indices_ci_dict = pkl_loader(previous_indices_ci_path)
        try:
            assert 0 in previous_indices_ci_dict, "缺少主键 0"
            assert "indices" in previous_indices_ci_dict[0], "缺少子键 indices"
            assert "ci_squared" in previous_indices_ci_dict[0], "缺少子键 ci_squared"
            assert previous_indices_ci_dict[0]["indices"] is not None, "indices 值为空"
            assert previous_indices_ci_dict[0]["ci_squared"] is not None, "ci_squared 值为空"
            
            logger.info("开始读取历次迭代的CI系数数据")
            
        except AssertionError as e:
            print(f"❌ 验证失败: {e}")
            
        accumulated_indices, accumulated_ci_squared = merge_historical_ci_data(
                                                            previous_indices_ci_dict, 
                                                            current_indices_ci_dict, 
                                                            logger)
        
        logger.info(f"训练数据")
        logger.info(f"CSF总数: {len(accumulated_indices)}")
        logger.info(f"当前轮次CSF数: {len(current_selected_indices)}")
        
    elif config.cal_loop_num == 1:
        accumulated_indices = current_indices_ci_dict[0]["indices"]
        accumulated_ci_squared = current_indices_ci_dict[0]["ci_squared"]
        logger.info(f"训练数据")
        logger.info(f"CSF总数: {len(accumulated_indices)}")
        logger.info(f"当前轮次CSF数: {len(current_selected_indices)}")
    else:
        logger.error(f"{config.cal_loop_num=} error")
        raise ValueError(f"Invalid cal_loop_num: {config.cal_loop_num}")
    
    # 初始化变量
    cutoff_value = np.float64(config.cutoff_value)

    selected_csfs_descriptors = raw_csfs_descriptors[accumulated_indices]
    important_csfs_mask = accumulated_ci_squared >= cutoff_value

    logger.info(f"生成完整训练数据: {selected_csfs_descriptors.shape[0]} 个CSF")
    logger.info(f"正样本数量: {np.sum(important_csfs_mask)} (占比: {np.sum(important_csfs_mask)/len(important_csfs_mask):.4f})")

    # 返回完整的训练数据（类似旧版ann3_proba.py的处理方式）
    caled_csfs_descriptors = np.column_stack([selected_csfs_descriptors, important_csfs_mask])
    
    accumulated_ci_data = {
            0: {
                "indices": accumulated_indices,  # CSF索引（对应总池）
                "ci_squared": accumulated_ci_squared  # 对应的CI系数平方（正确能级 × 当前计算CSF）
            }
        }

    accumulated_ci_path = config.root_path / 'results' / f'{config.conf}_previous_ci_squared.pkl'
    pkl_storange(accumulated_ci_data, accumulated_ci_path)
    logger.info(f"保存累积CI系数数据: {accumulated_ci_path} (包含{len(accumulated_indices)}个CSFs)")

    # 保存描述符文件
    cal_path = config.root_path / f'{config.conf}_{config.cal_loop_num}'
    save_descriptors(caled_csfs_descriptors, f'{cal_path}/{config.conf}_{config.cal_loop_num}_full', 'npy')
    logger.info(f"保存完整历史数据并集描述符文件: {cal_path}/{config.conf}_{config.cal_loop_num}_full.npy")

    logger.info(f"CSFs描述符标签生成完成")
    logger.info(f"正样本数量: {np.sum(important_csfs_mask)} (在正确能级位置混合系数 ≥ {cutoff_value})")
    logger.info(f"负样本总数量: {len(important_csfs_mask) - np.sum(important_csfs_mask)}")
    logger.info(f"正样本比例: {np.sum(important_csfs_mask) / len(important_csfs_mask):.4f}")

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
    uncurrent_selected_indices = list(all_indices - chosen_indices)
    
    # 返回对应的描述符
    return raw_csfs_descriptors[uncurrent_selected_indices]


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
