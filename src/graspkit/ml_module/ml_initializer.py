# -*- encoding: utf-8 -*-
"""
@Id :machine_learning_initialization.py
@date :2025/06/09 15:13:58
@author :YenochQin (秦毅)
"""

import logging
from pathlib import Path
from typing import Tuple

import csv
import numpy as np
import pandas as pd
import rtoml

from ..data_IO import (
    GraspFileLoad,
    pkl_storage,
    pkl_loader,
    save_descriptors,
    load_descriptors_with_multi_block,
    load_config
)
from ..data_IO.h5_descriptor_loader import load_hdf5_descriptors
from ..grasp_data_extractor.ASF_data_collection import LevelsEnergyData

from ..utils.environment_config import get_environment_config
from ..utils.data_modules import MixCoefficientData


def setup_config(config_path: str | Path):
    """
    初始化机器学习配置并设置相关路径

    Args:
        config_path (str | Path): 配置文件的路径

    Returns:
        config: 配置对象，包含所有初始化后的路径和参数

    该函数执行以下步骤：
    1. 从指定路径加载配置文件
    2. 调用内部函数设置所有相关文件路径
    3. 返回完整的配置对象
    """
    # 从配置文件路径加载配置信息
    config = load_config(config_path)

    # 设置配置对象中的所有相关路径
    return _setup_config_paths(config)

def _setup_config_paths(config):
    """
    为配置对象设置所有必需的文件路径

    Args:
        config: 包含基本配置信息的对象，应包含root_path、target和cal_settings属性

    Returns:
        config: 更新后的配置对象，包含所有路径信息

    该函数设置以下路径：
    1. 全量CSF集合相关文件路径
    2. 压缩的二进制CSF文件路径
    3. 当前计算循环的路径
    4. 结果文件存储路径
    5. 如果是后续循环，设置前一轮的重要索引和ML结果路径
    """
    # 获取根目录路径，确保是Path对象类型
    root_path = Path(config.cal_settings.root_path)

    # 设置全量CSF集合文件的完整路径
    full_CSFs_set_path = root_path / config.target.full_CSFs_set_file
    config.cal_path.full_CSFs_set_file_path = full_CSFs_set_path
    # 设置CSF二进制和头文件的路径
    config.cal_path.full_CSFs_set_parquet_path = full_CSFs_set_path.with_stem(full_CSFs_set_path.stem).with_suffix('.parquet')
    config.cal_path.full_CSFs_set_header_path = full_CSFs_set_path.with_stem(f"{full_CSFs_set_path.stem}_header").with_suffix('.toml')
    config.cal_path.iteration_results = config.cal_path.results_path / "iteration_results.csv"
    config.cal_path.training_results = config.cal_path.results_path / "training_results.csv"


    config.cal_path.loop_file_name = f'{config.target.conf}_{config.cal_settings.cal_loop_num}'

    # 设置当前计算循环的工作目录路径，格式：{配置名}_{循环编号}
    config.cal_path.cal_loop_path = root_path / config.cal_path.loop_file_name

    # 设置计算结果文件的存储路径
    config.cal_path.results_path = root_path / 'results'
    config.cal_path.test_data_path = root_path / "test_data"
    config.cal_path.models_path = root_path / "models"
    config.cal_path.roc_curves_path = root_path / "roc_curves"
    config.cal_path.log_dir = root_path / "logs"

    # 如果是第二轮及之后的计算循环，需要设置前一轮的相关文件路径
    if config.cal_settings.cal_loop_num > 1:
        # 前一轮计算保存的重要索引文件路径
        config.cal_path.previous_idxs_file = config.cal_path.results_path / f'{config.target.conf}_{config.cal_settings.cal_loop_num-1}_important_idxs'
        # 前一轮机器学习生成的最终采样索引文件路径
        config.cal_path.ml_results_path = config.cal_path.results_path / f'{config.target.conf}_{config.cal_settings.cal_loop_num-1}_final_sampled_idxs'
        config.cal_path.previous_idxs_ci_path = config.cal_path.results_path / f'{config.target.conf}_{config.cal_settings.cal_loop_num-1}_previous_ci_squared.pkl'

    return config


def setup_logging(log_dir: Path):
    """配置日志系统，支持环境感知"""
    env_config = get_environment_config()
    log_config = env_config.get_logging_config()

    # 创建日志目录
    log_dir.mkdir(exist_ok=True)

    # 配置日志级别
    log_level = getattr(logging, log_config["level"])

    # 创建处理器列表
    handlers = []
    handlers.append(
        logging.FileHandler(log_dir / "ml_training.log", encoding="utf-8")
    )

    # 在调试模式下添加控制台输出
    if not env_config.is_production_mode:
        handlers.append(logging.StreamHandler())

    # 配置日志
    logging.basicConfig(
        level=log_level,
        format=log_config["format"],
        handlers=handlers,
        force=True,  # 强制重新配置
    )

    logger = logging.getLogger(__name__)

    # 输出环境信息
    env_info = env_config.get_environment_info()
    logger.info(
        f"环境配置 - SLURM: {env_info['is_slurm']}, 调试模式: {env_info['is_debug']}, 生产模式: {env_info['is_production']}"
    )

    if env_info["slurm_job_id"]:
        logger.info(f"SLURM作业ID: {env_info['slurm_job_id']}")

    return logger


def setup_directories(root_path: Path):
    """创建必要的目录结构"""

    directories = ["models", "roc_curves", "results"]

    for dir in directories:
        (root_path / dir).mkdir(parents=True, exist_ok=True)

    return "目录创建成功"


def initialize_iteration_results_csv(iteration_results_path: Path, logger=None):
    """
    初始化迭代结果CSV文件的表头

    Args:
        config: 配置对象
        logger: 日志记录器
    """

    # 如果文件已存在，不重新创建表头
    if iteration_results_path.exists():
        if logger:
            logger.info(f"迭代结果文件已存在: {iteration_results_path}")
        return

    # 创建目录
    iteration_results_path.parent.mkdir(parents=True, exist_ok=True)

    # 写入表头
    headers = [
                "cal_loop_num",  # 迭代轮次
                "important_count",  # 重要组态数量
                "ml_predicted_count",  # ML预测的高概率组态总数
                "ml_new_count",  # ML新增的组态数（下次计算用）
                "total_original_count",  # 原始CSFs总数
                "current_calculation_count",  # 本轮计算的组态数
                "data_retention_rate",  # 数据留存率（交集/本轮计算）
                "important_retention_rate",  # 重要组态占原始比例
                "ml_retention_rate",  # ML预测组态占原始比例
                "training_time",
                "actual_eval_time",  # 推理时间
                "execution_time",
                "execution_time",  # 总时间（现在与执行时间相同）
                "test_f1",
                "test_roc_auc",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "train_f1",
                "train_roc_auc",
                "train_accuracy",
                "train_precision",
                "train_recall",
                "overfitting_gap",  # 过拟合差距
            ]

    with open(iteration_results_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

    if logger:
        logger.info(f"初始化迭代结果CSV文件: {iteration_results_path}")


def training_data_loader(
        paths_cfg, 
        cal_method: str, 
        use_cpp_descriptor_generator: bool, 
        logger
    ) -> tuple:
    """加载数据文件

    Args:
        paths_cfg: config的cal_path子类，即config.cal_path
        logger: 日志记录器
        use_cpp: 是否使用C++生成的HDF5文件格式

    Returns:
        tuple: (energy_level_data_pd, rmix_file_data, raw_csfs_descriptors, total_csfs_count, cal_csfs_data, caled_csfs_idxs_array)
    """

    # 加载初始 CSFs 描述符文件
    if use_cpp_descriptor_generator:
        # 使用C++生成的HDF5文件
        hdf5_file_path = paths_cfg.full_CSFs_set_file_path.with_suffix(".h5")

        hdf5_data = load_hdf5_descriptors(str(hdf5_file_path))
        raw_csfs_descriptors = hdf5_data["descriptors"]
        logger.info(f"使用C++ HDF5文件加载初始 CSFs 描述符: {hdf5_file_path}")
    else:
        # 使用传统文件格式
        result = load_descriptors_with_multi_block(paths_cfg.full_CSFs_set_file_path, "npy")
        if result is None:
            raise FileNotFoundError(
                f"无法加载初始 CSFs 描述符文件: {paths_cfg.full_CSFs_set_file_path}"
            )
        raw_csfs_descriptors, raw_csfs_idxs = result
        logger.info(f"加载初始 CSFs 描述符文件: {paths_cfg.full_CSFs_set_file_path}")

    raw_csfs_header_file = paths_cfg.full_CSFs_set_header_path 
    csfs_header = rtoml.load(raw_csfs_header_file)
    raw_csfs_num = csfs_header.get('conversion_stats', {}).get('csf_count', 0)
    raw_csfs_desc_count = raw_csfs_descriptors.shape[0]
    if raw_csfs_num == 0:
        error_msg = f"raw_csfs_header_file: {str(raw_csfs_header_file)} 中没有'csf_count'数据请重新运行initial_csfs.py"
        logger.error(error_msg)
        raise ValueError(error_msg)
    elif raw_csfs_desc_count != raw_csfs_num:
        error_msg = f"raw_csfs_header_file: {str(raw_csfs_header_file)} 中'csf_count'数值与描述符文件长度不一致"
        logger.error(error_msg)
        raise ValueError(error_msg)
    else:
        total_csfs_count = raw_csfs_desc_count

    # 加载能级文件
    energy_level_file_path = (
        paths_cfg.cal_loop_path / f"{paths_cfg.loop_file_name}.level"
    )
    energy_level_file_load = LevelsEnergyData.from_filepath(
        str(energy_level_file_path), "LEVEL"
    ) ## !TODO 存在严重问题
    energy_level_data_pd = energy_level_file_load.energy_level_2_pd()
    logger.info(f"加载能级数据: {energy_level_file_path}")

    # 加载本轮计算CSFs文件
    cal_csfs_file_path = paths_cfg.cal_loop_path / f"{paths_cfg.loop_file_name}.c"
    cal_csfs_file_laod = GraspFileLoad.from_filepath(str(cal_csfs_file_path), "CSFs")
    cal_csfs_data = cal_csfs_file_laod.get_csfs_data()
    logger.info(f"加载本轮计算 CSFs 文件: {cal_csfs_file_path}")

    # 加载rmix文件
    # 根据计算轮次确定文件后缀
    if cal_method == "rmcdhf":
        rmix_file_path = paths_cfg.cal_loop_path / f"{paths_cfg.loop_file_name}.m"
    elif cal_method == "rci":
        rmix_file_path = paths_cfg.cal_loop_path / f"{paths_cfg.loop_file_name}.cm"
    else:
        raise ValueError(f"不支持的计算方法: {cal_method}")

    rmix_file_load = GraspFileLoad.from_filepath(str(rmix_file_path), "mix")
    rmix_file_data = rmix_file_load.get_mix_coefficient_data()
    logger.info(f"加载 mix coefficient 文件数据: {rmix_file_path}")

    # 数据一致性检查：确保CSFs数量的两个来源一致
    csfs_count_from_cfile = cal_csfs_data.CSFs_block_length[0]
    csfs_count_from_rmix = rmix_file_data.block_CSFs_nums[0]

    # 加载本轮选择的CSFs的索引文件
    caled_csfs_idxs_file_path = (
        paths_cfg.cal_loop_path / f"{paths_cfg.loop_file_name}_sampled_idxs.npy"
    )
    caled_csfs_idxs_array = np.load(caled_csfs_idxs_file_path)
    logger.info(f"加载本轮选择的 CSFs 的索引文件: {caled_csfs_idxs_file_path}")

    if csfs_count_from_cfile != csfs_count_from_rmix and caled_csfs_idxs_array.shape[0] != csfs_count_from_rmix:
        logger.error(f"本轮计算CSFs数量不一致: cal_csfs_data.CSFs_block_length[0]={csfs_count_from_cfile}, rmix_file_data.block_CSFs_nums[0]={csfs_count_from_rmix}, {caled_csfs_idxs_array.shape[0]=}")
        raise ValueError("本轮计算的CSFs数量数据不一致，请检查数据文件")
    cal_csfs_count = csfs_count_from_rmix

    return (
        raw_csfs_descriptors,
        total_csfs_count,
        energy_level_data_pd,
        rmix_file_data,
        cal_csfs_data,
        caled_csfs_idxs_array,
        cal_csfs_count
    )


def check_configuration_coupling(
        paths_cfg,
        energy_level_data_pd: pd.DataFrame, 
        rmix_file_data: MixCoefficientData,
        spectral_term: list,
        cal_loop_num: int,
        logger
    ):
    """检查组态耦合是否正确"""
    cal_configuration_list = energy_level_data_pd["configuration"].tolist()

    # 统计spectral_term中每个谱项的出现次数
    spectral_term_counts = {}
    for term in spectral_term:
        spectral_term_counts[term] = spectral_term_counts.get(term, 0) + 1

    # 检查每个光谱项的出现次数是否与配置中的要求一致，并记录位置
    spectral_term_positions = []
    all_found_correctly = True

    for term in set(spectral_term):  # 使用set去重，避免重复检查
        expected_count = spectral_term_counts[term]
        actual_count = cal_configuration_list.count(term)

        if actual_count == expected_count:
            # 找到所有出现位置
            positions = [i for i, x in enumerate(cal_configuration_list) if x == term]
            spectral_term_positions.extend(positions)
            if expected_count == 1:
                logger.info(f"光谱项 '{term}' 在位置 {positions[0]} 找到")
            else:
                logger.info(
                    f"光谱项 '{term}' 在位置 {positions} 找到（期望 {expected_count} 次，实际 {actual_count} 次）"
                )
        elif actual_count == 0:
            logger.error(f"光谱项 '{term}' 未找到")
            all_found_correctly = False
        else:
            logger.error(
                f"光谱项 '{term}' 出现 {actual_count} 次，期望 {expected_count} 次"
            )
            all_found_correctly = False

    if all_found_correctly:
        # 按位置排序，保持一致的输出顺序
        spectral_term_positions.sort()
        logger.info(
            f"cal_loop {cal_loop_num} 组态耦合正确，位置索引: {spectral_term_positions}"
        )
        # 选择spectral_term_positions索引对应的行
        selected_energy_data = energy_level_data_pd.iloc[spectral_term_positions]
        # 选择spectral_term_positions索引在rmix文件中对应的ci系数
        correct_levels_ci = rmix_file_data.mix_coefficient_list[0][spectral_term_positions]
        # 保存正确的能级数据为CSV
        correct_levels_csv_path = paths_cfg.cal_loop_path / f'{paths_cfg.loop_file_name}_correct_levels.csv'
        selected_energy_data.to_csv(correct_levels_csv_path, index=False)
        logger.info(f"正确的能级数据已保存到: {correct_levels_csv_path}")

        return True, selected_energy_data, correct_levels_ci
    else:
        error_msg = f"cal_loop {cal_loop_num} 组态耦合错误"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def check_energy_convergence(
        config,
        logger,
        current_energy_data: pd.DataFrame,
        convergence_threshold: float = 0.001,
    ) -> bool:
    """
    检查能量收敛性：比较当前轮与上一轮的能量差异
    ! TODO 这个还没改，上一轮的路径如何输入是个问题

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
        previous_energy_path = (
            config.cal_settings.root_path
            / f"{config.target.conf}_{config.loop_num}"
            / f"{config.target.conf}_{config.cal_settings.cal_loop_num - 1}_correct_levels.csv"
        )

        if not previous_energy_path.exists():
            logger.warning(f"未找到上一轮能量数据: {previous_energy_path}")
            return True

        # 加载上一轮能量数据
        previous_energy_data = pd.read_csv(previous_energy_path)

        # 检查数据一致性
        if len(current_energy_data) != len(previous_energy_data):
            logger.warning(
                f"当前轮与能量数据数量不一致: 当前{len(current_energy_data)} vs 上轮{len(previous_energy_data)}"
            )
            return True

        # 获取能量列
        current_energies = np.asarray(
            current_energy_data["EnergyTotal"], dtype=np.float64
        )
        previous_energies = np.asarray(
            previous_energy_data["EnergyTotal"], dtype=np.float64
        )

        # 计算绝对能量差异
        energy_diffs = np.abs(current_energies - previous_energies)
        max_diff = np.max(energy_diffs)

        # 检查是否超过阈值
        if max_diff > convergence_threshold:
            max_diff_idx = np.argmax(energy_diffs)
            logger.warning(
                f"检测到能量不收敛: 第{max_diff_idx + 1}个能级能量差异{max_diff:.6f} (阈值: {convergence_threshold:.6f})"
            )
            logger.warning(f"当前能量: {current_energies[max_diff_idx]:.7f}")
            logger.warning(f"上轮能量: {previous_energies[max_diff_idx]:.7f}")
            return False

        logger.info("能量收敛检查通过，继续计算")
        return True

    except Exception as e:
        logger.error(f"能量收敛检查过程中发生错误: {str(e)}")
        return True  # 发生错误时继续计算，避免阻塞


def evaluate_calculation_convergence(
        config, 
        logger, 
        cal_loop_csfs_count=None
    ):
    """
    检查GRASP计算的收敛性

    使用最近三次计算结果的能级数据和组态数量数据，通过计算：
    1. 每个能级的标准差
    2. 组态数量的相对标准差
    来判定收敛。

    Args:
        config: 配置对象
        logger: 日志记录器
        cal_loop_csfs_count: 当前轮的CSFs数量（可选，如果提供则不从文件读取）

    Returns:
        bool: True表示继续计算，False表示已收敛停止计算
    """
    try:
        # 读取最近3次计算的能级数据
        energy_data_list = []

        for i in range(3):
            loop_num = config.cal_settings.cal_loop_num - 2 + i  # 前3次：当前-2, 当前-1, 当前
            csv_path = (
                config.cal_settings.root_path
                / f"{config.target.conf}_{loop_num}"
                / f"{config.target.conf}_{loop_num}_correct_levels.csv"
            )

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
        iteration_results_path = config.cal_settings.root_path / "results" / "iteration_results.csv"
        csfs_num = []  # 存储每轮的组态数量

        # 如果提供了当前轮的CSFs数量，则优先使用
        if cal_loop_csfs_count is not None:
            # 从iteration_results.csv读取前两轮的数据，使用传入的当前轮数据
            if iteration_results_path.exists():
                try:
                    iteration_df = pd.read_csv(iteration_results_path)
                    # 获取前两轮的组态数量
                    for i in range(2):  # 只读取前两轮
                        loop_num = config.cal_settings.cal_loop_num - 2 + i
                        # 查找对应轮次的数据
                        loop_data = iteration_df[iteration_df["iteration"] == loop_num]
                        if not loop_data.empty:
                            current_count = loop_data["current_calculation_count"].iloc[
                                0
                            ]
                            csfs_num.append(current_count)
                            logger.info(f"读取第{loop_num}轮组态数量: {current_count}")
                        else:
                            logger.warning(
                                f"在iteration_results.csv中未找到第{loop_num}轮的数据"
                            )
                            return True  # 数据不完整，继续计算

                    # 添加当前轮的CSFs数量
                    csfs_num.append(cal_loop_csfs_count)
                    logger.info(
                        f"读取第{config.cal_settings.cal_loop_num}轮组态数量: {cal_loop_csfs_count}"
                    )

                    if len(csfs_num) < 3:
                        logger.warning("无法读取完整的3轮组态数量数据，继续计算")
                        return True

                except Exception as e:
                    logger.warning(f"读取iteration_results.csv文件出错: {e}")
                    return True
            else:
                logger.warning(
                    f"未找到iteration_results.csv文件: {iteration_results_path}"
                )
                return True
        else:
            # 原有逻辑：从iteration_results.csv读取所有3轮数据
            if iteration_results_path.exists():
                try:
                    iteration_df = pd.read_csv(iteration_results_path)
                    # 获取最近3轮的组态数量
                    for i in range(3):
                        loop_num = config.cal_settings.cal_loop_num - 2 + i
                        # 查找对应轮次的数据
                        loop_data = iteration_df[iteration_df["iteration"] == loop_num]
                        if not loop_data.empty:
                            current_count = loop_data["current_calculation_count"].iloc[
                                0
                            ]
                            csfs_num.append(current_count)
                            logger.info(f"读取第{loop_num}轮组态数量: {current_count}")
                        else:
                            logger.warning(
                                f"在iteration_results.csv中未找到第{loop_num}轮的数据"
                            )
                            return True  # 数据不完整，继续计算

                    if len(csfs_num) < 3:
                        logger.warning("无法读取完整的3轮组态数量数据，继续计算")
                        return True

                except Exception as e:
                    logger.warning(f"读取iteration_results.csv文件出错: {e}")
                    return True
            else:
                logger.warning(
                    f"未找到iteration_results.csv文件: {iteration_results_path}"
                )
                return True

        # === 1. 能级标准差计算 ===
        # 获取所有configuration
        configurations = energy_data_list[0]["configuration"].tolist()

        # 存储每个能级的标准差
        std_deviations = []

        for config_name in configurations:
            # 获取该configuration在3轮计算中的能级值
            energy_values = []
            for df in energy_data_list:
                if config_name in df["configuration"].values:
                    energy = df[df["configuration"] == config_name]["EnergyTotal"].iloc[
                        0
                    ]
                    energy_values.append(energy)
                else:
                    logger.warning(
                        f"在第{len(energy_values) + 1}轮数据中未找到configuration: {config_name}"
                    )
                    return True  # 数据不完整，继续计算

            if len(energy_values) == 3:
                # 计算标准差
                energy_std = np.std(energy_values)
                std_deviations.append(energy_std)

                logger.debug(
                    f"Configuration {config_name}: "
                    f"能级值={energy_values}, "
                    f"标准差={energy_std:.5e}"
                )

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
        energy_std_threshold = getattr(
            config, "energy_std_threshold", 1e-5
        )  # 能级标准差阈值
        csfs_num_relative_std_threshold = getattr(
            config, "csfs_num_relative_std_threshold", 1e-3
        )  # 组态数量相对标准差阈值（5%）

        logger.info(f"收敛性统计:")
        logger.info(f"  最近3轮组态数量: {csfs_num}")
        logger.info(f"  组态数量平均值: {csfs_num_mean:.1f}")
        logger.info(f"  组态数量标准差: {csfs_num_std:.2f}")
        logger.info(
            f"  组态数量相对标准差: {csfs_num_relative_std:.4f} (阈值: {csfs_num_relative_std_threshold:.4f})"
        )
        logger.info(
            f"  能级平均标准差: {avg_energy_std:.5e} (阈值: {energy_std_threshold:.5e})"
        )
        logger.info(f"  能级标准差收敛: {avg_energy_std < energy_std_threshold}")
        logger.info(
            f"  组态数量相对标准差收敛: {csfs_num_relative_std < csfs_num_relative_std_threshold}"
        )

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


def merge_historical_ci_data(
        previous_idxs_ci_dict, 
        current_idxs_ci_dict, 
        logger
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    合并历史CI系数数据，取索引并集并比较共有索引的CI系数大小

    合并规则：
    1. 取idxs的并集
    2. 两个字典idxs中的交集对应的ci_squared取较大值

    Args:
        previous_idxs_ci_dict: 历史CI数据字典，格式为 
        {"idxs": np.ndarray[...], "ci_squared": np.ndarray[...]}
        current_idxs_ci_dict: 当前CI数据字典，格式为 
        {"idxs": np.ndarray[...], "ci_squared": np.ndarray[...]}
        logger: 日志记录器

    Returns:
        Tuple[np.ndarray, np.ndarray]: 合并后的索引数组和CI系数平方数组
    """

    # 获取历史数据和当前数据
    previous_idxs = np.array(previous_idxs_ci_dict["idxs"])
    previous_ci_squared = np.array(previous_idxs_ci_dict["ci_squared"])

    current_idxs = np.array(current_idxs_ci_dict["idxs"])
    current_ci_squared = np.array(current_idxs_ci_dict["ci_squared"])

    # 创建索引到CI系数的映射
    previous_dict = dict(zip(previous_idxs, previous_ci_squared))
    current_dict = dict(zip(current_idxs, current_ci_squared))

    # 获取索引的并集
    all_idxs = set(previous_idxs) | set(current_idxs)

    # 合并CI系数：对于交集索引，取较大值
    merged_idxs = []
    merged_ci_squared = []

    for idx in sorted(all_idxs):
        merged_idxs.append(idx)

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

    merged_idxs = np.array(merged_idxs)
    merged_ci_squared = np.array(merged_ci_squared)

    logger.info(f"历史数据合并统计:")
    logger.info(f"历史数据CSFs数量: {len(previous_idxs)}")
    logger.info(f"当前数据CSFs数量: {len(current_idxs)}")
    logger.info(f"合并后CSFs数量: {len(merged_idxs)}")
    logger.info(f"新增CSFs数量: {len(all_idxs - set(previous_idxs))}")
    logger.info(f"重复CSFs数量: {len(set(previous_idxs) & set(current_idxs))}")

    return merged_idxs, merged_ci_squared


def generate_train_csfs_descriptors(
        config, 
        raw_csfs_descriptors: np.ndarray, 
        logger
    ) -> np.ndarray:
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
    current_idxs_ci_path = (
        config.cal_path.results_path
        / f"{config.cal_path.loop_file_name}_ci_squared.pkl"
    )
    if not current_idxs_ci_path.exists():
        raise FileNotFoundError(f"当前轮次CSF索引文件不存在: {current_idxs_ci_path}")

    current_idxs_ci_dict = pkl_loader(current_idxs_ci_path)
    try:
        assert "idxs" in current_idxs_ci_dict, "缺少子键 idxs"
        assert "ci_squared" in current_idxs_ci_dict, "缺少子键 ci_squared"
        assert current_idxs_ci_dict["idxs"] is not None, "idxs 值为空"
        assert current_idxs_ci_dict["ci_squared"] is not None, "ci_squared 值为空"
        
        logger.info("所有键值验证通过")

    except AssertionError as e:
        logger.error(f"验证失败: {e}")

    current_sampled_idxs = np.array(current_idxs_ci_dict["idxs"])

    if config.cal_settings.cal_loop_num > 1:
        # 读取历次迭代保存的CI系数数据
        previous_idxs_ci_path = config.cal_path.previous_idxs_ci_path

        if not previous_idxs_ci_path.exists():
            raise FileNotFoundError(
                f"当前轮次CSF索引文件不存在: {previous_idxs_ci_path}"
            )

        previous_idxs_ci_dict = pkl_loader(previous_idxs_ci_path)
        try:
            assert "idxs" in previous_idxs_ci_dict, "缺少子键 idxs"
            assert "ci_squared" in previous_idxs_ci_dict, "缺少子键 ci_squared"
            assert previous_idxs_ci_dict["idxs"] is not None, "idxs 值为空"
            assert previous_idxs_ci_dict["ci_squared"] is not None, (
                "ci_squared 值为空"
            )

            logger.info("开始读取历次迭代的CI系数数据")

        except AssertionError as e:
            logger.error(f"验证失败: {e}")

        accumulated_idxs, accumulated_ci_squared = merge_historical_ci_data(
            previous_idxs_ci_dict, current_idxs_ci_dict, logger
        )

        logger.info(f"训练数据")
        logger.info(f"CSF总数: {len(accumulated_idxs)}")
        logger.info(f"当前轮次CSF数: {len(current_sampled_idxs)}")

    elif config.cal_settings.cal_loop_num == 1:
        accumulated_idxs = current_idxs_ci_dict["idxs"]
        accumulated_ci_squared = current_idxs_ci_dict["ci_squared"]
        logger.info(f"训练数据")
        print(accumulated_idxs)
        logger.info(f"CSF总数: {len(accumulated_idxs)}")
        logger.info(f"当前轮次CSF数: {len(current_sampled_idxs)}")
    else:
        logger.error(f"{config.cal_settings.cal_loop_num=} error")
        raise ValueError(f"Invalid cal_loop_num: {config.cal_settings.cal_loop_num}")

    # 初始化变量
    cutoff_value = np.float64(config.cal_settings.cutoff_value)

    sampled_csfs_descriptors = raw_csfs_descriptors[accumulated_idxs]
    important_csfs_mask = accumulated_ci_squared >= cutoff_value

    logger.info(f"生成完整训练数据: {sampled_csfs_descriptors.shape[0]} 个CSF")
    logger.info(
        f"正样本数量: {np.sum(important_csfs_mask)} (占比: {np.sum(important_csfs_mask) / len(important_csfs_mask):.4f})"
    )

    # 返回完整的训练数据（类似旧版ann3_proba.py的处理方式）
    caled_csfs_descriptors = np.column_stack(
        [sampled_csfs_descriptors, important_csfs_mask]
    )

    accumulated_ci_data = {
            "idxs": accumulated_idxs,  # CSF索引（对应总池）
            "ci_squared": accumulated_ci_squared,  # 对应的CI系数平方（正确能级 × 当前计算CSF）
    }

    accumulated_ci_path = (
        config.cal_path.results_path / f"{config.target.conf}_previous_ci_squared.pkl"
    )
    pkl_storage(accumulated_ci_data, accumulated_ci_path)
    logger.info(
        f"保存累积CI系数数据: {accumulated_ci_path} (包含{len(accumulated_idxs)}个CSFs)"
    )

    # 保存描述符文件
    save_descriptors(
        caled_csfs_descriptors,
        f"{config.cal_path.cal_loop_path}/{config.cal_path.loop_file_name}_full",
        "npy",
    )
    logger.info(
        f"保存完整历史数据并集描述符文件: {config.cal_path.cal_loop_path}/{config.cal_path.loop_file_name}_full.npy"
    )

    logger.info(f"CSFs描述符标签生成完成")
    logger.info(
        f"正样本数量: {np.sum(important_csfs_mask)} (在正确能级位置混合系数 ≥ {cutoff_value})"
    )
    logger.info(
        f"负样本总数量: {len(important_csfs_mask) - np.sum(important_csfs_mask)}"
    )
    logger.info(
        f"正样本比例: {np.sum(important_csfs_mask) / len(important_csfs_mask):.4f}"
    )

    return caled_csfs_descriptors

def get_stay_descriptors(
            raw_csfs_descriptors: np.ndarray, 
            sampled_csfs_idxs_array: np.ndarray
        ) -> np.ndarray:
    """
    找出不在sampled_csfs_idxs_array索引中的描述符

    Args:
        raw_csfs_descriptors: 原始CSFs描述符数组
        sampled_csfs_idxs_array: 已选择的CSFs索引

    Returns:
        np.ndarray: 不在sampled_csfs_idxs_array中的描述符数组
    """
    # 验证字典并安全获取所有已选择的索引
    if sampled_csfs_idxs_array.size == 0:
        raise ValueError("sampled_csfs_idxs_array为空，无法获取选中的CSFs索引")
    
    mask = np.ones(raw_csfs_descriptors.shape[0], dtype=bool)
    
    # 2. 把要剔除的行的位置设为 False
    mask[sampled_csfs_idxs_array] = False
    
    # 3. 数组切片：当你对二维数组使用一维布尔掩码时，NumPy 默认就是筛选“行”
    # 这样会自动保留 mask 为 True 的整行数据
    return raw_csfs_descriptors[mask]

