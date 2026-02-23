# -*- encoding: utf-8 -*-
"""
@Id :machine_learning_initialization.py
@date :2025/06/09 15:13:58
@author :YenochQin (秦毅)
"""

from graspkit.data_IO.ml_cal_config_module import MLCalConfig


import logging
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import rtoml

from ..data_IO import (
    load_config,
    pkl_loader,
    pkl_storage,
    save_descriptors,
    scan_descriptors_polars,
)
from ..data_IO.loaders.energy_file_loader import EnergyFileLoader
from ..data_IO.loaders.mix_coef_loader import MixCoefLoader
from ..utils.data_modules import MixCoefficientData
from ..utils.environment_config import get_environment_config


def setup_config(config_path: str | Path) -> MLCalConfig:
    """
    初始化机器学习配置并设置相关路径

    Args:
        config_path (str | Path): 配置文件的路径

    Returns:
        config: 配置对象，包含所有初始化后的路径和参数

    该函数执行以下步骤：
    1. 从指定路径加载配置文件
    2. 调用配置对象的 setup_paths() 方法设置所有相关文件路径
    3. 返回完整的配置对象
    """
    # 从配置文件路径加载配置信息
    config = load_config(config_path)

    # 调用配置对象的 setup_paths() 方法设置所有相关路径
    config.setup_paths()

    return config


def setup_directories(root_path: Path):
    """创建必要的目录结构"""

    directories = ["models", "roc_curves", "results"]

    for dir in directories:
        (root_path / dir).mkdir(parents=True, exist_ok=True)

    return "目录创建成功"


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
    handlers.append(logging.FileHandler(log_dir / "ml_training.log", encoding="utf-8"))

    # 在调试模式下添加控制台输出
    if not env_config.is_production_mode:
        handlers.append(logging.StreamHandler())

    # 配置日志
    logging.basicConfig(
        level=log_level,
        format=log_config["format"],
        datefmt="%m-%d %H:%M:%S",
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


def training_data_loader(
    paths_cfg, cal_method: str, logger: logging.Logger
) -> tuple[pl.LazyFrame, int, pl.DataFrame, MixCoefficientData, np.ndarray, int]:
    """加载数据文件

    Args:
        paths_cfg: config的cal_path子类，即config.cal_path
        cal_method: 计算方法 ("rmcdhf" 或 "rci")
        descriptor_file_type:
            h5: 是否使用C++生成的HDF5文件格式
            parquet: 是否使用Rust生成的parquet文件（惰性加载，只加载需要的索引）
        logger: 日志记录器

    Returns:
        tuple: (
            raw_csfs_descriptors,
            total_csfs_count,
            energy_level_data,
            rmix_file_data,
            caled_csfs_idxs_array,
            cal_csfs_count,
        )
    """

    # 加载初始 CSFs 描述符文件
    # 使用 rcsfs 生成的 parquet 文件
    raw_desc_file_path = paths_cfg.full_CSFs_set_desc_path.with_suffix(".parquet")

    raw_csfs_descriptors, parquet_meta_data = scan_descriptors_polars(
        raw_desc_file_path
    )
    logger.info(f"加载 parquet raw_CSFs 描述符: {raw_desc_file_path}")
    raw_csfs_desc_count = parquet_meta_data["n_rows"]

    raw_csfs_header_file = paths_cfg.full_CSFs_set_header_path
    csfs_header = rtoml.load(raw_csfs_header_file)
    raw_csfs_num = csfs_header.get("conversion_stats", {}).get("csf_count", 0)

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
    energy_level_file_load = EnergyFileLoader(energy_level_file_path)
    energy_level_data = energy_level_file_load.load()
    logger.info(f"加载能级数据: {energy_level_file_path}")

    # 加载rmix文件
    # 根据计算轮次确定文件后缀
    if cal_method == "rmcdhf":
        rmix_file_path = paths_cfg.cal_loop_path / f"{paths_cfg.loop_file_name}.m"
    elif cal_method == "rci":
        rmix_file_path = paths_cfg.cal_loop_path / f"{paths_cfg.loop_file_name}.cm"
    else:
        raise ValueError(f"不支持的计算方法: {cal_method}")

    rmix_file_load = MixCoefLoader(rmix_file_path)
    rmix_file_data = rmix_file_load.load()
    logger.info(f"加载 mix coefficient 文件数据: {rmix_file_path}")

    csfs_count_from_rmix = rmix_file_data.block_CSFs_nums[0]

    # 加载本轮选择的CSFs的索引文件
    caled_csfs_idxs_file_path = (
        paths_cfg.cal_loop_path / f"{paths_cfg.loop_file_name}_sampled_idxs.npy"
    )
    caled_csfs_idxs_array = np.load(caled_csfs_idxs_file_path)
    logger.info(f"加载本轮选择的 CSFs 的索引文件: {caled_csfs_idxs_file_path}")

    if caled_csfs_idxs_array.shape[0] != csfs_count_from_rmix:
        logger.error(
            f"rmix_file_data.block_CSFs_nums[0]={csfs_count_from_rmix}, {caled_csfs_idxs_array.shape[0]=}"
        )
        raise ValueError("本轮计算的CSFs数量数据不一致，请检查数据文件")
    cal_csfs_count = csfs_count_from_rmix

    return (
        raw_csfs_descriptors,
        total_csfs_count,
        energy_level_data,
        rmix_file_data,
        caled_csfs_idxs_array,
        cal_csfs_count,
    )


def check_configuration_coupling(
    paths_cfg,
    energy_level_data: pl.DataFrame,
    rmix_file_data: MixCoefficientData,
    spectral_term: list,
    cal_loop_num: int,
    logger: logging.Logger,
):
    """检查组态耦合是否正确

    优化版本：一次遍历完成计数和位置记录，时间复杂度从 O(n*m) 降至 O(n)
    """
    cal_configuration_list = energy_level_data["configuration"].to_list()

    # 优化1: 一次遍历同时构建计数和位置映射
    term_positions = {}
    actual_counts = Counter()

    for idx, term in enumerate(cal_configuration_list):
        if term in spectral_term:
            actual_counts[term] += 1
            term_positions.setdefault(term, []).append(idx)

    # 优化2: 使用Counter构建期望计数
    expected_counts = Counter(spectral_term)

    # 检查并收集位置
    spectral_term_positions = []
    errors = []

    for term in expected_counts:
        expected = expected_counts[term]
        actual = actual_counts.get(term, 0)
        positions = term_positions.get(term, [])

        if actual == expected:
            spectral_term_positions.extend(positions)
            if expected == 1:
                logger.info(f"光谱项 '{term}' 在位置 {positions[0]} 找到")
            else:
                logger.info(
                    f"光谱项 '{term}' 在位置 {positions} 找到（期望{expected}次，实际{actual}次）"
                )
        elif actual == 0:
            errors.append(f"光谱项 '{term}' 未找到")
        else:
            errors.append(f"光谱项 '{term}' 出现{actual}次，期望{expected}次")

    # 优化3: 提前返回模式，减少嵌套
    if errors:
        for err in errors:
            logger.error(err)
        error_msg = f"cal_loop {cal_loop_num} 组态耦合错误"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # 成功路径
    spectral_term_positions.sort()
    logger.info(
        f"cal_loop {cal_loop_num} 组态耦合正确，位置索引: {spectral_term_positions}"
    )

    selected_energy_data = energy_level_data[spectral_term_positions]
    correct_levels_ci = rmix_file_data.mix_coefficient_list[0][spectral_term_positions]
    correct_levels_csv_path = (
        paths_cfg.cal_loop_path / f"{paths_cfg.loop_file_name}_correct_levels.csv"
    )
    selected_energy_data.write_csv(correct_levels_csv_path)
    logger.info(f"正确的能级数据已保存到: {correct_levels_csv_path}")

    return True, selected_energy_data, correct_levels_ci


def check_energy_convergence(
    config,
    logger: logging.Logger,
    current_energy_data: pl.DataFrame,
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
        previous_energy_data = pl.read_csv(previous_energy_path)

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
    config, logger: logging.Logger, cal_loop_csfs_count: int
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
        cal_loop_csfs_count: 当前轮的CSFs数量

    Returns:
        bool: True表示继续计算, False表示已收敛停止计算
    """

    # 读取最近3次计算的能级数据
    energy_data_list = []

    for i in range(3):
        loop_num = (
            config.cal_settings.cal_loop_num - 2 + i
        )  # 前3次：当前-2, 当前-1, 当前
        csv_path = (
            config.cal_settings.root_path
            / f"{config.target.conf}_{loop_num}"
            / f"{config.target.conf}_{loop_num}_correct_levels.csv"
        )

        if csv_path.exists():
            df = pl.read_csv(csv_path)
            energy_data_list.append(df)
            logger.info(f"读取第{loop_num}轮能级数据: {csv_path}")
        else:
            logger.warning(f"未找到第{loop_num}轮能级数据文件: {csv_path}")
            return True  # 文件不存在，继续计算

    if len(energy_data_list) < 3:
        logger.warning("无法读取完整的3轮能级数据，继续计算")
        return True

    # 读取组态数量数据
    csfs_num = []  # 存储每轮的组态数量
    iteration_df = pl.read_csv(config.cal_path.iteration_results)
    # 获取前两轮的组态数量
    for i in range(2):  # 只读取前两轮
        loop_num = config.cal_settings.cal_loop_num - 2 + i
        # 查找对应轮次的数据
        loop_data = iteration_df.filter(pl.col("cal_loop_num") == loop_num)
        if loop_data.height > 0:
            current_count = loop_data["current_calculation_count"][0]
            csfs_num.append(current_count)
            logger.info(f"读取第{loop_num}轮组态数量: {current_count}")
        else:
            logger.warning(f"在iteration_results.csv中未找到第{loop_num}轮的数据")
            return True  # 数据不完整，继续计算
    # 添加当前轮的CSFs数量
    csfs_num.append(cal_loop_csfs_count)
    logger.info(
        f"读取第{config.cal_settings.cal_loop_num}轮组态数量: {cal_loop_csfs_count}"
    )
    # === 1. 能级标准差计算 ===
    # 获取所有configuration
    configurations = energy_data_list[0]["configuration"].to_list()

    # 存储每个能级的标准差
    std_deviations = []

    for level_cfg in configurations:
        # 获取该configuration在3轮计算中的能级值
        energy_values = []
        for df in energy_data_list:
            if level_cfg in df["configuration"].to_list():
                energy = df.filter(pl.col("configuration") == level_cfg)["EnergyTotal"][
                    0
                ]
                energy_values.append(energy)
            else:
                logger.warning(
                    f"在第{len(energy_values) + 1}轮数据中未找到configuration: {level_cfg}"
                )
                return True  # 数据不完整，继续计算

        if len(energy_values) == 3:
            # 计算标准差
            energy_std = np.std(energy_values)
            std_deviations.append(energy_std)

            logger.info(
                f"Configuration {level_cfg}: "
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
        config.cal_settings, "energy_std_threshold", 1e-5
    )  # 能级标准差阈值
    csfs_num_relative_std_threshold = getattr(
        config.cal_settings, "csfs_num_relative_std_threshold", 1e-3
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


def merge_historical_ci_data(
    previous_idxs_ci_dict: dict, current_idxs_ci_dict: dict, logger: logging.Logger
) -> tuple[np.ndarray, np.ndarray]:
    """
    合并历史CI系数数据，取索引并集并以当前数据更新共有索引的CI系数

    合并规则：
    1. 取idxs的并集
    2. 两个字典idxs中的交集对应的ci_squared使用当前数据中的值（更新制）

    Args:
        previous_idxs_ci_dict: 历史CI数据字典，格式为
        {"idxs": np.ndarray[...], "ci_squared": np.ndarray[...]}
        current_idxs_ci_dict: 当前CI数据字典，格式为
        {"idxs": np.ndarray[...], "ci_squared": np.ndarray[...]}
        logger: 日志记录器

    Returns:
        tuple[np.ndarray, np.ndarray]: 合并后的索引数组和CI系数平方数组
    """

    # 获取历史数据和当前数据
    previous_idxs = np.array(previous_idxs_ci_dict["idxs"])
    previous_ci_squared = np.array(previous_idxs_ci_dict["ci_squared"])

    current_idxs = np.array(current_idxs_ci_dict["idxs"])
    current_ci_squared = np.array(current_idxs_ci_dict["ci_squared"])

    # 创建索引到CI系数的映射
    # 注意：ci_squared的shape是(n_correct_levels, n_csfs)，需要转置使每个CSF对应一行
    previous_ci_squared_t = previous_ci_squared.T
    current_ci_squared_t = current_ci_squared.T
    previous_dict = dict(zip(previous_idxs, previous_ci_squared_t))
    current_dict = dict(zip(current_idxs, current_ci_squared_t))

    # 获取索引的并集
    all_idxs = set(previous_idxs) | set(current_idxs)

    # 合并CI系数：对于交集索引，使用当前数据中的值
    merged_idxs = []
    merged_ci_squared = []

    for idx in sorted(all_idxs):
        merged_idxs.append(idx)

        # 如果索引在两个字典中都存在，使用当前数据的CI系数
        if idx in previous_dict and idx in current_dict:
            merged_ci_squared.append(current_dict[idx])
        # 如果只在历史数据中存在
        elif idx in previous_dict:
            merged_ci_squared.append(previous_dict[idx])
        # 如果只在当前数据中存在
        else:  # idx in current_dict
            merged_ci_squared.append(current_dict[idx])

    merged_idxs = np.array(merged_idxs)
    # 转置回原始shape: (n_correct_levels, n_csfs)
    merged_ci_squared = np.array(merged_ci_squared).T

    logger.info(f"历史数据合并统计:")
    logger.info(f"历史数据CSFs数量: {len(previous_idxs)}")
    logger.info(f"当前数据CSFs数量: {len(current_idxs)}")
    logger.info(f"合并后CSFs数量: {len(merged_idxs)}")
    logger.info(f"新增CSFs数量: {len(all_idxs - set(previous_idxs))}")
    logger.info(f"重复CSFs数量: {len(set(previous_idxs) & set(current_idxs))}")

    return merged_idxs, merged_ci_squared


def ci_idx_data_processor(
    correct_levels_ci: np.ndarray,
    caled_csfs_idxs_array: np.ndarray,
    config,
    logger: logging.Logger,
) -> np.ndarray:
    """
    处理CI系数数据，并与历史数据合并后保存

    优化说明：
    - 本轮计算生成的数据直接与历史数据合并
    - 合并后的数据保存到 previous_idxs_ci_path 供下次使用
    - 避免在 generate_train_csfs_descriptors 中的冗余读写操作

    Args:
        correct_levels_ci: 正确能级位置的CI系数
        caled_csfs_idxs_array: 当前计算的CSF索引数组
        config: 配置对象
        logger: 日志记录器

    Returns:
        np.ndarray: CI系数平方数组
    """

    # 保存正确能级位置的CI系数平方对应CSF总池索引（用于历史数据累积）
    correct_levels_ci_2d = np.atleast_2d(correct_levels_ci)
    correct_levels_ci_squared = (
        correct_levels_ci_2d**2
    )  # shape: (n_correct_levels, n_current_csfs)

    current_ci_squared_data_dict = {
        "idxs": caled_csfs_idxs_array,  # CSF索引（对应总池）
        "ci_squared": correct_levels_ci_squared,  # 对应的CI系数平方（正确能级 × 当前计算CSF）
    }

    accumulated_idxs_ci_path = config.cal_path.accumulated_idxs_ci_path
    # 如果是第二轮及之后，读取历史数据并合并
    if config.cal_settings.cal_loop_num > 1:
        if accumulated_idxs_ci_path.exists():
            logger.info("读取历史CI系数数据并进行合并")
            previous_idxs_ci_dict = pkl_loader(accumulated_idxs_ci_path)
            accumulated_idxs, accumulated_ci_squared = merge_historical_ci_data(
                previous_idxs_ci_dict, current_ci_squared_data_dict, logger
            )
            accumulated_ci_data = {
                "idxs": accumulated_idxs,
                "ci_squared": accumulated_ci_squared,
            }
        else:
            logger.warning(
                f"历史数据文件不存在: {accumulated_idxs_ci_path}，仅使用当前轮次数据"
            )
            accumulated_ci_data = current_ci_squared_data_dict
    else:
        # 第一轮直接使用当前数据
        accumulated_ci_data = current_ci_squared_data_dict

    # 保存合并后的累积数据到 accumulated_idxs_ci_path（供下次计算使用）

    pkl_storage(accumulated_ci_data, accumulated_idxs_ci_path)
    logger.info(
        f"保存累积CI系数数据: {accumulated_idxs_ci_path} "
        f"(包含{len(accumulated_ci_data['idxs'])}个CSFs)"
    )

    return correct_levels_ci_squared


def generate_train_csfs_descriptors(
    config, raw_csfs_descriptors: np.ndarray, logger: logging.Logger
) -> np.ndarray:
    """
    生成用于机器学习训练的CSFs描述符数据

    优化说明：
    - 数据合并逻辑已移至 ci_idx_data_processor 函数
    - 本函数直接读取合并后的累积数据，避免冗余读写操作
    - previous_idxs_ci_path 包含所有历史轮次的数据并集

    Args:
        config: 配置对象
        raw_csfs_descriptors: 原始CSFs描述符数组
        logger: 日志记录器

    Returns:
        np.ndarray: 包含描述符和标签的训练数据
    """

    # 直接读取合并后的累积CI系数数据
    # 该数据已由 ci_idx_data_processor 函数预先合并并保存
    logger.info("加载累积的CSF索引和CI系数数据（已包含所有历史轮次）")
    accumulated_idxs_ci_path = config.cal_path.accumulated_idxs_ci_path

    if config.cal_settings.cal_loop_num > 1 and not accumulated_idxs_ci_path.exists():
        raise FileNotFoundError(f"累积CI系数文件不存在: {accumulated_idxs_ci_path}")

    accumulated_ci_data = pkl_loader(accumulated_idxs_ci_path)
    try:
        assert "idxs" in accumulated_ci_data, "缺少子键 idxs"
        assert "ci_squared" in accumulated_ci_data, "缺少子键 ci_squared"
        assert accumulated_ci_data["idxs"] is not None, "idxs 值为空"
        assert accumulated_ci_data["ci_squared"] is not None, "ci_squared 值为空"
        logger.info("所有键值验证通过")
    except AssertionError as e:
        logger.error(f"验证失败: {e}")
        raise

    accumulated_idxs = accumulated_ci_data["idxs"]
    accumulated_ci_squared = accumulated_ci_data["ci_squared"]

    logger.info(f"训练数据统计")
    logger.info(f"CSF总数（累积）: {len(accumulated_idxs)}")
    logger.info(f"当前轮次: {config.cal_settings.cal_loop_num}")

    # 初始化变量
    cutoff_value = np.float64(config.cal_settings.cutoff_value)

    sampled_csfs_descriptors = raw_csfs_descriptors[accumulated_idxs]
    # 转置以匹配描述符的行维度: (n_current_csfs, n_correct_levels)
    important_csfs_mask = (accumulated_ci_squared >= cutoff_value).T

    logger.info(f"生成完整训练数据: {sampled_csfs_descriptors.shape[0]} 个CSF")
    logger.info(
        f"正样本数量: {np.sum(important_csfs_mask)} (占比: {np.sum(important_csfs_mask) / len(important_csfs_mask):.4f})"
    )

    # 返回完整的训练数据（类似旧版ann3_proba.py的处理方式）
    caled_csfs_descriptors = np.column_stack(
        [sampled_csfs_descriptors, important_csfs_mask]
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
    raw_csfs_descriptors: np.ndarray, sampled_csfs_idxs_array: np.ndarray
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
