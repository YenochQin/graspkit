# -*- encoding: utf-8 -*-
"""
@Id :machine_learning_initialization.py
@date :2025/06/09 15:13:58
@author :YenochQin (秦毅)
"""

import logging
from collections import Counter
from pathlib import Path

import numpy as np
from numpy._typing._array_like import NDArray

import polars as pl
import rtoml

from ..data_IO import (
    CalPath,
    EnergyFileLoader,
    MLCalConfig,
    MixCoefLoader,
    csfs_idxs_ci_loader,
    csfs_idxs_ci_storage,
    save_descriptors,
    scan_descriptors_polars,
)
from ..utils import (
    MixCoefficientData,
    get_environment_config,
)


def setup_directories(root_path: Path):
    """创建必要的目录结构"""

    directories = ["models", "roc_curves", "results"]

    for dir in directories:
        (root_path / dir).mkdir(parents=True, exist_ok=True)


def setup_logging(log_dir: Path):
    """配置日志系统，支持环境感知"""
    env_config = get_environment_config()
    log_config = env_config.get_logging_config()

    # 创建日志目录
    log_dir.mkdir(exist_ok=True)

    # 配置日志级别
    log_level = getattr(logging, log_config["level"])

    # 创建处理器列表
    handlers: list[logging.Handler] = []
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
    paths_cfg: CalPath,
    cal_method: str,
    logger: logging.Logger
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
    paths_cfg: CalPath,
    energy_level_data: pl.DataFrame,
    rmix_file_data: MixCoefficientData,
    spectral_term: list[str],
    cal_loop_num: int,
    logger: logging.Logger,
):
    """检查组态耦合是否正确

    优化版本：一次遍历完成计数和位置记录，时间复杂度从 O(n*m) 降至 O(n)
    """
    cal_configuration_list: list[str] = energy_level_data["configuration"].to_list()

    # 优化1: 一次遍历同时构建计数和位置映射
    term_positions: dict[str, list[int]] = {}
    actual_counts: Counter[str] = Counter()

    for idx, term in enumerate(cal_configuration_list):
        if term in spectral_term:
            actual_counts[term] += 1
            term_positions.setdefault(term, []).append(idx)

    # 优化2: 使用Counter构建期望计数
    expected_counts = Counter(spectral_term)

    # 检查并收集位置
    spectral_term_positions: list[int] = []
    errors: list[str] = []

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
    config: MLCalConfig,
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
            / f"{config.target.conf}_{config.cal_settings.cal_loop_num - 1}"
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
    config: MLCalConfig,
    logger: logging.Logger,
    cal_loop_csfs_count: int,
) -> bool:
    """
    检查GRASP计算的收敛性

    使用最近三次计算结果的能级数据和组态数量数据，通过计算：
    1. 三轮共有能级的跨轮次标准差均值（能级收敛）
    2. 组态数量的相对标准差（CSF数量稳定性）
    来判定收敛。两个条件同时满足才停止计算。

    Args:
        config: 配置对象
        logger: 日志记录器
        cal_loop_csfs_count: 当前轮的CSFs数量

    Returns:
        bool: True 表示继续计算，False 表示已收敛停止计算
    """
    try:
        current_loop = config.cal_settings.cal_loop_num

        # === 1. 加载最近三轮能级数据，取列 configuration 和 EnergyTotal ===
        energy_frames: list[pl.DataFrame] = []
        for i, loop_num in enumerate(range(current_loop - 2, current_loop + 1)):
            csv_path = (
                config.cal_settings.root_path
                / f"{config.target.conf}_{loop_num}"
                / f"{config.target.conf}_{loop_num}_correct_levels.csv"
            )
            if not csv_path.exists():
                logger.warning(f"未找到第{loop_num}轮能级数据: {csv_path}")
                return True
            energy_frames.append(
                pl.read_csv(csv_path)
                .select(["configuration", "EnergyTotal"])
                .rename({"EnergyTotal": f"E{i}"})
            )
            logger.info(f"读取第{loop_num}轮能级数据: {csv_path}")

        # === 2. 三轮内连接取共有能级，向量化计算跨轮标准差均值 ===
        merged = (
            energy_frames[0]
            .join(energy_frames[1], on="configuration", how="inner")
            .join(energy_frames[2], on="configuration", how="inner")
        )
        if merged.height < energy_frames[0].height:
            logger.warning(
                f"部分 configuration 在某轮数据中缺失，"
                f"仅对 {merged.height}/{energy_frames[0].height} 个共有能级计算收敛性"
            )
        energies: NDArray[np.float64] = np.asarray(merged.select(["E0", "E1", "E2"]).to_numpy(), dtype=np.float64)
        per_level_std: NDArray[np.float64] = np.std(energies, axis=1)
        avg_energy_std: float = np.mean(per_level_std).item()

        # === 3. 加载三轮 CSF 数量（前两轮来自文件，当前轮来自参数）===
        iteration_df = pl.read_csv(config.cal_path.iteration_results)
        csfs_num: list[int] = []
        for loop_num in range(current_loop - 2, current_loop):
            row = iteration_df.filter(pl.col("cal_loop_num") == loop_num)
            if row.height == 0:
                logger.warning(f"未在 iteration_results.csv 中找到第{loop_num}轮数据")
                return True
            count = int(row["current_calculation_count"][0])
            csfs_num.append(count)
            logger.info(f"读取第{loop_num}轮组态数量: {count}")
        csfs_num.append(cal_loop_csfs_count)
        logger.info(f"读取第{current_loop}轮组态数量: {cal_loop_csfs_count}")

        # === 4. 计算 CSF 数量相对标准差 ===
        csfs_mean = float(np.mean(csfs_num))
        csfs_std = float(np.std(csfs_num))
        csfs_relative_std = csfs_std / csfs_mean if csfs_mean > 0 else csfs_std

        # === 5. 读取阈值并判断收敛 ===
        energy_threshold = config.cal_settings.energy_std_threshold
        csfs_threshold = config.cal_settings.csfs_num_relative_std_threshold

        energy_converged = avg_energy_std < energy_threshold
        csfs_converged = csfs_relative_std < csfs_threshold

        logger.info("收敛性统计:")
        logger.info(f"  最近3轮组态数量: {csfs_num}")
        logger.info(
            f"  组态数量相对标准差: {csfs_relative_std:.4f}"
            f" (阈值: {csfs_threshold:.4f}) → {'收敛' if csfs_converged else '未收敛'}"
        )
        logger.info(
            f"  能级平均标准差:     {avg_energy_std:.5e}"
            f" (阈值: {energy_threshold:.5e}) → {'收敛' if energy_converged else '未收敛'}"
        )

        if energy_converged and csfs_converged:
            logger.info("能级和组态数量均已收敛，停止计算")
            return False

        if not energy_converged:
            logger.info("能级未完全收敛，继续计算")
        if not csfs_converged:
            logger.info("组态数量未稳定收敛，继续计算")
        return True

    except Exception as e:
        logger.error(f"收敛性检查过程中发生错误: {e}")
        return True


def merge_historical_ci_data(
    previous_idxs: NDArray[np.int64],
    previous_ci_squared: NDArray[np.float64],
    current_idxs: NDArray[np.int64],
    current_ci_squared: NDArray[np.float64], 
    logger: logging.Logger
) -> tuple[np.ndarray, np.ndarray]:
    """
    将历史轮次与当前轮次的 CI 系数平方数据合并为累积数据集。

    合并规则：
    - 索引取并集，覆盖写入：先写历史数据，再用当前数据覆盖，
      交集 CSF 的 CI 系数以当前轮次为准（反映最新自洽场收敛结果）。
    - 并集索引经 np.union1d 排序，保证输出有序。

    Args:
        previous_idxs:      前序轮次累积的 CSF 索引，一维，shape (n_prev,)
        previous_ci_squared: 前序轮次对应的 CI 系数平方，二维，shape (n_levels, n_prev)
        current_idxs:       当前轮次参与计算的 CSF 索引，一维，shape (n_curr,)
        current_ci_squared: 当前轮次对应的 CI 系数平方，二维，shape (n_levels, n_curr)
        logger:             日志记录器

    Returns:
        tuple:
            merged_idxs:      合并后的 CSF 索引，一维，shape (n_merged,)，已排序
            merged_ci_squared: 合并后的 CI 系数平方，二维，shape (n_levels, n_merged)
    """

    # 取索引并集，np.union1d 保证结果升序且无重复
    merged_idxs = np.union1d(previous_idxs, current_idxs)

    # 用 searchsorted 在有序 merged_idxs 中定位历史/当前 CSF 各自的落点
    prev_pos = np.searchsorted(merged_idxs, previous_idxs)
    curr_pos = np.searchsorted(merged_idxs, current_idxs)

    # 以转置视图 (n_merged, n_levels) 按行赋值，避免非连续切片
    # 先填历史数据，再用当前数据逐行覆盖——交集位置自动取当前轮次值
    merged_ci_t = np.empty(
        (len(merged_idxs), previous_ci_squared.shape[0]),
        dtype=previous_ci_squared.dtype,
    )
    merged_ci_t[prev_pos] = previous_ci_squared.T
    merged_ci_t[curr_pos] = current_ci_squared.T
    merged_ci_squared = merged_ci_t.T  # 转回 (n_levels, n_merged)

    # |交集| = |历史| + |当前| - |并集|，无需额外构建 set
    n_repeated = len(previous_idxs) + len(current_idxs) - len(merged_idxs)
    n_new = len(current_idxs) - n_repeated

    logger.info("历史数据合并统计:")
    logger.info(f"历史数据CSFs数量: {len(previous_idxs)}")
    logger.info(f"当前数据CSFs数量: {len(current_idxs)}")
    logger.info(f"合并后CSFs数量: {len(merged_idxs)}")
    logger.info(f"新增CSFs数量: {n_new}")
    logger.info(f"重复CSFs数量: {n_repeated}")

    return merged_idxs, merged_ci_squared


def ci_idx_data_processor(
    correct_levels_ci: NDArray[np.float64],
    caled_csfs_idxs_array: NDArray[np.int64],
    config: MLCalConfig,
    logger: logging.Logger,
) -> np.ndarray:
    """
    处理CI系数数据，并与历史数据合并后保存

    Args:
        correct_levels_ci: 正确能级位置的CI系数
        caled_csfs_idxs_array: 当前计算的CSF索引数组
        config: 配置对象
        logger: 日志记录器

    Returns:
        np.ndarray: CI系数平方数组
    """

    # 保存正确能级位置的CI系数平方对应CSF总池索引（用于历史数据累积）
    correct_levels_ci_2d: NDArray[np.float64] = np.atleast_2d(correct_levels_ci)
    correct_levels_ci_squared: NDArray[np.float64] = (
        correct_levels_ci_2d**2
    )  # shape: (n_correct_levels, n_current_csfs)

    accumulated_idxs_ci_path = config.cal_path.accumulated_idxs_ci_path
    # 如果是第二轮及之后，读取历史数据并合并
    if config.cal_settings.cal_loop_num > 1:
        if accumulated_idxs_ci_path.exists():
            logger.info("读取历史CI系数数据并进行合并")

            previous_idxs, previous_ci_squared = csfs_idxs_ci_loader(accumulated_idxs_ci_path)

            accumulated_idxs, accumulated_ci_squared = merge_historical_ci_data(
                previous_idxs,
                previous_ci_squared,
                caled_csfs_idxs_array,
                correct_levels_ci_squared,
                logger
            )
        else:
            raise FileNotFoundError(
                f"历史CI数据文件不存在: {accumulated_idxs_ci_path}，"
                f"第 {config.cal_settings.cal_loop_num} 轮计算依赖前序轮次保存的累积数据"
            )
            
    else:
        # 第一轮直接使用当前数据
        accumulated_idxs: NDArray[np.int64] = caled_csfs_idxs_array
        accumulated_ci_squared: NDArray[np.float64] = correct_levels_ci_squared

    # 保存合并后的累积数据到 accumulated_idxs_ci_path（供下次计算使用）

    csfs_idxs_ci_storage(accumulated_idxs_ci_path, accumulated_idxs, accumulated_ci_squared)
    logger.info(
        f"保存累积CI系数数据: {accumulated_idxs_ci_path}(包含{len(accumulated_idxs)}个CSFs)"
    )

    return correct_levels_ci_squared


def generate_train_csfs_descriptors(
    config: MLCalConfig,
    raw_csfs_descriptors: NDArray[np.float64],
    logger: logging.Logger
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

    accumulated_idxs, accumulated_ci_squared = csfs_idxs_ci_loader(accumulated_idxs_ci_path)

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
    raw_csfs_descriptors: NDArray[np.float64],
    sampled_csfs_idxs_array: NDArray[np.int64]
) -> NDArray[np.float64]:
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
