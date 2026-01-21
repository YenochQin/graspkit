# -*- encoding: utf-8 -*-
"""
@Id :transition_data_processor.py
@date :2024/01/15 15:15:57
@author :YenochQin (秦毅)

@version 2.0: 迁移到使用新的 TransitionLoader
@version 2.1: 移除重复的数据解析功能，全部使用 TransitionLoader
@version 3.0: 移除 TransitionDataCollection 和 LSJTransitionDataCollection 类，
            数据加载和解析功能完全由 TransitionLoader 提供
@version 4.0: 合并 transition_data_analyzer.py，
            整合跃迁数据处理和分析功能
"""

import numpy as np
import polars as pl

from ..utils.progress_manager import wrap_iterator

#######################################################################
# 跃迁数据分析函数
#######################################################################


def transition_dT_cal(
    transition_rate_B: float | np.float64, transition_rate_C: float | np.float64
) -> float | np.float64:
    """计算跃迁几率的相对差异

    Args:
        transition_rate_B: B规范计算的跃迁几率
        transition_rate_C: C规范计算的跃迁几率

    Returns:
        跃迁几率的相对差异值
    """
    if max(transition_rate_B, transition_rate_C) == 0:
        return 0.0

    transition_dT = abs(transition_rate_B - transition_rate_C) / max(
        transition_rate_B, transition_rate_C
    )

    return transition_dT


def transition_data_level_location(
    transition_data_df: pl.DataFrame, level_df: pl.DataFrame
) -> pl.DataFrame:
    """为标准格式（.[c]t文件）跃迁数据添加能级索引

    根据能级的 Pos, J, Parity 匹配能级位置，将能级数据表中的 No 列
    添加到跃迁数据中作为上下能级的索引（Upper_index 和 Lower_index）。

    Args:
        transition_data_df: 跃迁数据 DataFrame，包含以下列：
            - upper_pos, upper_j, upper_parity (上能级信息)
            - lower_pos, lower_j, lower_parity (下能级信息)
        level_df: 能级数据 DataFrame，包含以下列：
            - Pos, J, Parity (能级定位信息)
            - No (能级编号列)

    Returns:
        添加了 Upper_index 和 Lower_index 列的跃迁数据 DataFrame

    Raises:
        KeyError: 当缺少必需的列时
        ValueError: 当无法找到匹配的能级时
    """
    # 验证跃迁数据包含必需的列
    required_transition_cols = [
        "upper_pos",
        "upper_j",
        "upper_parity",
        "lower_pos",
        "lower_j",
        "lower_parity",
    ]
    transition_cols = transition_data_df.columns
    missing_transition_cols = [
        col for col in required_transition_cols if col not in transition_cols
    ]
    if missing_transition_cols:
        raise KeyError(
            f"跃迁数据缺少必需的列: {missing_transition_cols}。"
            f"需要: {required_transition_cols}"
        )

    # 验证能级数据包含必需的列
    required_level_cols = ["Pos", "J", "Parity", "No"]
    level_cols = level_df.columns
    missing_level_cols = [col for col in required_level_cols if col not in level_cols]
    if missing_level_cols:
        raise KeyError(
            f"能级数据缺少必需的列: {missing_level_cols}。需要: {required_level_cols}"
        )

    # 去重：确保每个 Pos+J+Parity 组合只有一条记录（取 No 最小的）
    level_df_unique = level_df.sort("No").group_by(["Pos", "J", "Parity"]).agg(
        pl.col("No").first().alias("No")
    )

    # 准备能级查找表（用于上能级匹配）
    upper_level_lookup = level_df_unique.select(
        pl.col("Pos").alias("upper_pos"),
        pl.col("J").alias("upper_j"),
        pl.col("Parity").alias("upper_parity"),
        pl.col("No").alias("Upper_index"),
    )

    # 准备能级查找表（用于下能级匹配）
    lower_level_lookup = level_df_unique.select(
        pl.col("Pos").alias("lower_pos"),
        pl.col("J").alias("lower_j"),
        pl.col("Parity").alias("lower_parity"),
        pl.col("No").alias("Lower_index"),
    )

    # 使用 join 添加上能级索引
    with_upper_index = transition_data_df.join(
        upper_level_lookup, on=["upper_pos", "upper_j", "upper_parity"], how="left"
    )

    # 使用 join 添加下能级索引
    with_level_indexes = with_upper_index.join(
        lower_level_lookup, on=["lower_pos", "lower_j", "lower_parity"], how="left"
    )

    # 检查是否有未匹配的能级
    unmatched_upper_count = with_level_indexes.filter(pl.col("Upper_index").is_null()).height
    unmatched_lower_count = with_level_indexes.filter(pl.col("Lower_index").is_null()).height

    if unmatched_upper_count > 0 or unmatched_lower_count > 0:
        unmatched_count = max(unmatched_upper_count, unmatched_lower_count)
        raise ValueError(
            f"有 {unmatched_count} 条跃迁数据无法找到匹配的能级。"
            f"请检查跃迁数据和能级数据的一致性。"
        )

    return with_level_indexes


def lsj_transition_data_level_location(
    transition_data_df: pl.DataFrame, level_df: pl.DataFrame
) -> pl.DataFrame:
    """为LSJ格式（.[c]t.lsj文件）跃迁数据添加能级索引

    根据能级的 J 和 configuration_raw 匹配能级位置，将能级数据表中的 No 列
    添加到跃迁数据中作为上下能级的索引（Upper_index 和 Lower_index）。

    Args:
        transition_data_df: 跃迁数据 DataFrame，包含以下列：
            - upper_j, upper_configuration (上能级信息)
            - lower_j, lower_configuration (下能级信息)
        level_df: 能级数据 DataFrame，包含以下列：
            - J, configuration_raw (能级定位信息)
            - No (能级编号列)

    Returns:
        添加了 Upper_index 和 Lower_index 列的跃迁数据 DataFrame

    Raises:
        KeyError: 当缺少必需的列时
        ValueError: 当无法找到匹配的能级时
    """
    # 验证跃迁数据包含必需的列
    required_transition_cols = [
        "upper_j",
        "upper_configuration",
        "lower_j",
        "lower_configuration",
    ]
    transition_cols = transition_data_df.columns
    missing_transition_cols = [
        col for col in required_transition_cols if col not in transition_cols
    ]
    if missing_transition_cols:
        raise KeyError(
            f"跃迁数据缺少必需的列: {missing_transition_cols}。"
            f"需要: {required_transition_cols}"
        )

    # 验证能级数据包含必需的列
    required_level_cols = ["J", "configuration_raw", "No"]
    level_cols = level_df.columns
    missing_level_cols = [col for col in required_level_cols if col not in level_cols]
    if missing_level_cols:
        raise KeyError(
            f"能级数据缺少必需的列: {missing_level_cols}。需要: {required_level_cols}"
        )

    # 去重：确保每个 J+configuration_raw 组合只有一条记录（取 No 最小的）
    level_df_unique = level_df.sort("No").group_by(["J", "configuration_raw"]).agg(
        pl.col("No").first().alias("No")
    )

    # 准备能级查找表（用于上能级匹配）
    upper_level_lookup = level_df_unique.select(
        pl.col("J").alias("upper_j"),
        pl.col("configuration_raw").alias("upper_configuration"),
        pl.col("No").alias("Upper_index"),
    )

    # 准备能级查找表（用于下能级匹配）
    lower_level_lookup = level_df_unique.select(
        pl.col("J").alias("lower_j"),
        pl.col("configuration_raw").alias("lower_configuration"),
        pl.col("No").alias("Lower_index"),
    )

    # 使用 join 添加上能级索引
    with_upper_index = transition_data_df.join(
        upper_level_lookup,
        on=["upper_j", "upper_configuration"],
        how="left",
    )

    # 使用 join 添加下能级索引
    with_level_indexes = with_upper_index.join(
        lower_level_lookup,
        on=["lower_j", "lower_configuration"],
        how="left",
    )

    # 检查是否有未匹配的能级
    unmatched_upper_count = with_level_indexes.filter(pl.col("Upper_index").is_null()).height
    unmatched_lower_count = with_level_indexes.filter(pl.col("Lower_index").is_null()).height

    if unmatched_upper_count > 0 or unmatched_lower_count > 0:
        unmatched_count = max(unmatched_upper_count, unmatched_lower_count)
        raise ValueError(
            f"有 {unmatched_count} 条跃迁数据无法找到匹配的能级。"
            f"请检查跃迁数据和能级数据的一致性。"
        )

    return with_level_indexes

#######################################################################
# 接口函数
#######################################################################


def add_transition_level_index(
    transition_df: pl.DataFrame, level_df: pl.DataFrame
) -> pl.DataFrame:
    """为跃迁数据添加能级索引（自动检测格式）

    根据 TransitionLoader.to_dataframe() 生成的跃迁数据格式，
    自动选择合适的能级定位函数，将能级数据表中的 No 列
    添加到跃迁数据中作为上下能级的索引（Upper_index 和 Lower_index）。

    支持的格式：
    - 标准格式（.[c]t 文件）：匹配 upper_pos/upper_j/upper_parity → Pos/J/Parity
    - LSJ 格式（.[c]t.lsj 文件）：匹配 upper_j/upper_configuration → J/configuration_raw

    Args:
        transition_df: TransitionLoader.to_dataframe() 生成的跃迁数据 DataFrame
        level_df: 能级数据 DataFrame，必须包含 No 列

    Returns:
        添加了 Upper_index 和 Lower_index 列的跃迁数据 DataFrame

    Raises:
        KeyError: 当缺少必需的列时
        ValueError: 当无法识别数据格式或无法找到匹配的能级时
    """
    # 检测跃迁数据格式
    transition_cols = transition_df.columns

    # 检查是否为标准格式（.[c]t）
    has_standard_cols = all(
        col in transition_cols
        for col in ["upper_pos", "upper_j", "upper_parity", "lower_pos", "lower_j", "lower_parity"]
    )

    # 检查是否为 LSJ 格式（.[c]t.lsj）
    has_lsj_cols = all(
        col in transition_cols
        for col in ["upper_j", "upper_configuration", "lower_j", "lower_configuration"]
    )

    if has_standard_cols:
        with_level_indexes = transition_data_level_location(transition_df, level_df)
    elif has_lsj_cols:
        with_level_indexes = lsj_transition_data_level_location(transition_df, level_df)
    else:
        raise ValueError(
            "无法识别跃迁数据格式。"
            "标准格式需要包含: upper_pos, upper_j, upper_parity, lower_pos, lower_j, lower_parity；"
            "LSJ 格式需要包含: upper_j, upper_configuration, lower_j, lower_configuration"
        )

    # 按 Lower_index 和 Upper_index 排序
    sorted_transitions = with_level_indexes.sort(["Lower_index", "Upper_index"])

    return sorted_transitions


def calculate_branching_fraction(
    transition_df: pl.DataFrame,
    level_df: pl.DataFrame,
    branching_fraction_threshold: float = 0.0001,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """计算跃迁的分支比和能级寿命

    计算每个上能级的总跃迁率、每条跃迁的分支比，并添加寿命信息到能级数据。
    删除分支比小于阈值的跃迁。

    Args:
        transition_df: 包含 Upper_index, Lower_index, transition_type,
                       transition_rate_C, transition_rate_M 列的跃迁数据
        level_df: 能级数据 DataFrame
        branching_fraction_threshold: 分支比阈值，小于此值的跃迁将被删除

    Returns:
        (transition_df, level_df) 元组：
        - transition_df: 添加了 sum_A, branching_fraction 列，并过滤了低分支比跃迁
        - level_df: 添加了 lifetime_C 列的能级数据
    """
    # 验证必需的列
    required_cols = ["Upper_index", "Lower_index", "transition_type"]
    missing_cols = [col for col in required_cols if col not in transition_df.columns]
    if missing_cols:
        raise KeyError(f"跃迁数据缺少必需的列: {missing_cols}")

    # 确定使用的跃迁率列
    if "transition_rate_M" in transition_df.columns:
        rate_col = "transition_rate_M"  # 磁性跃迁使用 M 规范
    elif "transition_rate_C" in transition_df.columns:
        rate_col = "transition_rate_C"  # 电性跃迁使用 C 规范
    else:
        raise ValueError("跃迁数据缺少 transition_rate_C 或 transition_rate_M 列")

    # 计算每个上能级的总跃迁率
    sum_A = (
        transition_df.group_by("Upper_index")
        .agg(pl.col(rate_col).sum().alias("sum_A"))
        .sort("Upper_index")
    )

    # 将总跃迁率添加到跃迁数据
    with_sum_a = transition_df.join(sum_A, on="Upper_index", how="left")

    # 计算分支比
    with_branching_fraction = with_sum_a.with_columns(
        (pl.col(rate_col) / pl.col("sum_A")).alias("branching_fraction")
    )

    # 过滤掉分支比小于阈值的跃迁
    # 使用 >= 以便 threshold=0 时保留所有数据（包括 branching_fraction=0 的跃迁）
    filtered_transitions = with_branching_fraction.filter(
        pl.col("branching_fraction") >= branching_fraction_threshold
    )

    # 计算能级寿命并添加到 level_df
    # 将 No 列转换为 Upper_index 进行匹配
    level_with_lifetime = level_df.with_columns(
        pl.col("No").alias("Upper_index")
    )
    level_with_lifetime = level_with_lifetime.join(
        sum_A, on="Upper_index", how="left"
    )
    level_with_lifetime = level_with_lifetime.with_columns(
        pl.when(pl.col("sum_A").is_null() | (pl.col("sum_A") == 0))
        .then(None)
        .otherwise(1.0 / pl.col("sum_A"))
        .alias("lifetime_C")
    )
    level_with_lifetime = level_with_lifetime.drop("Upper_index", "sum_A")

    return filtered_transitions, level_with_lifetime


def level_transition_data_processing(
    transition_df: pl.DataFrame,
    level_df: pl.DataFrame,
    branching_fraction_threshold: float = 0.0001,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """处理跃迁数据的完整流程

    按照 data_process 的逻辑进行处理，但使用新的函数和 Polars DataFrame。
    执行以下步骤：
    1. 添加能级索引（自动检测格式）
    2. 按 Lower_index 和 Upper_index 排序
    3. 计算 A_B/A_C 比值（如果有 B 规范数据）
    4. 计算能级寿命和分支比
    5. 过滤低分支比跃迁

    Args:
        transition_df: TransitionLoader.to_dataframe() 生成的跃迁数据 DataFrame
        level_df: 能级数据 DataFrame，必须包含 No 列
        branching_fraction_threshold: 分支比阈值，小于此值的跃迁将被删除

    Returns:
        (transition_df, level_df) 元组：
        - transition_df: 处理后的跃迁数据，包含能级索引、分支比等
        - level_df: 添加了寿命信息的能级数据

    Raises:
        KeyError: 当缺少必需的列时
        ValueError: 当无法识别数据格式或无法找到匹配的能级时
    """
    # 1. 添加能级索引并排序
    with_level_indexes = add_transition_level_index(transition_df, level_df)

    # 2. 计算 A_B/A_C 比值（如果有 B 规范数据）
    if "transition_rate_B" in with_level_indexes.columns and "transition_rate_C" in with_level_indexes.columns:
        with_ab_ratio = with_level_indexes.with_columns(
            (pl.col("transition_rate_B") / pl.col("transition_rate_C")).alias(
                "A_B_to_A_C"
            )
        )
    else:
        with_ab_ratio = with_level_indexes

    # 3. 计算分支比和能级寿命
    processed_transitions, level_with_lifetime = calculate_branching_fraction(
        with_ab_ratio, level_df, branching_fraction_threshold
    )

    return processed_transitions, level_with_lifetime
