# -*- encoding: utf-8 -*-
"""
@Id :ml_results_analyzer.py
@date :2025/08/29 10:28:25
@author :YenochQin (秦毅)
"""

import csv
import logging

import numpy as np
from numpy.typing import NDArray
import polars as pl

from ..utils.data_modules import MLDataCounts
from .ml_types import EvaluationResults, PredictionOutputs
from ..data_IO import MLCalConfig

def validate_csf_desc_coverage(
    final_sampled_idxs: NDArray[np.int64],
    raw_csfs_descriptors: NDArray[np.float64],
    logger: logging.Logger,
) -> NDArray[np.int64]:
    """
    验证已选取的 CSF 描述符子集是否满足轨道覆盖条件。

    覆盖条件：对于描述符中涉及的每个轨道，至少存在一个 CSF，
    其对应的电子填充数（descriptor 中每隔 3 列取一次的位置）不为零。
    若存在未覆盖的轨道，则从剩余候选 CSF 中自动补选，直至所有轨道均被覆盖。

    Args:
        final_sampled_idxs (NDArray[np.int64]): 已选取的 CSF 在原始描述符数组中的索引，
            形状为 (n_selected,)，不得为空。
        raw_csfs_descriptors (NDArray[np.float64]): 全量 CSF 描述符数组，
            形状为 (n_total_csfs, n_features)，每个轨道占 3 列，
            第 0 列为电子填充数。
        logger (logging.Logger): 日志记录器，用于输出覆盖检验及补选过程信息。

    Returns:
        NDArray[np.int64]: 满足覆盖条件后的 CSF 索引数组（已排序、去重），
            形状为 (n_final,)。若初始选取已满足覆盖条件则直接返回原索引；
            否则追加补选索引后返回。

    Raises:
        RuntimeError: 若 final_sampled_idxs 为空数组。
    """
    # 检查输入参数

    if final_sampled_idxs.size == 0:
        error_msg = f"final_sampled_idxs 为空，输入错误"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    current_sampled_descriptors = raw_csfs_descriptors[final_sampled_idxs]

    # 确定每个轨道的电子填充位置索引
    values_per_orbital = 3
    electron_idx_in_orbital = 0

    # 直接通过切片获取每个轨道的电子填充
    electron_idxs: NDArray[np.int64] = np.arange(
        electron_idx_in_orbital,
        current_sampled_descriptors.shape[1],
        values_per_orbital,
        dtype=np.int64,
    )

    # 提取所有CSF的电子数信息
    electron_counts = current_sampled_descriptors[:, electron_idxs]
    # 形状为 (num_csfs, actual_n_orbitals)

    # 检查每个轨道是否至少有一个CSF的电子数不为零
    has_nonzero_electrons = np.any(electron_counts > 0, axis=0)
    # 形状为 (actual_n_orbitals,)

    # 找出未覆盖的轨道索引
    uncovered_orbitals_idxs = np.where(~has_nonzero_electrons)[0].tolist()

    # 返回验证结果
    is_covered = len(uncovered_orbitals_idxs) == 0
    if not is_covered:
        logger.info(f"检测到未覆盖的轨道索引: {uncovered_orbitals_idxs}")
        logger.info(f"开始补充选择以满足轨道覆盖条件")
        all_csfs_idxs = np.arange(raw_csfs_descriptors.shape[0], dtype=np.int64)
        remaining_candidates_idxs = np.setdiff1d(all_csfs_idxs, final_sampled_idxs)
        remaining_descriptors = raw_csfs_descriptors[remaining_candidates_idxs]
        # 修复：使用正确的参数顺序调用select_csfs_for_coverage函数
        _, additional_idxs_relative = select_csfs_for_coverage(
            current_sampled_descriptors,  # 当前已选择的CSFs描述符
            uncovered_orbitals_idxs,  # 未覆盖的轨道索引列表
            remaining_descriptors,  # 剩余候选CSFs的描述符
        )
        additional_idxs = remaining_candidates_idxs[additional_idxs_relative]
        additional_idxs_array = np.array(additional_idxs)
        final_sampled_idxs = np.unique(
            np.sort(np.concatenate([final_sampled_idxs, additional_idxs_array]))
        )
        logger.info(
            f"使用新函数补充选择了 {len(additional_idxs)} 个CSF以满足轨道覆盖条件"
        )

    return final_sampled_idxs


def select_csfs_for_coverage(
    descriptors: NDArray[np.float64],
    uncovered_orbitals: list[int],
    candidate_descriptors: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    当覆盖验证失败时，从候选 CSF 中选取能覆盖所有缺失轨道的最小子集。

    等价性：对每个未覆盖轨道 o，令 f(o) 为候选数组中第一个覆盖 o 的行索引，
    则贪心顺序遍历所选出的集合恒等于 {f(o) | o ∈ uncovered_orbitals}。
    因此可用 np.argmax 逐列定位第一个 True，再 np.unique 去重，
    完全替代原有的 Python 循环与列表操作。

    Args:
        descriptors (NDArray[np.float64]): 当前已选取的 CSF 描述符数组，
            形状为 (n_csfs, n_features)，允许为空数组。
        uncovered_orbitals (list[int]): 未覆盖的轨道索引列表，索引对应
            candidate_electron_counts 的列（即 electron_idxs 的位置序号）。
        candidate_descriptors (NDArray[np.float64]): 候选 CSF 描述符数组，
            形状为 (n_candidates, n_features)，每个轨道占 3 列，第 0 列为电子填充数。

    Returns:
        tuple[NDArray[np.float64], NDArray[np.int64]]:
            - 更新后的描述符数组：将新选取的行追加到 descriptors 之后，
            形状为 (n_csfs + n_selected, n_features)。
            - 选取的相对索引数组：对应 candidate_descriptors 中被选行的索引，形状为 (n_selected,)，已排序、去重；若无可选则为空数组。
    """
    empty_idxs: NDArray[np.int64] = np.empty(0, dtype=np.int64)

    if not uncovered_orbitals:
        return descriptors, empty_idxs

    if candidate_descriptors.size == 0:
        return descriptors, empty_idxs

    values_per_orbital = 3
    electron_idx_in_orbital = 0

    electron_idxs: NDArray[np.int64] = np.arange(
        electron_idx_in_orbital,
        candidate_descriptors.shape[1],
        values_per_orbital,
        dtype=np.int64,
    )

    # shape: (n_candidates, n_orbitals)
    candidate_electron_counts = candidate_descriptors[:, electron_idxs]

    uncovered_arr = np.array(uncovered_orbitals, dtype=np.int64)

    # coverage_matrix[i, j] = True 表示候选 CSF i 覆盖第 j 个未覆盖轨道
    # shape: (n_candidates, n_uncovered)
    coverage_matrix = candidate_electron_counts[:, uncovered_arr] > 0

    # 过滤掉完全没有候选覆盖的轨道（避免 argmax 在全 False 列返回 0 的歧义）
    has_coverage = coverage_matrix.any(axis=0)  # shape: (n_uncovered,)
    if not has_coverage.any():
        return descriptors, empty_idxs

    # 对每个可覆盖轨道，取第一个覆盖它的候选行索引
    # argmax 在 bool 数组上返回第一个 True 的位置
    first_covering_idxs = np.argmax(coverage_matrix[:, has_coverage], axis=0)

    selected_relative_idxs: NDArray[np.int64] = np.unique(first_covering_idxs).astype(
        np.int64
    )

    new_descriptors = candidate_descriptors[selected_relative_idxs]

    if descriptors.size == 0:
        updated_descriptors = new_descriptors
    else:
        updated_descriptors = np.vstack([descriptors, new_descriptors])

    return updated_descriptors, selected_relative_idxs


def save_training_results(
    config: MLCalConfig,
    evaluation_results: EvaluationResults,
    logger: logging.Logger,
) -> None:
    """
    保存训练结果到CSV文件

    Args:
        config: 配置对象
        training_time: 训练时间
        eval_time: 评估时间（模型推理时间）
        execution_time: 总执行时间
        evaluation_results: evaluate_model函数返回的结果字典
        selection_results: 选择结果字典，包含实际的组态选择信息
        logger: 日志记录器
    """

    labeled_metrics = evaluation_results["labeled_metrics"]

    # 保存到CSV文件
    results_file = config.cal_path.training_results

    # 检查文件是否存在，如果不存在则写入表头
    if not results_file.exists():
        with open(results_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "cal_loop_num",
                    "metric_scope",
                    "labeled_samples",
                    "labeled_f1",
                    "labeled_roc_auc",
                    "labeled_accuracy",
                    "labeled_precision",
                    "labeled_recall",
                ]
            )

    cal_loop_num = config.cal_settings.cal_loop_num
    metadata = evaluation_results["metadata"]

    with open(results_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                cal_loop_num,
                metadata["metric_scope"],
                metadata["labeled_samples"],
                labeled_metrics["f1"],
                labeled_metrics["roc_auc"],
                labeled_metrics["accuracy"],
                labeled_metrics["precision"],
                labeled_metrics["recall"],
            ]
        )

    logger.info(f"训练结果已保存到: {results_file}")


def save_iteration_results(
    config: MLCalConfig,
    train_data_counts: MLDataCounts,
    logger: logging.Logger
) -> None:
    """
    保存迭代结果到CSV文件

    Args:
        config: 配置对象
        selection_results: 选择结果字典，包含实际的组态选择信息
        logger: 日志记录器
    """

    # 保存到CSV文件
    results_file = config.cal_path.iteration_results

    # 检查文件是否存在，如果不存在则写入表头
    if not results_file.exists():
        with open(results_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "cal_loop_num",  # 迭代轮次
                    "important_count",  # 重要组态数量
                    "ml_sampled_count",  # ML预测的高概率组态总数
                    "total_original_count",  # 原始CSFs总数
                    "current_calculation_count",  # 本轮计算的组态数
                    "screening_retention_rate",  # 验证留存率
                    "ml_retention_rate",  # ML 预测留存率
                    "iteration_retention_rate",  # 迭代增长率
                ]
            )

    # 提取选择结果的实际数据

    important_csfs_count = getattr(train_data_counts, "important_csfs_count", 0)
    ml_sampled_count = getattr(train_data_counts, "ml_sampled_count", 0)
    total_original_count = getattr(train_data_counts, "total_csfs_count", 0)
    current_calculation_count = getattr(train_data_counts, "cal_csfs_count", 0)
    screening_retention_rate = getattr(train_data_counts, "screening_retention_rate", 0)
    ml_retention_rate = getattr(train_data_counts, "ml_retention_rate", 0)
    iteration_retention_rate = getattr(train_data_counts, "iteration_retention_rate", 0)

    cal_loop_num = config.cal_settings.cal_loop_num

    with open(results_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                cal_loop_num,  # 迭代轮次
                important_csfs_count,  # 重要组态数量
                ml_sampled_count,  # ML预测的高概率组态总数
                total_original_count,  # 原始CSFs总数
                current_calculation_count,  # 本轮计算的组态数
                screening_retention_rate,  # 验证留存率
                ml_retention_rate,  # ML 预测留存率
                iteration_retention_rate,  # 迭代增长率
            ]
        )

    logger.info(f"迭代结果已保存到: {results_file}")


def save_and_plot_results(
    config: MLCalConfig,
    logger: logging.Logger,
    evaluation_results: EvaluationResults,
    prediction_outputs: PredictionOutputs,
    model: ANNClassifier,
    correct_levels_ci: np.ndarray,
    y_current_cal_probability: np.ndarray | None = None,
    save_model: bool = True,
    save_data: bool = True,
    plot_curves: bool = True,
) -> dict[str, str]:
    """
    保存模型预测结果、模型文件和绘制性能曲线
    使用setup_directories创建的标准目录结构

    Args:
        config: 配置对象
        logger: 日志记录器
        evaluation_results: evaluate_model函数返回的结果字典
        model: 训练好的模型对象
        correct_levels_ci: 混合系数数据Ci用于绘图，shape: (n_levels, n_csfs) 或 (n_csfs,)
        y_current_cal_probability: 当前计算CSF的预测概率，shape: (n_csfs, n_levels)
        save_model: 是否保存模型文件
        save_data: 是否保存预测结果数据
        plot_curves: 是否绘制ROC/PR曲线

    Returns:
        dict: 包含所有保存文件路径的字典
    """
    # 内部定义path_cfg
    path_cfg = config.cal_path

    if logger:
        logger.info("开始保存结果和绘制图表")

    saved_files = {}

    # 1. 保存累计已标注样本的诊断结果
    if save_data:
        labeled_file = (
            path_cfg.results_path / f"{path_cfg.loop_file_name}_labeled_results.parquet"
        )
        pl.DataFrame(
            {
                "y_true": evaluation_results["true_labels"]["y_labeled"],
                "y_prediction": prediction_outputs["y_prediction_labeled"],
                "y_proba": evaluation_results["probabilities"]["y_probability_labeled"],
            }
        ).write_parquet(labeled_file)
        saved_files["labeled_data"] = str(labeled_file)

        if logger:
            logger.info(f"累计已标注样本诊断数据已保存到: {labeled_file}")

    # 2. 保存模型文件到models目录
    if save_model:
        model_file = path_cfg.models_path / f"{path_cfg.loop_file_name}.pt"
        model.save_model(str(model_file))
        saved_files["model"] = str(model_file)

        if logger:
            logger.info(f"模型已保存到: {model_file}")

    # 3. 正式迭代流程不再保留 holdout 测试集，因此跳过 ROC/PR 曲线绘制
    if plot_curves and logger:
        logger.info("当前迭代模式未切分测试集，跳过 ROC/PR 曲线绘制")

    if logger:
        logger.info("所有结果保存完成")

    return saved_files


def ml_results_statistics(
    train_data_counts: MLDataCounts,
    logger: logging.Logger
) -> MLDataCounts:
    """
    统计ML结果并返回完整的selection_results字典

    Returns:
        dict: 包含selection_results所需的所有字段
    """
    total_csfs_count = train_data_counts.total_csfs_count
    cal_csfs_count = train_data_counts.cal_csfs_count

    # 除零保护
    if total_csfs_count <= 0 and cal_csfs_count <= 0:
        logger.error(
            f"train_data_counts 中初始计数无效: {total_csfs_count}， {cal_csfs_count}"
        )
        return train_data_counts

    logger.info(f"统计 ML sampling 信息:")
    logger.info(f"- 原始CSFs总数: {total_csfs_count}")

    important_csfs_count = getattr(train_data_counts, "important_csfs_count", None)
    if important_csfs_count is not None:
        train_data_counts.screening_retention_rate = (
            important_csfs_count / cal_csfs_count
        )
        logger.info(
            f"""- 计算重要 CSFs 数量: {important_csfs_count}
            - (验证留存率: {train_data_counts.screening_retention_rate:.4%})
            - (重要组态留存率: {train_data_counts.important_retention_rate:.4%})"""
        )

    ml_sampled_count = getattr(train_data_counts, "ml_sampled_count", None)
    if ml_sampled_count is not None:
        train_data_counts.ml_retention_rate = ml_sampled_count / (
            total_csfs_count - cal_csfs_count
        )
        logger.info(
            f"- ML预测 CSFs 数量: {ml_sampled_count} (ML 预测留存率: {train_data_counts.ml_retention_rate:.4%})"
        )

        train_data_counts.iteration_retention_rate = (
            important_csfs_count + ml_sampled_count
        ) / cal_csfs_count
        logger.info(
            f"- ML预测 CSFs 数量: {ml_sampled_count} (迭代增长率: {train_data_counts.iteration_retention_rate:.4%})"
        )

    return train_data_counts
