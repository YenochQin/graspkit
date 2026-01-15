# -*- encoding: utf-8 -*-
"""
@Id :ml_results_analyzer.py
@date :2025/08/29 10:28:25
@author :YenochQin (秦毅)
"""

import csv
import logging
from typing import Tuple

import joblib
import numpy as np
import polars as pl

from ..grasp_data_extractor.ASF_data_collection import ConfigurationFormatter
from ..utils.data_modules import MLDataCounts
from .neural_network import ANNClassifier


def validate_csf_desc_coverage(
    final_sampled_idxs: np.ndarray,
    raw_csfs_descriptors: np.ndarray,
    logger: logging.Logger
) -> np.ndarray:
    """
    验证选取的CSFs描述符子集是否满足覆盖条件:
    对于每个轨道,至少有一个CSF在其对应的电子填充数位置不为零
    Args:
        descriptors (np.ndarray): 选取出的CSFs描述符数组,形状为 (n_csfs, n_features)

    Returns:
        tuple[bool, list[int]]: (是否满足覆盖条件, 未覆盖的轨道索引列表)
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
    electron_idxs = np.arange(
        electron_idx_in_orbital,
        current_sampled_descriptors.shape[1],
        values_per_orbital,
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
        all_csfs_idxs = np.arange(raw_csfs_descriptors.shape[0])
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
    descriptors: np.ndarray,
    uncovered_orbitals: list[int],
    candidate_descriptors: np.ndarray,
) -> Tuple[np.ndarray, list[int]]:
    """
    当覆盖验证失败时,从给定的候选描述符中按顺序选取包含缺少轨道的CSF描述符

    Args:
        descriptors (np.ndarray): 当前的CSFs描述符数组,形状为 (n_csfs, n_features)
        uncovered_orbitals (list[int]): 未覆盖的轨道索引列表
        candidate_descriptors (np.ndarray): 候选CSFs描述符数组,形状为 (n_candidates, n_features)

    Returns:
        tuple[np.ndarray, list[int]]: (更新后的描述符数组, 选取的CSF索引列表)
            - 更新后的描述符数组包含原有描述符和新选取的描述符
            - 选取的CSF索引列表对应于candidate_descriptors中的相对索引
    """
    if not uncovered_orbitals:
        return descriptors, []

    if candidate_descriptors.size == 0:
        return descriptors, []

    # 确定每个轨道的电子填充位置索引
    values_per_orbital = 3
    electron_idx_in_orbital = 0

    # 获取每个轨道的电子填充位置索引
    electron_idxs = np.arange(
        electron_idx_in_orbital, 
        candidate_descriptors.shape[1], 
        values_per_orbital
    )

    # 提取候选描述符中的电子数信息
    candidate_electron_counts = candidate_descriptors[:, electron_idxs]

    selected_relative_idxs = []
    remaining_uncovered = set(uncovered_orbitals)

    # 按顺序遍历候选描述符
    for idx in range(len(candidate_descriptors)):
        # 检查当前CSF是否包含任何剩余未覆盖的轨道
        csf_electrons = candidate_electron_counts[idx]
        covers_orbitals = [orb for orb in remaining_uncovered if csf_electrons[orb] > 0]

        if covers_orbitals:
            selected_relative_idxs.append(idx)
            remaining_uncovered -= set(covers_orbitals)

            # 如果所有轨道都已覆盖，提前退出
            if not remaining_uncovered:
                break

    if not selected_relative_idxs:
        return descriptors, []

    # 构建更新后的描述符数组
    new_descriptors = candidate_descriptors[selected_relative_idxs]

    if descriptors.size == 0:
        updated_descriptors = new_descriptors
    else:
        updated_descriptors = np.vstack([descriptors, new_descriptors])

    return updated_descriptors, selected_relative_idxs


def save_training_results(
    config, 
    evaluation_results: dict, 
    logger: logging.Logger
):
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

    # 从新的结果结构中提取指标
    test_metrics = evaluation_results["test_metrics"]
    train_metrics = evaluation_results["train_metrics"]

    # 保存到CSV文件
    results_file = config.cal_path.training_results

    # 检查文件是否存在，如果不存在则写入表头
    if not results_file.exists():
        with open(results_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "cal_loop_num",  # 迭代轮次
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
            )

    # 计算过拟合差距
    overfitting_gap = train_metrics["f1"] - test_metrics["f1"]

    # 安全获取配置参数
    cal_loop_num = getattr(config.cal_settings, "cal_loop_num", 1)

    with open(results_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                cal_loop_num,  # 迭代轮次
                test_metrics["f1"],
                test_metrics["roc_auc"],
                test_metrics["accuracy"],
                test_metrics["precision"],
                test_metrics["recall"],
                train_metrics["f1"],
                train_metrics["roc_auc"],
                train_metrics["accuracy"],
                train_metrics["precision"],
                train_metrics["recall"],
                overfitting_gap,  # 过拟合差距
            ]
        )

    logger.info(f"训练结果已保存到: {results_file}")


def save_iteration_results(
    config,
    train_data_counts: MLDataCounts,
    logger: logging.Logger
):
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

    # 安全获取配置参数
    cal_loop_num = getattr(config.cal_settings, "cal_loop_num", 1)

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
    config,
    logger: logging.Logger,
    evaluation_results,
    model,
    correct_levels_ci: np.ndarray,
    y_current_cal_probability=None,
    save_model: bool = True,
    save_data: bool = True,
    plot_curves: bool = True,
):
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

    # 1. 保存预测结果数据到test_data目录
    if save_data:
        # 保存测试集结果
        test_file = path_cfg.results_path / f"{path_cfg.loop_file_name}_test_results.parquet"
        pl.DataFrame(
            {
                "y_true": evaluation_results["true_labels"]["y_test"],
                "y_prediction": evaluation_results["predictions"]["y_prediction_test"],
                "y_proba": evaluation_results["probabilities"]["y_probability_test"],
            }
        ).write_parquet(test_file)
        saved_files["test_data"] = str(test_file)

        # 保存训练集结果到results目录
        train_file = path_cfg.results_path / f"{path_cfg.loop_file_name}_train_results.parquet"
        pl.DataFrame(
            {
                "y_true": evaluation_results["true_labels"]["y_train"],
                "y_prediction": evaluation_results["predictions"]["y_prediction_train"],
                "y_proba": evaluation_results["probabilities"]["y_probability_train"],
            }
        ).write_parquet(train_file)
        saved_files["train_data"] = str(train_file)

        # 注意：不再保存other_predictions，因为evaluate_model不再对X_unselected进行预测
        # X_unselected的预测应在推理阶段单独进行（参考旧版ann3_proba.py）

        if logger:
            logger.info(f"预测数据已保存到: {test_file} 和 {train_file}")

    # 2. 保存模型文件到models目录
    if save_model:
        model_file = path_cfg.models_path / f"{path_cfg.loop_file_name}.pkl"
        joblib.dump(model, model_file)
        saved_files["model"] = str(model_file)

        if logger:
            logger.info(f"模型已保存到: {model_file}")

    # 3. 绘制性能曲线到roc_curves目录
    if plot_curves:
        try:
            # 生成latex格式的谱项符号列表
            latex_form_spectral_term: list[str] = []
            spectral_term = config.cal_settings.spectral_term
            if spectral_term is not None:
                for term in spectral_term:
                    _, format_LS_coupling = ConfigurationFormatter(term).conf_format()
                    latex_form_spectral_term.append(format_LS_coupling)

            # 确定能级数量
            if len(correct_levels_ci.shape) > 1:
                n_levels = correct_levels_ci.shape[0]  # 多能级情况
            else:
                n_levels = 1  # 单能级情况

            if logger:
                logger.info(f"检测到能级数量: {n_levels}")
                if spectral_term:
                    logger.info(f"谱项符号: {spectral_term}")
                    logger.info(f"LaTeX格式: {latex_form_spectral_term}")

            # 为每个能级绘制图表
            for level_idx in range(n_levels):
                # 提取当前能级的CI系数
                if len(correct_levels_ci.shape) > 1:
                    level_ci = np.abs(correct_levels_ci[level_idx, :])
                else:
                    level_ci = np.abs(correct_levels_ci)

                # 提取当前能级的预测概率（当前计算的CSFs）
                if y_current_cal_probability is not None:
                    if len(y_current_cal_probability.shape) > 1:
                        # y_current_cal_probability: (n_csfs, n_levels)
                        level_probability = y_current_cal_probability[:, level_idx]
                    else:
                        # 单能级情况
                        level_probability = y_current_cal_probability
                else:
                    # 回退：使用全局概率
                    y_prob_all = evaluation_results["probabilities"]["y_probability_all"]
                    if len(y_prob_all.shape) > 1:
                        level_probability = y_prob_all[:, level_idx]
                    else:
                        level_probability = y_prob_all

                # 提取当前能级的测试集标签和概率
                y_test = evaluation_results["true_labels"]["y_test"]
                y_probability_test = evaluation_results["probabilities"]["y_probability_test"]
                if len(y_test.shape) > 1:
                    # 多标签情况：提取当前能级的列
                    level_y_test = y_test[:, level_idx]
                    level_y_probability_test = y_probability_test[:, level_idx]
                else:
                    # 单标签情况：直接使用
                    level_y_test = y_test
                    level_y_probability_test = y_probability_test

                if logger:
                    logger.info(
                        f"能级 {level_idx}: CI系数shape={level_ci.shape}, 预测概率shape={level_probability.shape}, 测试集shape={level_y_test.shape}"
                    )

                # 生成图表标题（LaTeX 格式支持）
                level_title = f"${latex_form_spectral_term[level_idx]}$ Ci Values vs Predicted Probability"

                # 绘制当前能级的ROC和PR曲线
                plot_file = path_cfg.roc_curves_path / f"{path_cfg.loop_file_name}_level{level_idx}_roc_pr_curves.png"
                ANNClassifier.plot_curve(
                    level_ci,
                    level_probability,
                    level_y_test,
                    level_y_probability_test,
                    str(plot_file),
                    level_title=level_title,
                )
                saved_files[f"roc_pr_plot_level{level_idx}"] = str(plot_file)

                if logger:
                    logger.info(f"能级 {level_idx} 性能图表已保存到: {plot_file}")

        except Exception as e:
            if logger:
                logger.warning(f"绘图过程出现错误: {e}")
            else:
                print(f"绘图错误: {e}")

    if logger:
        logger.info("所有结果保存完成")

    return saved_files


def ml_results_statistics(
    train_data_counts: MLDataCounts, logger: logging.Logger
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
        train_data_counts.ml_retention_rate = ml_sampled_count / (total_csfs_count - cal_csfs_count)
        logger.info(
            f"- ML预测 CSFs 数量: {ml_sampled_count} (ML 预测留存率: {train_data_counts.ml_retention_rate:.4%})"
        )

        train_data_counts.iteration_retention_rate = (important_csfs_count + ml_sampled_count) / cal_csfs_count
        logger.info(
            f"- ML预测 CSFs 数量: {ml_sampled_count} (迭代增长率: {train_data_counts.iteration_retention_rate:.4%})"
        )

    return train_data_counts
