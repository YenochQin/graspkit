# -*- encoding: utf-8 -*-
"""
@Id :ml_results_analyzer.py
@date :2025/08/29 10:28:25
@author :YenochQin (秦毅)
"""

from pathlib import Path
from typing import List, Tuple
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

from .neural_network import ANNClassifier


def validate_csf_descriptors_coverage(
        descriptors: np.ndarray
        ) -> Tuple[bool, List[int]]:
    """
    验证选取的CSFs描述符子集是否满足覆盖条件:
    对于每个轨道,至少有一个CSF在其对应的电子填充数位置不为零
    Args:
        descriptors (np.ndarray): 选取出的CSFs描述符数组,形状为 (n_csfs, n_features)

    Returns:
        tuple[bool, list[int]]: (是否满足覆盖条件, 未覆盖的轨道索引列表)
    """
    # 检查输入参数
    if descriptors.size == 0:
        return False, []

    # 确定每个轨道的电子填充位置索引
    values_per_orbital = 3
    electron_index_in_orbital = 0

    # 直接通过切片获取每个轨道的电子填充
    electron_indices = np.arange(
                            electron_index_in_orbital,
                            descriptors.shape[1],
                            values_per_orbital,
                            )

    # 提取所有CSF的电子数信息
    electron_counts = descriptors[
                            :, electron_indices
                            ]  # 形状为 (num_csfs, actual_n_orbitals)

    # 检查每个轨道是否至少有一个CSF的电子数不为零
    has_nonzero_electrons = np.any(
                                electron_counts > 0, axis=0
                                )  # 形状为 (actual_n_orbitals,)

    # 找出未覆盖的轨道索引
    uncovered_orbitals_indices = np.where(~has_nonzero_electrons)[0].tolist()

    # 返回验证结果
    is_covered = len(uncovered_orbitals_indices) == 0
    return is_covered, uncovered_orbitals_indices


def select_csfs_for_coverage(
        descriptors: np.ndarray,
        uncovered_orbitals: List[int],
        candidate_descriptors: np.ndarray,
        ) -> Tuple[np.ndarray, List[int]]:
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
    electron_index_in_orbital = 0

    # 获取每个轨道的电子填充位置索引
    electron_indices = np.arange(
                                electron_index_in_orbital, 
                                candidate_descriptors.shape[1], 
                                values_per_orbital
                                )

    # 提取候选描述符中的电子数信息
    candidate_electron_counts = candidate_descriptors[:, electron_indices]

    selected_relative_indices = []
    remaining_uncovered = set(uncovered_orbitals)

    # 按顺序遍历候选描述符
    for idx in range(len(candidate_descriptors)):
        # 检查当前CSF是否包含任何剩余未覆盖的轨道
        csf_electrons = candidate_electron_counts[idx]
        covers_orbitals = [orb for orb in remaining_uncovered if csf_electrons[orb] > 0]

        if covers_orbitals:
            selected_relative_indices.append(idx)
            remaining_uncovered -= set(covers_orbitals)

            # 如果所有轨道都已覆盖，提前退出
            if not remaining_uncovered:
                break

    if not selected_relative_indices:
        return descriptors, []

    # 构建更新后的描述符数组
    new_descriptors = candidate_descriptors[selected_relative_indices]

    if descriptors.size == 0:
        updated_descriptors = new_descriptors
    else:
        updated_descriptors = np.vstack([descriptors, new_descriptors])

    return updated_descriptors, selected_relative_indices


def save_iteration_results(
        config,
        training_time,
        eval_time,
        execution_time,
        evaluation_results,
        selection_results,
        logger,):
    """
    保存迭代结果到CSV文件

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
    metadata = evaluation_results["metadata"]

    # 获取实际的评估时间（如果evaluation_results中有的话）
    actual_eval_time = metadata.get("eval_time", eval_time)

    # 保存到CSV文件
    results_file = config.cal_path.results_path / "iteration_results.csv"
    
    # 检查文件是否存在，如果不存在则写入表头
    if not results_file.exists():
        with open(results_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["iteration", "train_f1", "test_f1", "overfitting_gap", 
                            "important_count", "ml_predicted_count", "ml_new_count",
                            "total_original_count", "current_calculation_count",
                            "data_retention_rate", "important_retention_rate", 
                            "ml_retention_rate", "cal_loop_num"])

    # 计算过拟合差距
    overfitting_gap = train_metrics["f1"] - test_metrics["f1"]

    # 提取选择结果的实际数据
    important_count = selection_results.get("important_count", 0)
    ml_predicted_count = selection_results.get("ml_predicted_count", 0)
    ml_new_count = selection_results.get("ml_new_count", 0)
    total_original_count = selection_results.get("total_original_count", 1)
    current_calculation_count = selection_results.get("current_calculation_count", 1)
    data_retention_rate = selection_results.get("data_retention_rate", 0.0)
    important_retention_rate = selection_results.get("important_retention_rate", 0.0)
    ml_retention_rate = selection_results.get("ml_retention_rate", 0.0)

    # 安全获取配置参数
    cal_loop_num = getattr(config.cal_settings, "cal_loop_num", 1)

    with open(results_file, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                cal_loop_num,  # 迭代轮次
                important_count,  # 重要组态数量
                ml_predicted_count,  # ML预测的高概率组态总数
                ml_new_count,  # ML新增的组态数（下次计算用）
                total_original_count,  # 原始CSFs总数
                current_calculation_count,  # 本轮计算的组态数
                data_retention_rate,  # 数据留存率（交集/本轮计算）
                important_retention_rate,  # 重要组态占原始比例
                ml_retention_rate,  # ML预测组态占原始比例
                training_time,
                actual_eval_time,  # 推理时间
                execution_time,
                execution_time,  # 总时间（现在与执行时间相同）
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

    logger.info(f"迭代结果已保存到: {results_file}")
    logger.info(
        f"第{cal_loop_num}轮 - 重要组态: {important_count} (占原始: {important_retention_rate:.4%})"
    )
    logger.info(
        f"第{cal_loop_num}轮 - ML预测组态: {ml_new_count} (占原始: {ml_retention_rate:.4%})"
    )
    logger.info(f"第{cal_loop_num}轮 - 数据留存率: {data_retention_rate:.4%}")


def save_and_plot_results(
        evaluation_results,
        model,
        path_cfg,
        correct_levels_ci: np.ndarray,
        asfs_position: List[int],
        caled_csfs_indices_dict=None,
        y_current_cal_probability=None,
        save_model: bool = True,
        save_data: bool = True,
        plot_curves: bool = True,
        logger=None,):
    """
    保存模型预测结果、模型文件和绘制性能曲线
    使用setup_directories创建的标准目录结构

    Args:
        evaluation_results: evaluate_model函数返回的结果字典
        model: 训练好的模型对象
        correct_levels_ci: 混合系数数据Ci用于绘图
        asfs_position: 正确能级位置索引列表
        caled_csfs_indices_dict: 当前计算的CSF索引字典（用于数据对应检查）
        y_current_cal_probability: 当前计算CSF的预测概率，与混合系数维度匹配
        save_model: 是否保存模型文件
        save_data: 是否保存预测结果数据
        plot_curves: 是否绘制ROC/PR曲线
        logger: 日志记录器

    Returns:
        dict: 包含所有保存文件路径的字典
    """

    if logger:
        logger.info("开始保存结果和绘制图表")

    saved_files = {}

    # 1. 保存预测结果数据到test_data目录
    if save_data:

        # 保存测试集结果
        test_file = path_cfg.results_path / f"{path_cfg.loop_file_name}_test_results.parquet"
        pd.DataFrame(
            {
                "y_true": evaluation_results["true_labels"]["y_test"],
                "y_prediction": evaluation_results["predictions"]["y_prediction_test"],
                "y_proba": evaluation_results["probabilities"]["y_probability_test"],
            }
        ).to_parquet(test_file, index=False)
        saved_files["test_data"] = str(test_file)

        # 保存训练集结果到results目录
        train_file = path_cfg.results_path  / f"{path_cfg.loop_file_name}_train_results.parquet"
        pd.DataFrame(
            {
                "y_true": evaluation_results["true_labels"]["y_train"],
                "y_prediction": evaluation_results["predictions"]["y_prediction_train"],
                "y_proba": evaluation_results["probabilities"]["y_probability_train"],
            }
        ).to_parquet(train_file, index=False)
        saved_files["train_data"] = str(train_file)

        # 保存其他数据预测结果到results目录
        other_file = path_cfg.results_path  / f"{path_cfg.loop_file_name}_other_predictions.parquet"
        pd.DataFrame(
            {
                "y_prediction": evaluation_results["predictions"]["y_prediction_other"],
                "y_proba": evaluation_results["probabilities"]["y_probability_other"],
            }
        ).to_parquet(other_file, index=False)
        saved_files["other_predictions"] = str(other_file)

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
            # 获取真实的混合系数数据或使用占位数据
            y_prob_all = evaluation_results["probabilities"]["y_probability_all"]

            # 修复：使用正确能级位置的混合系数数据
            if len(correct_levels_ci.shape) > 1:
                # 如果是多维数组，计算每个CSF的混合系数幅值
                cal_mix_coeff_list = np.sqrt(np.sum(correct_levels_ci**2, axis=0))
            else:
                cal_mix_coeff_list = np.abs(correct_levels_ci)

            if logger:
                logger.info(f"使用正确能级位置的混合系数: {asfs_position}")
                logger.info(f"混合系数维度: {cal_mix_coeff_list.shape}")

            # 使用传入的当前计算CSF预测概率（与ann3_proba.py保持一致的数据处理）
            if y_current_cal_probability is not None:
                y_prob_current_cal = y_current_cal_probability

                if logger:
                    logger.info(
                        f"绘图数据检查 - 混合系数数量: {len(cal_mix_coeff_list)}, 预测概率数量: {len(y_prob_current_cal)}"
                    )

                # 数据维度验证
                if len(cal_mix_coeff_list) != len(y_prob_current_cal):
                    if logger:
                        logger.warning(
                            f"数据维度不匹配: 混合系数({len(cal_mix_coeff_list)}) vs 预测概率({len(y_prob_current_cal)})"
                        )
                    # 取较小的长度
                    min_len = min(len(cal_mix_coeff_list), len(y_prob_current_cal))
                    cal_mix_coeff_list = cal_mix_coeff_list[:min_len]
                    y_prob_current_cal = y_prob_current_cal[:min_len]
                    if logger:
                        logger.info(f"已调整为相同长度: {min_len}")
            elif caled_csfs_indices_dict is not None:
                # 回退到原有逻辑（从全局概率中提取对应部分）
                if 0 not in caled_csfs_indices_dict:
                    if logger:
                        logger.warning(
                            f"caled_csfs_indices_dict中缺少键0，可用键: {list(caled_csfs_indices_dict.keys())}，使用全部概率数据"
                        )
                    y_prob_current_cal = y_prob_all
                    if logger:
                        logger.info(
                            f"使用全部概率数据 - 混合系数数量: {len(cal_mix_coeff_list)}, 概率数量: {len(y_prob_current_cal)}"
                        )
                else:
                    current_cal_indices = caled_csfs_indices_dict[0]
                    y_prob_current_cal = y_prob_all[current_cal_indices]
                    if logger:
                        logger.info(
                            f"使用索引提取 - 混合系数数量: {len(cal_mix_coeff_list)}, 对应概率数量: {len(y_prob_current_cal)}"
                        )
                        logger.info(
                            f"当前计算CSF索引范围: {current_cal_indices.min()}-{current_cal_indices.max()}"
                        )
            else:
                # 如果没有提供任何信息，使用原有逻辑（可能有问题）
                y_prob_current_cal = y_prob_all
                if logger:
                    logger.warning(
                        "未提供y_current_cal_probability或caled_csfs_indices_dict，第四个子图可能显示不正确"
                    )

            # 绘制ROC和PR曲线
            plot_file = path_cfg.roc_curves_path / f"{path_cfg.loop_file_name}_roc_pr_curves.png"
            roc_auc, pr_auc = ANNClassifier.plot_curve(
                cal_mix_coeff_list,
                y_prob_current_cal,  # 使用对应的概率数据
                evaluation_results["true_labels"]["y_test"],
                evaluation_results["probabilities"]["y_probability_test"],
                str(plot_file),
            )
            saved_files["roc_pr_plot"] = str(plot_file)

            if logger:
                logger.info(f"性能图表已保存到: {path_cfg.roc_curves_path}")

        except Exception as e:
            if logger:
                logger.warning(f"绘图过程出现错误: {e}")
            else:
                print(f"绘图错误: {e}")

    if logger:
        logger.info("所有结果保存完成")

    return saved_files

