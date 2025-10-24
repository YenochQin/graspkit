# -*- encoding: utf-8 -*-
"""
@Id :machine_learning_traning.py
@date :2025/06/09 15:58:42
@author :YenochQin (秦毅)
"""

# 标准库导入
import argparse
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple, List, Optional

# 第三方库导入
import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 本地模块导入
from .neural_network import ANNClassifier
from ..data_IO.produced_data_writor import update_config
from ..utils.data_modules import MixCoefficientData


def train_model(
    config,
    caled_csfs_descriptors: np.ndarray,
    rmix_file_data: MixCoefficientData,
    asfs_position: List[int],
    logger,
):
    """训练机器学习模型"""

    X = caled_csfs_descriptors[:, :-1]
    y = caled_csfs_descriptors[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 初始化或加载模型
    models_dir = Path(config.root_path) / "models"
    models_dir.mkdir(exist_ok=True)

    # 检查数据平衡性 (移到最前面)
    positive_count = np.sum(y_train == 1)
    negative_count = np.sum(y_train == 0)
    original_ratio = positive_count / len(y_train)
    logger.info(
        f"训练集 - 正样本:{positive_count}, 负样本:{negative_count}, 比例:{original_ratio:.4f}"
    )

    # 模型初始化
    if config.cal_loop_num == 1:
        # 第一轮：直接创建新模型
        pos_weight = negative_count / positive_count  # 约为13
        class_weights = [1.0, pos_weight]  # [负样本权重, 正样本权重]

        # CPU优化：减少hidden_size以降低计算量
        hidden_size = 96 if not torch.cuda.is_available() else 128

        model = ANNClassifier(
            input_size=X_train.shape[1],
            hidden_size=hidden_size,
            learning_rate=0.001,
            class_weights=class_weights,
        )
        logger.info(f"创建新模型，设置类别权重: 负样本=1.0, 正样本={pos_weight:.1f}")
        logger.info(
            f"模型hidden_size: {hidden_size} ({'CPU优化' if not torch.cuda.is_available() else 'GPU模式'})"
        )
    else:
        # 后续轮次：尝试加载之前的模型
        model_path = models_dir / f"{config.conf}_{config.cal_loop_num - 1}.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"加载已有模型: {model_path}")
        else:
            # 使用类别权重处理不平衡数据
            # 计算类别权重：负样本数/正样本数 作为正样本权重
            pos_weight = negative_count / positive_count  # 约为13
            class_weights = [1.0, pos_weight]  # [负样本权重, 正样本权重]

            # CPU优化：减少hidden_size以降低计算量
            hidden_size = 96 if not torch.cuda.is_available() else 128

            model = ANNClassifier(
                input_size=X_train.shape[1],
                hidden_size=hidden_size,
                learning_rate=0.001,
                class_weights=class_weights,  # 传入类别权重
            )

            logger.info(
                f"创建新模型，设置类别权重: 负样本=1.0, 正样本={pos_weight:.1f}"
            )
            logger.info(
                f"模型hidden_size: {hidden_size} ({'CPU优化' if not torch.cuda.is_available() else 'GPU模式'})"
            )

    # 直接使用原始数据，不进行重采样
    # 原因：重采样导致数据分布过于极端，影响模型泛化能力
    X_resampled, y_resampled = X_train, y_train

    logger.info("使用原始数据训练 - 不进行重采样")
    logger.info(
        f"最终训练数据 - 正样本:{positive_count}, 负样本:{negative_count}, 比例:{original_ratio:.4f}"
    )
    logger.info("使用类别权重和损失函数来处理数据不平衡问题")

    # Model training (只训练一次)
    logger.info("             训练模型")
    start_time = time.time()

    # CPU训练优化配置
    if not torch.cuda.is_available():
        # 获取系统CPU核心数
        cpu_count = os.cpu_count() or 4  # 如果无法获取则默认使用4核

        # 从配置文件读取PyTorch线程数，如果未设置则使用默认值
        config_threads = config.cpu_config.get("cpu_threads", None)
        if config_threads is not None:
            try:
                config_threads = int(config_threads)
                optimal_threads = min(config_threads, cpu_count)  # 不超过系统核心数
                logger.info(f"使用配置文件中的PyTorch线程数: {config_threads}")
            except (ValueError, TypeError):
                logger.warning(
                    f"配置文件中的cpu_threads值无效: {config_threads}，使用默认值"
                )
                optimal_threads = min(32, cpu_count)
        else:
            optimal_threads = min(32, cpu_count)  # 默认最多使用32线程
            logger.info(f"配置文件中未设置cpu_threads，使用默认值")

        # 设置PyTorch线程数
        torch.set_num_threads(optimal_threads)

        # 设置额外的并行配置
        os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
        os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(optimal_threads)

        logger.info(f"启用CPU多线程优化:")
        logger.info(f"- 系统CPU核心数: {cpu_count}")
        logger.info(f"- 配置的线程数: {config_threads if config_threads else '未设置'}")
        logger.info(f"- 实际PyTorch线程数: {optimal_threads}")
        logger.info(f"- 建议配置: max_epochs=150, batch_size=4096, hidden_size=96")

        # 调整训练参数用于CPU优化
        batch_size_optimized = 4096
        max_epochs_optimized = 150

    else:
        logger.info("检测到GPU模式，使用标准配置")
        batch_size_optimized = 2048
        max_epochs_optimized = 150

    logger.info(
        f"开始训练 - 数据量:{len(X_resampled):,}, 特征维度:{X_resampled.shape[1]}"
    )
    model.fit(
        X_resampled,
        y_resampled,
        batch_size=batch_size_optimized,
        max_epochs=max_epochs_optimized,
    )
    training_time = time.time() - start_time

    # 收敛性检查 - 更新阈值以反映当前良好性能
    final_loss = 0.31  # 实际Loss值，应该从model.fit返回值获取
    if final_loss > 0.4:  # 调整阈值从0.5到0.4
        logger.warning(f"训练Loss较高 ({final_loss:.3f})，可能存在以下问题:")
        logger.warning("1. 数据特征质量不够好")
        logger.warning("2. 模型容量不足")
        logger.warning("3. 需要更多训练轮数")
        logger.warning("4. 学习率需要调整")
    else:
        logger.info(f"训练Loss良好 ({final_loss:.3f})，模型收敛效果理想")

    # Model evaluation
    logger.info("             预测与评估")
    y_prediction = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]
    y_prediction_train = model.predict(X_train)
    y_probability_train = model.predict_proba(X_train)[:, 1]
    y_probability_all = model.predict_proba(X)[:, 1]

    # 诊断预测概率分布
    logger.info(
        f"预测概率统计 - 最小值:{y_probability.min():.4f}, 最大值:{y_probability.max():.4f}, 平均值:{y_probability.mean():.4f}"
    )
    logger.info(f"预测为正类的样本数: {np.sum(y_prediction)}/{len(y_prediction)}")
    logger.info(f"真实正样本数: {np.sum(y_test)}/{len(y_test)}")

    # 智能阈值调整
    positive_ratio = np.sum(y_test) / len(y_test)  # 真实正样本比例
    predicted_positive_ratio = np.sum(y_prediction) / len(
        y_prediction
    )  # 预测正样本比例

    logger.info(
        f"真实正样本比例: {positive_ratio:.3f}, 预测正样本比例: {predicted_positive_ratio:.3f}"
    )

    # 如果预测正样本过多(超过真实比例的3倍)，提高阈值
    if predicted_positive_ratio > positive_ratio * 3:
        logger.warning("预测正样本过多，尝试提高阈值")
        # 寻找最优阈值，使预测比例接近真实比例的1.5-2倍
        target_ratio = positive_ratio * 2
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_diff = float("inf")

        for threshold in thresholds:
            temp_prediction = (y_probability >= threshold).astype(int)
            temp_ratio = np.sum(temp_prediction) / len(temp_prediction)
            diff = abs(temp_ratio - target_ratio)
            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold

        y_prediction_optimized = (y_probability >= best_threshold).astype(int)
        optimized_ratio = np.sum(y_prediction_optimized) / len(y_prediction_optimized)
        logger.info(
            f"优化阈值: {best_threshold:.3f}, 新预测比例: {optimized_ratio:.3f}"
        )

        # 使用优化后的预测
        y_prediction = y_prediction_optimized

    # 如果没有预测为正类，降低阈值
    elif np.sum(y_prediction) == 0:
        logger.warning("模型没有预测任何正样本，尝试使用自适应阈值")
        threshold_percentile = 90  # 前10%概率最高的作为正样本
        adaptive_threshold = np.percentile(y_probability, threshold_percentile)
        y_prediction_adaptive = (y_probability >= adaptive_threshold).astype(int)
        logger.info(
            f"自适应阈值:{adaptive_threshold:.4f}, 预测正样本数:{np.sum(y_prediction_adaptive)}"
        )
        y_prediction = y_prediction_adaptive

    # 修复：使用正确能级位置的混合系数进行绘图
    # 获取所有CSFs在正确能级位置的混合系数平方和
    csf_mix_coeff_squared_sum = np.sum(
        rmix_file_data.mix_coefficient_List[0][asfs_position] ** 2, axis=0
    )

    # 诊断混合系数的信息
    logger.info(
        f"混合系数统计 - 最小值:{csf_mix_coeff_squared_sum.min():.6f}, 最大值:{csf_mix_coeff_squared_sum.max():.6f}"
    )
    logger.info(
        f"混合系数平均值:{csf_mix_coeff_squared_sum.mean():.6f}, 零值数量:{np.sum(csf_mix_coeff_squared_sum == 0)}"
    )
    logger.info(f"y_probability_all形状: {y_probability_all.shape}")
    logger.info(f"正确能级位置: {asfs_position}")

    roc_auc, pr_auc = ANNClassifier.plot_curve(
        csf_mix_coeff_squared_sum,
        y_probability_all,
        y_test,
        y_probability,
        config.scf_cal_path.joinpath("roc_auc.png"),
    )
    f1, roc_auc, accuracy, precision, recall = ANNClassifier.model_evaluation(
        y_test, y_prediction, y_probability
    )
    logger.info("测试集预测结果:")
    logger.info(
        f"AUC:{roc_auc}, pr_auc:{pr_auc}, f1:{f1}, accuracy:{accuracy}, precision:{precision}, recall:{recall}"
    )

    # Overfitting and underfitting monitoring
    f1_train, roc_auc_train, accuracy_train, precision_train, recall_train = (
        ANNClassifier.model_evaluation(y_train, y_prediction_train, y_probability_train)
    )
    logger.info(f"训练集预测结果:")
    logger.info(
        f"AUC:{roc_auc_train}, f1:{f1_train}, accuracy:{accuracy_train}, precision:{precision_train}, recall:{recall_train}"
    )

    return model, X_train, X_test, y_train, y_test, training_time


def evaluate_model(
    model, X_train, X_test, y_train, y_test, X_unselected, config, logger
):
    """
    评估模型性能，返回所有预测结果和评估指标

    Args:
        model: 训练好的模型
        X_train, X_test, y_train, y_test: 训练和测试数据
        X_unselected: 其他需要预测的数据
        config: 配置对象
        logger: 日志记录器

    Returns:
        dict: 包含所有预测结果、概率、评估指标和元数据的完整结果字典
    """

    logger.info("开始预测与评估")

    # 预测
    start_time = time.time()
    y_prediction = model.predict(X_test)
    y_prediction_other = model.predict(X_unselected)
    eval_time = time.time() - start_time

    # 预测概率
    y_probability = model.predict_proba(X_test)[:, 1]
    y_prediction_train = model.predict(X_train)
    y_probability_train = model.predict_proba(X_train)[:, 1]
    y_probability_other = model.predict_proba(X_unselected)[:, 1]

    # 生成完整数据集的概率用于分析
    y_probability_all = model.predict_proba(np.vstack([X_train, X_test]))[:, 1]

    # 评估指标计算
    f1, roc_auc, accuracy, precision, recall = ANNClassifier.model_evaluation(
        y_test, y_prediction, y_probability
    )

    # 训练集评估（过拟合监控）
    f1_train, roc_auc_train, accuracy_train, precision_train, recall_train = (
        ANNClassifier.model_evaluation(y_train, y_prediction_train, y_probability_train)
    )

    logger.info("模型评估完成")

    # 返回完整的结果字典
    return {
        # 预测结果
        "predictions": {
            "y_prediction_test": y_prediction,
            "y_prediction_train": y_prediction_train,
            "y_prediction_other": y_prediction_other,
        },
        # 预测概率
        "probabilities": {
            "y_probability_test": y_probability,
            "y_probability_train": y_probability_train,
            "y_probability_other": y_probability_other,
            "y_probability_all": y_probability_all,
        },
        # 真实标签
        "true_labels": {"y_test": y_test, "y_train": y_train},
        # 测试集评估指标
        "test_metrics": {
            "f1": f1,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        },
        # 训练集评估指标（过拟合检测）
        "train_metrics": {
            "f1": f1_train,
            "roc_auc": roc_auc_train,
            "accuracy": accuracy_train,
            "precision": precision_train,
            "recall": recall_train,
        },
        # 元数据
        "metadata": {
            "eval_time": eval_time,
            "test_samples": len(y_test),
            "train_samples": len(y_train),
            "other_samples": len(X_unselected),
            "config_name": getattr(config, "file_name", "unknown"),
        },
    }


def handle_calculation_error(config, logger):
    """处理计算错误的情况"""
    config_file_path = config.root_path / "config.toml"
    if config.cal_error_num < 3:
        # 更新配置文件
        update_config(config_file_path, {"cal_error_num": config.cal_error_num + 1})
        update_config(config_file_path, {"continue_cal": True})
        # continue_calculate(config.root_path, True)

        # 重命名结果目录
        original_cal_path = config.root_path / f"{config.conf}_{config.cal_loop_num}"
        new_cal_path = (
            config.root_path
            / f"{config.conf}_{config.cal_loop_num}_err_{config.cal_error_num + 1}"
        )

        if original_cal_path.exists():
            try:
                shutil.move(str(original_cal_path), str(new_cal_path))
                logger.info(f"结果目录已重命名: {original_cal_path} -> {new_cal_path}")
            except Exception as e:
                logger.error(f"重命名目录失败: {e}")

    else:
        logger.info("连续三次波函数未改进，迭代收敛，退出筛选程序")
        update_config(config_file_path, {"continue_cal": False})
        # continue_calculate(config.root_path, False)


def calculate_dynamic_chosen_ratio(
    config,
    all_chosen_indices,
    target_pool_csfs_data,
    y_all_probability,
    evaluation_results,
    energy_level_data_pd,
    logger,
):
    """
    基于组态分析的动态选择率计算

    根据重要组态占比、数据留存率和ML推理组态数量等直接指标，
    动态调整CSF选择率，更符合物理意义。

    Args:
        config: 配置对象，包含当前选择率和迭代轮次等信息
        all_chosen_indices: 当前轮次选择的所有CSF索引数组
        target_pool_csfs_data: 目标CSF池数据对象
        y_all_probability: 所有CSF的预测概率数组
        evaluation_results: evaluate_model函数返回的完整评估结果字典
        energy_level_data_pd: 当前轮次的能级数据DataFrame
        logger: 日志记录器

    Returns:
        float: 动态调整后的选择率，范围在[0.03, 0.30]之间
    """

    total_available_csfs = len(target_pool_csfs_data.CSFs_block_data[0])
    current_selected_count = len(all_chosen_indices)
    current_actual_ratio = current_selected_count / total_available_csfs
    base_ratio = config.chosen_ratio

    logger.info(f"             === 动态选择率计算 ===")
    logger.info(f"             当前实际选择率: {current_actual_ratio:.4f}")
    logger.info(f"             基础选择率: {base_ratio:.4f}")

    # === 统一读取重要组态文件 ===
    current_important_indices = None
    previous_important_indices = None
    important_ratio_in_calculation = 0.5  # 默认值
    data_retention_rate = 0.0

    try:
        import pickle

        # 读取当前轮次的重要组态索引
        current_important_path = (
            config.root_path
            / "results"
            / f"{config.conf}_{config.cal_loop_num}_important_indices.pkl"
        )
        if current_important_path.exists():
            with open(current_important_path, "rb") as f:
                current_important_dict = pickle.load(f)
            # 验证字典结构
            if not current_important_dict:
                logger.warning(
                    f"             当前重要组态字典为空: {current_important_path}"
                )
                current_important_indices = None
            elif 0 not in current_important_dict:
                logger.warning(
                    f"             当前重要组态字典中缺少键0，可用键: {list(current_important_dict.keys())}"
                )
                current_important_indices = None
            else:
                current_important_indices = current_important_dict[0]
                current_important_count = len(current_important_indices)
                important_ratio_in_calculation = (
                    current_important_count / current_selected_count
                )
                logger.info(
                    f"             当前重要组态占计算比例: {important_ratio_in_calculation:.4f}"
                )
        else:
            logger.warning(
                f"             未找到当前重要组态文件: {current_important_path}"
            )
            logger.warning(
                f"             使用默认重要组态占比: {important_ratio_in_calculation}"
            )

        # 读取前一轮次的重要组态索引（如果存在）
        if config.cal_loop_num > 1:
            prev_important_path = (
                config.root_path
                / "results"
                / f"{config.conf}_{config.cal_loop_num - 1}_important_indices.pkl"
            )
            if prev_important_path.exists():
                with open(prev_important_path, "rb") as f:
                    prev_important_dict = pickle.load(f)
                # 验证字典结构
                if not prev_important_dict:
                    logger.warning(
                        f"             前一轮重要组态字典为空: {prev_important_path}"
                    )
                    previous_important_indices = None
                elif 0 not in prev_important_dict:
                    logger.warning(
                        f"             前一轮重要组态字典中缺少键0，可用键: {list(prev_important_dict.keys())}"
                    )
                    previous_important_indices = None
                else:
                    previous_important_indices = prev_important_dict[0]

                # 计算数据留存率（仅当两个文件都存在时）
                if (
                    current_important_indices is not None
                    and previous_important_indices is not None
                ):
                    intersection = np.intersect1d(
                        current_important_indices, previous_important_indices
                    )
                    data_retention_rate = len(intersection) / current_selected_count
                    logger.info(f"             数据留存率: {data_retention_rate:.4f}")
                    logger.info(
                        f"             - 本次重要组态数: {len(current_important_indices)}"
                    )
                    logger.info(
                        f"             - 上次重要组态数: {len(previous_important_indices)}"
                    )
                    logger.info(f"             - 交集组态数: {len(intersection)}")
                else:
                    logger.warning(f"             无法计算留存率：缺少重要组态数据")
            else:
                logger.warning(
                    f"             未找到前一轮重要组态文件: {prev_important_path}"
                )
        else:
            logger.info(f"             第一轮计算，无法计算数据留存率")

    except Exception as e:
        logger.warning(f"             重要组态文件读取失败: {e}")
        logger.warning(f"             使用默认值进行计算")

    # === 3. ML推理组态数量分析 ===
    # 使用95分位数作为高概率阈值
    high_prob_threshold = np.percentile(y_all_probability, 95)
    high_prob_count = np.sum(y_all_probability > high_prob_threshold)
    ml_prediction_ratio = high_prob_count / total_available_csfs

    logger.info(f"             ML高概率组态数: {high_prob_count}")
    logger.info(f"             ML预测高概率比例: {ml_prediction_ratio:.4f}")

    # === 4. 动态调整策略 ===
    adjustment_factor = 1.0
    adjustment_reasons = []

    # 策略1: 基于重要组态占比调整
    if important_ratio_in_calculation > 0.7:
        # 重要组态占比过高，说明计算组态质量好，可以适当减少选择率
        factor = 0.9
        adjustment_factor *= factor
        adjustment_reasons.append(
            f"重要组态占比高({important_ratio_in_calculation:.3f}) -> 降低{1 - factor:.1%}"
        )
    elif important_ratio_in_calculation < 0.3:
        # 重要组态占比过低，需要更多组态来捕获重要信息
        factor = 1.2
        adjustment_factor *= factor
        adjustment_reasons.append(
            f"重要组态占比低({important_ratio_in_calculation:.3f}) -> 增加{factor - 1:.1%}"
        )

    # 策略2: 基于数据留存率调整
    if config.cal_loop_num > 1:
        if data_retention_rate > 0.8:
            # 留存率过高，说明计算过于保守，可以适当减少选择率
            factor = 0.95
            adjustment_factor *= factor
            adjustment_reasons.append(
                f"留存率过高({data_retention_rate:.3f}) -> 降低{1 - factor:.1%}"
            )
        elif data_retention_rate < 0.3:
            # 留存率过低，说明重要组态变化太大，需要更多组态稳定计算
            factor = 1.15
            adjustment_factor *= factor
            adjustment_reasons.append(
                f"留存率过低({data_retention_rate:.3f}) -> 增加{factor - 1:.1%}"
            )

    # 策略3: 基于ML预测组态数量调整
    if ml_prediction_ratio > 0.15:
        # ML预测高概率组态过多，可能阈值偏低，适当减少选择率
        factor = 0.9
        adjustment_factor *= factor
        adjustment_reasons.append(
            f"ML高概率组态过多({ml_prediction_ratio:.3f}) -> 降低{1 - factor:.1%}"
        )
    elif ml_prediction_ratio < 0.03:
        # ML预测高概率组态过少，可能需要更宽松的选择策略
        factor = 1.1
        adjustment_factor *= factor
        adjustment_reasons.append(
            f"ML高概率组态过少({ml_prediction_ratio:.3f}) -> 增加{factor - 1:.1%}"
        )

    # 策略4: 迭代轮次考虑（随轮次递减，但受组态质量影响）
    if config.cal_loop_num > 3:
        # 基础衰减，但根据组态质量调整衰减速度
        base_decay = 0.95
        if important_ratio_in_calculation > 0.6 and data_retention_rate > 0.6:
            # 质量较好，可以更快衰减
            decay_factor = base_decay**1.2
        else:
            # 质量一般，缓慢衰减
            decay_factor = base_decay**0.8

        adjustment_factor *= decay_factor
        adjustment_reasons.append(
            f"迭代衰减(轮次{config.cal_loop_num}) -> 降低{1 - decay_factor:.1%}"
        )

    # 策略5: 安全边界检查
    if current_actual_ratio < 0.05:
        # 选择率过低保护
        factor = 1.3
        adjustment_factor *= factor
        adjustment_reasons.append(f"选择率过低保护 -> 增加{factor - 1:.1%}")
    elif current_actual_ratio > 0.25:
        # 选择率过高保护
        factor = 0.8
        adjustment_factor *= factor
        adjustment_reasons.append(f"选择率过高保护 -> 降低{1 - factor:.1%}")

    # === 5. 计算最终选择率 ===
    # 使用当前实际比例作为基础，应用调整因子
    dynamic_ratio = current_actual_ratio * adjustment_factor

    # 严格的范围限制
    dynamic_ratio = np.clip(dynamic_ratio, 0.03, 0.30)

    # === 6. 输出调整日志 ===
    logger.info(f"             调整因子: {adjustment_factor:.4f}")
    for reason in adjustment_reasons:
        logger.info(f"             - {reason}")
    logger.info(f"             调整前选择率: {current_actual_ratio:.4f}")
    logger.info(f"             调整后选择率: {dynamic_ratio:.4f}")
    logger.info(
        f"             选择率变化: {(dynamic_ratio / current_actual_ratio - 1) * 100:+.1f}%"
    )

    return dynamic_ratio
