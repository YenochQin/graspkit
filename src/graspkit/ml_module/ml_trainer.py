# -*- encoding: utf-8 -*-
"""
@Id :machine_learning_traning.py
@date :2025/06/09 15:58:42
@author :YenochQin (秦毅)
"""

# 标准库导入
import logging
import math
import os
import shutil
import time

# 第三方库导入
import joblib
import numpy as np
import torch

# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ..data_IO.produced_data_writor import update_config
from ..utils.data_modules import MLDataCounts
from .ml_results_analyzer import save_training_results

# 本地模块导入
from .neural_network import ANNClassifier


def train_model(
    config,
    caled_csfs_descriptors: np.ndarray,
    correct_levels_ci: np.ndarray,
    logger: logging.Logger,
):
    """训练机器学习模型"""

    X = caled_csfs_descriptors[:, :-1]
    y = caled_csfs_descriptors[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 初始化或加载模型
    config.cal_path.models_path.mkdir(exist_ok=True)

    # 检查数据平衡性 (移到最前面)
    positive_count = np.sum(y_train == 1)
    negative_count = np.sum(y_train == 0)
    original_ratio = positive_count / len(y_train)
    logger.info(
        f"训练集 - 正样本:{positive_count}, 负样本:{negative_count}, 比例:{original_ratio:.4f}"
    )
    logger.info(f"描述符长度{X_train.shape[1]}")

    # 模型初始化
    if config.cal_settings.cal_loop_num == 1:
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
            model_architecture="tensornet",
        )
        logger.info(f"创建新模型，设置类别权重: 负样本=1.0, 正样本={pos_weight:.1f}")
        logger.info(
            f"模型hidden_size: {hidden_size} ({'CPU优化' if not torch.cuda.is_available() else 'GPU模式'})"
        )
    else:
        # 后续轮次：尝试加载之前的模型
        model_path = (
            config.cal_path.models_path
            / f"{config.target.conf}_{config.cal_settings.cal_loop_num - 1}.pkl"
        )
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
                model_architecture="tensornet",
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

    # CPU训练优化配置
    if not torch.cuda.is_available():
        # 获取系统CPU核心数
        cpu_count = os.cpu_count() or 4  # 如果无法获取则默认使用4核

        # 从配置文件读取PyTorch线程数，如果未设置则使用默认值
        cpu_threads = getattr(config.server_settings, "cpu_threads", 16)
        if cpu_threads is not None:
            try:
                cpu_threads = int(cpu_threads)
                optimal_threads = min(cpu_threads, cpu_count)  # 不超过系统核心数
                logger.info(f"使用配置文件中的PyTorch线程数: {cpu_threads}")
            except (ValueError, TypeError):
                logger.warning(
                    f"配置文件中的cpu_threads值无效: {cpu_threads}，使用默认值"
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
        logger.info(f"- 配置的线程数: {cpu_threads if cpu_threads else '未设置'}")
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
    csf_mix_coeff_squared_sum = np.sum(correct_levels_ci**2, axis=0)

    # 诊断混合系数的信息
    logger.info(
        f"混合系数统计 - 最小值:{csf_mix_coeff_squared_sum.min():.6f}, 最大值:{csf_mix_coeff_squared_sum.max():.6f}"
    )
    logger.info(
        f"混合系数平均值:{csf_mix_coeff_squared_sum.mean():.6f}, 零值数量:{np.sum(csf_mix_coeff_squared_sum == 0)}"
    )
    logger.info(f"y_probability_all形状: {y_probability_all.shape}")

    f1, roc_auc, accuracy, precision, recall = ANNClassifier.model_evaluation(
        y_test, y_prediction, y_probability
    )
    logger.info("测试集预测结果:")
    logger.info(
        f"AUC:{roc_auc}, f1:{f1}, accuracy:{accuracy}, precision:{precision}, recall:{recall}"
    )

    # Overfitting and underfitting monitoring
    (f1_train, roc_auc_train, accuracy_train, precision_train, recall_train) = (
        ANNClassifier.model_evaluation(y_train, y_prediction_train, y_probability_train)
    )
    logger.info(f"训练集预测结果:")
    logger.info(
        f"AUC:{roc_auc_train}, f1:{f1_train}, accuracy:{accuracy_train}, precision:{precision_train}, recall:{recall_train}"
    )

    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, X_test, y_train, y_test, config, logger):
    """
    评估模型性能，返回所有预测结果和评估指标

    Args:
        model: 训练好的模型
        X_train, X_test, y_train, y_test: 训练和测试数据
        config: 配置对象
        logger: 日志记录器

    Returns:
        dict: 包含所有预测结果、概率、评估指标和元数据的完整结果字典

    Note:
        仅对训练集和测试集进行评估，不对 X_unselected 进行预测
        X_unselected 的预测应在推理阶段单独进行（参考旧版 ann3_proba.py）
    """

    logger.info("开始预测与评估")

    # 预测 - 仅对训练集和测试集
    start_time = time.time()
    y_prediction = model.predict(X_test)
    eval_time = time.time() - start_time

    # 预测概率
    y_probability = model.predict_proba(X_test)[:, 1]
    y_prediction_train = model.predict(X_train)
    y_probability_train = model.predict_proba(X_train)[:, 1]

    # 生成完整训练数据集的概率用于分析
    y_probability_all = model.predict_proba(np.vstack([X_train, X_test]))[:, 1]

    # 评估指标计算
    test_f1, test_roc_auc, test_accuracy, test_precision, test_recall = (
        ANNClassifier.model_evaluation(y_test, y_prediction, y_probability)
    )

    # 训练集评估（过拟合监控）
    train_f1, train_roc_auc, train_accuracy, train_precision, train_recall = (
        ANNClassifier.model_evaluation(y_train, y_prediction_train, y_probability_train)
    )

    logger.info("测试集预测结果:")
    logger.info(
        f"AUC: {test_roc_auc:.4f}, F1: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}"
    )
    logger.info(f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    logger.info("训练集预测结果:")
    logger.info(
        f"AUC: {train_roc_auc:.4f}, F1: {train_f1:.4f}, Accuracy: {train_accuracy:.4f}"
    )
    logger.info(f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")

    # 过拟合监控
    overfitting_check = train_f1 - test_f1
    logger.info(f"过拟合检查差异(训练-测试): {overfitting_check:.4f}")
    overfitting_threshold = getattr(config.ml_config, "overfitting_threshold", 0.1)
    underfitting_threshold = getattr(config.ml_config, "underfitting_threshold", -0.05)
    if overfitting_check > overfitting_threshold:
        logger.warning("检测到可能的过拟合现象")
    elif overfitting_check < underfitting_threshold:
        logger.warning("检测到可能的欠拟合现象")

    logger.info("模型评估完成")
    evaluation_results = {
        # 预测结果
        "predictions": {
            "y_prediction_test": y_prediction,
            "y_prediction_train": y_prediction_train,
        },
        # 预测概率
        "probabilities": {
            "y_probability_test": y_probability,
            "y_probability_train": y_probability_train,
            "y_probability_all": y_probability_all,
        },
        # 真实标签
        "true_labels": {"y_test": y_test, "y_train": y_train},
        # 测试集评估指标
        "test_metrics": {
            "f1": test_f1,
            "roc_auc": test_roc_auc,
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
        },
        # 训练集评估指标（过拟合检测）
        "train_metrics": {
            "f1": train_f1,
            "roc_auc": train_roc_auc,
            "accuracy": train_accuracy,
            "precision": train_precision,
            "recall": train_recall,
        },
        # 元数据
        "metadata": {
            "eval_time": eval_time,
            "test_samples": len(y_test),
            "train_samples": len(y_train),
            "config_name": getattr(config, "file_name", "unknown"),
        },
    }
    save_training_results(config, evaluation_results, logger)

    # 返回完整的结果字典
    return evaluation_results


def predict_model(
    model,
    raw_csfs_descriptors: np.ndarray,
    caled_csfs_idxs_array: np.ndarray,
    correct_levels_ci_squared: np.ndarray,
    config,
    train_data_counts: MLDataCounts,
    logger: logging.Logger,
):
    # 获取未选择的CSF索引
    total_csfs_count = raw_csfs_descriptors.shape[0]
    all_csfs_idxs = np.arange(total_csfs_count)
    current_calc_idxs = caled_csfs_idxs_array

    unselected_idxs = np.setdiff1d(all_csfs_idxs, current_calc_idxs)

    # 仅对未选择的CSF进行预测（使用分批处理避免内存溢出）
    X_unselected_for_prediction = raw_csfs_descriptors[unselected_idxs]
    y_unselected_probability = model.predict_proba_batch(
        X_unselected_for_prediction,
        batch_size=10_000_000,  # 可根据内存调整
    )
    # 修复：使用正类概率（第1列）进行预测，而不是整个概率矩阵
    y_unselected_prediction = (y_unselected_probability[:, 1] > 0.5).astype(int)

    logger.info(f"推理了 {len(y_unselected_probability)} 个未选择CSF组态")

    # 为绘图准备当前计算CSF的预测概率
    # 对当前计算的CSF也进行预测（用于绘图和分析）
    X_current_calc = raw_csfs_descriptors[current_calc_idxs]
    y_current_cal_probability = model.predict_proba(X_current_calc)[:, 1]
    logger.info(f"当前计算CSF数量: {len(current_calc_idxs)}")

    # 基于混合系数选择重要组态（已验证重要组态）
    cutoff_value = getattr(config.cal_settings, "cutoff_value", 1e-10)
    csfs_above_threshold_idxs = np.where(
        np.any(correct_levels_ci_squared >= np.float64(cutoff_value), axis=0)
    )[0]
    verified_important_idxs = caled_csfs_idxs_array[csfs_above_threshold_idxs]
    logger.info(f"已验证重要组态数: {len(verified_important_idxs)}")

    # 提取已验证重要组态对应的CI系数
    verified_important_ci_coefficients = correct_levels_ci_squared[
        :, csfs_above_threshold_idxs
    ]
    logger.info(
        f"已提取 {verified_important_ci_coefficients.shape[1]} 个重要组态的CI系数，维度: {verified_important_ci_coefficients.shape}"
    )

    # ============ 智能动态选择机制 ============
    logger.info("      组态采样")
    logger.info("更新重要组态索引")

    # 计算当前重要组态数量作为基准
    current_important_count = len(verified_important_idxs)

    # 获取最小重要组态数量保护
    min_important_count = min(50, int(total_csfs_count * 0.01))  # 默认1%或50个
    if current_important_count <= min_important_count:
        current_important_count = min_important_count
        logger.info(f"重要组态数目小于等于最小值，调整为{min_important_count}")

    # 获取扩展比例
    expansion_ratio = getattr(config.cal_settings, "expansion_ratio", 2)
    new_sampling_CSFs_num = math.ceil(expansion_ratio * current_important_count)

    # 在未选择的CSF中找出被预测为重要的组态
    ml_predicted_important_mask = y_unselected_prediction == 1
    ml_predicted_important_local_idxs = np.where(ml_predicted_important_mask)[0]

    ml_predicted_important_global_idxs = unselected_idxs[
        ml_predicted_important_local_idxs
    ]

    logger.info(f"开始选择组态，当前重要组态数为：{len(verified_important_idxs)}")
    logger.info(
        f"ML预测的重要组态数（在未选择中）：{ml_predicted_important_global_idxs.shape}"
    )
    logger.info(f"目标新增组态数：{new_sampling_CSFs_num}")

    # 设置上限
    sampling_ratio = getattr(config.cal_settings, "sampling_ratio", 0.085)
    max_sampling_CSFs_num = math.ceil(total_csfs_count * sampling_ratio)
    if new_sampling_CSFs_num + current_important_count > max_sampling_CSFs_num:
        new_sampling_CSFs_num = max_sampling_CSFs_num - current_important_count
        logger.info(f"目标新增组态数超过最大选择数，调整为{new_sampling_CSFs_num}")

    if len(ml_predicted_important_local_idxs) >= new_sampling_CSFs_num:
        # 情况1：ML预测的重要组态数量充足，按概率排序选择top-k
        logger.info(f"ML预测组态充足，按概率排序选择前{new_sampling_CSFs_num}个")

        # 获取ML预测重要组态的正类概率（只取第1列）
        ml_predicted_important_probabilities = y_unselected_probability[
            ml_predicted_important_local_idxs, 1
        ]

        # 按概率降序排序
        probability_sorted_idxs = np.argsort(ml_predicted_important_probabilities)[::-1]

        # 选择前new_sampling_CSFs_num个
        top_k_local_idxs = ml_predicted_important_local_idxs[
            probability_sorted_idxs[:new_sampling_CSFs_num]
        ]

        ml_sampled_idxs = unselected_idxs[top_k_local_idxs]
        logger.info(
            f"从{len(ml_predicted_important_local_idxs)}个ML预测重要组态中选择了{len(ml_sampled_idxs)}个"
        )
    else:
        # 情况2：ML预测的重要组态数量不足，全部采用
        logger.info(
            f"ML预测组态不足，全部采用{len(ml_predicted_important_global_idxs)}个"
        )
        ml_sampled_idxs = ml_predicted_important_global_idxs

    train_data_counts.important_csfs_count = verified_important_idxs.shape[0]
    train_data_counts.ml_predicted_count = ml_predicted_important_global_idxs.shape[0]
    train_data_counts.ml_sampled_count = ml_sampled_idxs.shape[0]

    return (
        ml_sampled_idxs,
        verified_important_idxs,
        y_current_cal_probability,
        train_data_counts
    )


def handle_calculation_error(config, logger: logging.Logger):
    """处理计算错误的情况"""
    config_file_path = config.cal_settings.root_path / "config.toml"
    if config.cal_settings.cal_error_num < 3:
        # 更新配置文件
        update_config(
            config_file_path,
            {
                "cal_settings": {
                    "cal_error_num": config.cal_settings.cal_error_num + 1,
                    "continue_cal": True,
                }
            },
        )

        # 重命名结果目录
        original_cal_path = (
            config.cal_settings.root_path
            / f"{config.target.conf}_{config.cal_settings.cal_loop_num}"
        )
        new_cal_path = (
            config.cal_settings.root_path
            / f"{config.target.conf}_{config.cal_settings.cal_loop_num}_err_{config.cal_settings.cal_error_num + 1}"
        )

        if original_cal_path.exists():
            try:
                shutil.move(str(original_cal_path), str(new_cal_path))
                logger.info(f"结果目录已重命名: {original_cal_path} -> {new_cal_path}")
            except Exception as e:
                logger.error(f"重命名目录失败: {e}")

    else:
        logger.info("连续三次波函数未改进，迭代收敛，退出筛选程序")
        update_config(config_file_path, {"cal_settings": {"continue_cal": True}})
