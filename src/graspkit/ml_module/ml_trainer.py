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
import numpy as np
import torch

# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from sklearn.ensemble import RandomForestClassifier

from ..data_IO import (
        MLCalConfig,
        update_config
    )
from ..utils.data_modules import MLDataCounts
from .ml_types import EvaluationResults, PredictionOutputs
from .ml_results_analyzer import save_training_results

# 本地模块导入
from .neural_network import ANNClassifier


def _select_model_architecture(
    sample_count: int,
    positive_count: int,
    logger: logging.Logger,
) -> str:
    """为小样本训练选择更稳健的模型结构。"""
    if sample_count < 20_000 or positive_count < 2_048:
        logger.info(
            "检测到小样本训练场景，使用 standard 架构以避免 TensorNet 在低覆盖率数据下塌缩"
        )
        return "standard"

    logger.info("使用 tensornet 架构进行训练")
    return "tensornet"


def train_model(
    config: MLCalConfig,
    caled_csfs_descriptors: np.ndarray,
    correct_levels_ci: np.ndarray,
    logger: logging.Logger,
) -> tuple[ANNClassifier, np.ndarray, np.ndarray]:
    """训练机器学习模型（支持多标签分类）"""

    # 数据提取：支持多标签分类
    # caled_csfs_descriptors 形状: (n_current_csfs, descriptor_features + n_correct_levels)
    # 前 descriptor_features 列是描述符特征，后 n_correct_levels 列是每个能级的标签

    # 从 correct_levels_ci 推断能级数量
    n_correct_levels = correct_levels_ci.shape[0] if correct_levels_ci.ndim == 2 else 1
    descriptor_features = caled_csfs_descriptors.shape[1] - n_correct_levels

    X = caled_csfs_descriptors[:, :descriptor_features]
    y = caled_csfs_descriptors[:, descriptor_features:]  # 多标签：所有能级的标签

    # 迭代流程中使用累计的全部已标注样本训练，不再切分验证集或测试集
    X_train = X
    y_train = y

    # 初始化或加载模型
    config.cal_path.models_path.mkdir(exist_ok=True)

    # 检查数据平衡性（多标签分类）
    # 统计所有能级的正负样本总数
    positive_count = np.sum(y_train == 1)
    negative_count = np.sum(y_train == 0)

    # 计算每个能级的正样本比例
    per_level_positive_ratio = np.mean(y_train, axis=0)
    avg_positive_ratio = float(np.mean(per_level_positive_ratio))

    logger.info(f"多标签分类 - {n_correct_levels} 个能级")
    logger.info(
        f"训练集 - 正样本总数:{positive_count}, 负样本总数:{negative_count}, "
        f"平均正样本比例:{avg_positive_ratio:.4f}"
    )
    logger.info(
        f"各能级正样本比例: {np.array2string(per_level_positive_ratio, precision=4)}"
    )
    logger.info(f"描述符长度:{X_train.shape[1]}, 输出维度:{n_correct_levels}")

    # 模型初始化
    # 计算正类权重（用于处理不平衡数据）
    pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
    class_weights = [1.0, pos_weight]  # [负样本权重, 正样本权重]
    desired_architecture = _select_model_architecture(
        sample_count=len(X_train),
        positive_count=int(positive_count),
        logger=logger,
    )

    # CPU优化：减少hidden_size以降低计算量
    hidden_size = 96 if not torch.cuda.is_available() else 128

    if config.cal_settings.cal_loop_num == 1:
        # 第一轮：直接创建新模型
        model: ANNClassifier = ANNClassifier(
            input_size=X_train.shape[1],
            output_size=n_correct_levels,  # 自动根据 output_size>1 启用多标签分类
            hidden_size=hidden_size,
            learning_rate=0.001,
            class_weights=class_weights,
            model_architecture=desired_architecture,
            random_seed=config.model_params.random_state,
        )
        logger.info(
            f"创建新模型（{'多标签' if model.multi_label else '单标签'}分类，输出维度={n_correct_levels}），"
            f"类别权重: 负样本=1.0, 正样本={pos_weight:.1f}"
        )
        logger.info(
            f"模型hidden_size: {hidden_size} ({'CPU优化' if not torch.cuda.is_available() else 'GPU模式'})"
        )
    else:
        # 后续轮次：尝试加载之前的模型
        model_path = (
            config.cal_path.models_path
            / f"{config.target.conf}_{config.cal_settings.cal_loop_num - 1}.pt"
        )
        legacy_model_path = (
            config.cal_path.models_path
            / f"{config.target.conf}_{config.cal_settings.cal_loop_num - 1}.pkl"
        )
        if model_path.exists():
            model = ANNClassifier.load_model(str(model_path))
            logger.info(f"加载已有模型: {model_path}")
            # 验证模型输出维度和架构是否匹配
            if (
                model.output_size != n_correct_levels
                or model.model_architecture != desired_architecture
            ):
                logger.warning(
                    "加载的模型与当前训练需求不匹配："
                    f"输出维度 {model.output_size}->{n_correct_levels}, "
                    f"架构 {model.model_architecture}->{desired_architecture}，将创建新模型"
                )
                model = ANNClassifier(
                    input_size=X_train.shape[1],
                    output_size=n_correct_levels,
                    hidden_size=hidden_size,
                    learning_rate=0.001,
                    class_weights=class_weights,
                    model_architecture=desired_architecture,
                    random_seed=config.model_params.random_state,
                )
                logger.info(
                    f"创建新模型（输出维度={n_correct_levels}, 架构={desired_architecture}）"
                )
            else:
                logger.info(
                    f"已加载模型支持{'多标签' if model.multi_label else '单标签'}分类，"
                    f"输出维度={model.output_size}, 架构={model.model_architecture}"
                )
        elif legacy_model_path.exists():
            logger.warning(
                "检测到旧版 .pkl 模型文件，建议重新保存为 .pt checkpoint 格式"
            )
            import joblib

            model = joblib.load(legacy_model_path)
            logger.info(f"加载旧版模型: {legacy_model_path}")
            legacy_architecture = getattr(model, "model_architecture", "standard")
            if (
                model.output_size != n_correct_levels
                or legacy_architecture != desired_architecture
            ):
                logger.warning(
                    "加载的旧版模型与当前训练需求不匹配："
                    f"输出维度 {model.output_size}->{n_correct_levels}, "
                    f"架构 {legacy_architecture}->{desired_architecture}，将创建新模型"
                )
                model = ANNClassifier(
                    input_size=X_train.shape[1],
                    output_size=n_correct_levels,
                    hidden_size=hidden_size,
                    learning_rate=0.001,
                    class_weights=class_weights,
                    model_architecture=desired_architecture,
                    random_seed=config.model_params.random_state,
                )
                logger.info(
                    f"创建新模型（输出维度={n_correct_levels}, 架构={desired_architecture}）"
                )
            else:
                logger.info(
                    f"已加载旧版模型支持{'多标签' if model.multi_label else '单标签'}分类，"
                    f"输出维度={model.output_size}, 架构={legacy_architecture}"
                )
        else:
            # 模型文件不存在，创建新模型
            model = ANNClassifier(
                input_size=X_train.shape[1],
                output_size=n_correct_levels,
                hidden_size=hidden_size,
                learning_rate=0.001,
                class_weights=class_weights,
                model_architecture=desired_architecture,
                random_seed=config.model_params.random_state,
            )
            logger.info(
                f"创建新模型（{'多标签' if model.multi_label else '单标签'}分类，输出维度={n_correct_levels}），"
                f"类别权重: 负样本=1.0, 正样本={pos_weight:.1f}"
            )
            logger.info(
                f"模型hidden_size: {hidden_size} ({'CPU优化' if not torch.cuda.is_available() else 'GPU模式'})"
            )

    logger.info("使用累计的全部已标注样本训练 - 不进行重采样、不切验证集")
    logger.info(
        f"最终训练数据 - 正样本:{positive_count}, 负样本:{negative_count}, "
        f"平均正样本比例:{avg_positive_ratio:.4f}"
    )
    logger.info(f"累计已标注样本数:{len(X_train)}")
    logger.info("使用类别权重和损失函数来处理数据不平衡问题")

    # Model training (只训练一次)
    logger.info("             训练模型")

    # CPU训练优化配置
    if not torch.cuda.is_available():
        # 获取系统CPU核心数
        cpu_count = os.cpu_count() or 4  # 如果无法获取则默认使用4核

        # 从配置文件读取PyTorch线程数，如果未设置则使用默认值
        cpu_threads = config.server_settings.cpu_threads
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
        f"开始训练 - 数据量:{len(X_train):,}, 特征维度:{X_train.shape[1]}"
    )
    model.fit(
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        batch_size=batch_size_optimized,
        max_epochs=max_epochs_optimized,
        early_stopping_patience=999999,
        min_delta=0.0,
    )

    logger.info("训练完成，返回累计已标注样本供迭代评估与后续推理使用")

    return model, X_train, y_train


def evaluate_model(
        model: ANNClassifier,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        config: MLCalConfig,
        logger: logging.Logger,
    ) -> tuple[EvaluationResults, PredictionOutputs]:
    """
    对累计的全部已标注样本做训练内诊断评估。

    Args:
        model: 训练好的模型
        X_labeled: 截至当前轮累计的全部已标注描述符
        y_labeled: 与 X_labeled 对应的累计标签
        config: 配置对象
        logger: 日志记录器

    Returns:
        包含训练内诊断结果、概率和元数据的结果字典
    """

    logger.info("开始基于累计已标注样本的训练内诊断评估")

    start_time = time.time()
    y_prediction_labeled = model.predict(X_labeled)
    y_probability_labeled = model.predict_proba(X_labeled)
    eval_time = time.time() - start_time

    labeled_f1, labeled_roc_auc, labeled_accuracy, labeled_precision, labeled_recall = (
        ANNClassifier.model_evaluation(
            y_labeled, y_prediction_labeled, y_probability_labeled
        )
    )

    logger.info("累计已标注样本诊断结果:")
    logger.info(
        f"AUC: {labeled_roc_auc:.4f}, F1: {labeled_f1:.4f}, Accuracy: {labeled_accuracy:.4f}"
    )
    logger.info(
        f"Precision: {labeled_precision:.4f}, Recall: {labeled_recall:.4f}"
    )
    logger.info("注意：以上指标仅反映训练内拟合程度，不代表跨轮泛化能力")

    prediction_outputs: PredictionOutputs = {
        "y_prediction_labeled": y_prediction_labeled,
    }

    evaluation_results: EvaluationResults = {
        "probabilities": {
            "y_probability_labeled": y_probability_labeled,
            "y_probability_all": y_probability_labeled,
        },
        "true_labels": {"y_labeled": y_labeled},
        "labeled_metrics": {
            "f1": labeled_f1,
            "roc_auc": labeled_roc_auc,
            "accuracy": labeled_accuracy,
            "precision": labeled_precision,
            "recall": labeled_recall,
        },
        "metadata": {
            "eval_time": eval_time,
            "labeled_samples": len(y_labeled),
            "metric_scope": "in_sample",
        },
    }
    save_training_results(config, evaluation_results, logger)

    return evaluation_results, prediction_outputs


def predict_model(
    model: ANNClassifier,
    raw_csfs_descriptors: np.ndarray,
    caled_csfs_idxs_array: np.ndarray,
    correct_levels_ci_squared: np.ndarray,
    config: MLCalConfig,
    train_data_counts: MLDataCounts,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MLDataCounts]:
    # 获取未选择的CSF索引
    total_csfs_count = raw_csfs_descriptors.shape[0]
    all_csfs_idxs = np.arange(total_csfs_count, dtype=np.int64)
    current_calc_idxs = caled_csfs_idxs_array

    unselected_idxs = np.setdiff1d(all_csfs_idxs, current_calc_idxs)

    # 仅对未选择的CSF进行预测（使用分批处理避免内存溢出）
    X_unselected_for_prediction = raw_csfs_descriptors[unselected_idxs]
    y_unselected_probability = model.predict_proba_batch(
        X_unselected_for_prediction,
        batch_size=10_000_000,  # 可根据内存调整
    )
    # 多能级情况：在所有能级中取最大概率值，然后与阈值比较
    max_unselected_probability = np.max(y_unselected_probability, axis=1)
    y_unselected_prediction = (max_unselected_probability > 0.5).astype(int)

    logger.info(f"推理了 {len(y_unselected_probability)} 个未选择CSF组态")
    logger.info(
        "未选择CSF预测概率统计 - 最小值: %.4f, 最大值: %.4f, 平均值: %.4f",
        float(np.min(max_unselected_probability)),
        float(np.max(max_unselected_probability)),
        float(np.mean(max_unselected_probability)),
    )

    # 为绘图准备当前计算CSF的预测概率
    # 对当前计算的CSF也进行预测（用于绘图和分析）
    X_current_calc = raw_csfs_descriptors[current_calc_idxs]
    # 多能级情况：保留所有能级的预测概率，shape: (n_current_csfs, n_levels)
    y_current_cal_probability = model.predict_proba(X_current_calc)
    logger.info(f"当前计算CSF数量: {len(current_calc_idxs)}, 预测概率shape: {y_current_cal_probability.shape}")

    # 基于混合系数选择重要组态（已验证重要组态）
    cutoff_value = config.cal_settings.cutoff_value
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
    expansion_ratio = config.cal_settings.expansion_ratio
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
    sampling_ratio = config.cal_settings.sampling_ratio
    max_sampling_CSFs_num = math.ceil(total_csfs_count * sampling_ratio)
    if new_sampling_CSFs_num + current_important_count > max_sampling_CSFs_num:
        new_sampling_CSFs_num = max_sampling_CSFs_num - current_important_count
        logger.info(f"目标新增组态数超过最大选择数，调整为{new_sampling_CSFs_num}")

    if new_sampling_CSFs_num <= 0:
        logger.info("目标新增组态数为0，本轮不新增ML采样组态")
        ml_sampled_idxs = np.array([], dtype=np.int64)
        train_data_counts.important_csfs_count = verified_important_idxs.shape[0]
        train_data_counts.ml_predicted_count = ml_predicted_important_global_idxs.shape[0]
        train_data_counts.ml_sampled_count = ml_sampled_idxs.shape[0]
        return (
            ml_sampled_idxs,
            verified_important_idxs,
            y_current_cal_probability,
            train_data_counts,
        )

    if len(ml_predicted_important_local_idxs) >= new_sampling_CSFs_num:
        # 情况1：ML预测的重要组态数量充足，按概率排序选择top-k
        logger.info(f"ML预测组态充足，按概率排序选择前{new_sampling_CSFs_num}个")

        # 获取ML预测重要组态的最大概率（在所有能级中取最大值）
        ml_predicted_important_probabilities = max_unselected_probability[
            ml_predicted_important_local_idxs
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
        if len(ml_predicted_important_global_idxs) > 0:
            logger.info(
                f"ML预测组态不足，全部采用{len(ml_predicted_important_global_idxs)}个"
            )
            ml_sampled_idxs = ml_predicted_important_global_idxs
        else:
            logger.warning(
                "固定阈值 0.5 下未预测到任何重要组态，回退为按概率排序选择 top-%s",
                new_sampling_CSFs_num,
            )
            probability_sorted_idxs = np.argsort(max_unselected_probability)[::-1]
            top_k_local_idxs = probability_sorted_idxs[:new_sampling_CSFs_num]
            ml_sampled_idxs = unselected_idxs[top_k_local_idxs]
            logger.info(
                "回退采样完成：选取了 %s 个概率最高的未选择组态，最高概率=%.4f，最低入选概率=%.4f",
                len(ml_sampled_idxs),
                float(max_unselected_probability[top_k_local_idxs[0]])
                if len(top_k_local_idxs) > 0
                else 0.0,
                float(max_unselected_probability[top_k_local_idxs[-1]])
                if len(top_k_local_idxs) > 0
                else 0.0,
            )

    train_data_counts.important_csfs_count = verified_important_idxs.shape[0]
    train_data_counts.ml_predicted_count = ml_predicted_important_global_idxs.shape[0]
    train_data_counts.ml_sampled_count = ml_sampled_idxs.shape[0]

    return (
        ml_sampled_idxs,
        verified_important_idxs,
        y_current_cal_probability,
        train_data_counts,
    )


def handle_calculation_error(config: MLCalConfig, logger: logging.Logger) -> None:
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
