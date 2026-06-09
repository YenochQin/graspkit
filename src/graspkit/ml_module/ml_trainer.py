# -*- encoding: utf-8 -*-
# 标准库导入
from collections import deque
from collections.abc import Iterator
import heapq
import logging
import math
import os
import shutil
import time

# 第三方库导入
import numpy as np
from numpy.typing import NDArray
import polars as pl
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
from .ml_selection import (
    ContributionSource,
    log_reference_level_order_diagnostics,
    select_ml_candidate_indices,
)
from .ml_initializer import (
    combine_importance_and_reference_scores,
    compute_pairwise_gap_error_matrix,
    score_correction_candidates_from_ci,
)
from .streaming_descriptors import iter_indexed_descriptor_batches

# 本地模块导入
from .neural_network import CSFClassifier


def _is_hybrid_reference_ranking_enabled(
    config: MLCalConfig,
    selected_energy_data: pl.DataFrame | None,
) -> bool:
    """Check whether hybrid reference-energy ranking can be applied.

    Args:
        config: ML calculation configuration.
        selected_energy_data: Energy data for the currently selected levels.

    Returns:
        True when reference levels, selected energy data, and positive scoring
        weight are all configured.
    """
    return (
        selected_energy_data is not None
        and len(config.cal_settings.reference_energy_levels) > 1
        and getattr(config.cal_settings, "reference_energy_mode", "monitor")
        == "hybrid_rank"
        and getattr(config.cal_settings, "reference_energy_score_weight", 0.0) > 0
    )


def _write_candidate_hybrid_scores(
    unselected_idxs: np.ndarray,
    per_level_scores: np.ndarray,
    importance_scores: np.ndarray,
    correction_scores: np.ndarray,
    final_scores: np.ndarray,
    dominant_pairs: np.ndarray,
    selected_idxs: np.ndarray,
    config: MLCalConfig,
    logger: logging.Logger,
) -> None:
    """Write per-candidate hybrid ranking scores to CSV.

    Args:
        unselected_idxs: Global CSF indices for prediction candidates.
        per_level_scores: Per-level model probabilities.
        importance_scores: Model-derived base importance score per candidate.
        correction_scores: Reference-energy correction score per candidate.
        final_scores: Combined ranking score per candidate.
        dominant_pairs: Dominant reference-level pair indices per candidate.
        selected_idxs: Global CSF indices selected for the next calculation.
        config: ML calculation configuration with output paths.
        logger: Logger used for the save message.
    """
    spectral_term = config.cal_settings.spectral_term
    selected_idx_set = {int(idx) for idx in selected_idxs.tolist()}
    rows: list[dict[str, str | int | float | bool]] = []

    for row_idx, global_idx in enumerate(unselected_idxs.tolist()):
        dominant_pair = dominant_pairs[row_idx]
        if dominant_pair[0] >= 0 and dominant_pair[1] >= 0:
            top_pair = (
                f"{spectral_term[int(dominant_pair[0])]}|"
                f"{spectral_term[int(dominant_pair[1])]}"
            )
        else:
            top_pair = ""

        dominant_level_idx = int(np.argmax(per_level_scores[row_idx]))
        rows.append(
            {
                "csf_index": int(global_idx),
                "importance_score": float(importance_scores[row_idx]),
                "correction_score": float(correction_scores[row_idx]),
                "final_score": float(final_scores[row_idx]),
                "selected": int(global_idx) in selected_idx_set,
                "dominant_target_level": spectral_term[dominant_level_idx],
                "top_contributing_level_pair": top_pair,
            }
        )

    output_path = config.cal_path.results_path / "candidate_hybrid_scores.csv"
    pl.DataFrame(rows).write_csv(output_path)
    logger.info("候选CSF混合评分已保存到: %s", output_path)


def _select_model_architecture(
    current_loop_sample_count: int,
    accumulated_sample_count: int,
    positive_sample_count: int,
    logger: logging.Logger,
) -> str:
    """Select a robust classifier architecture for the current iteration.

    Args:
        current_loop_sample_count: Number of samples from the current loop.
        accumulated_sample_count: Number of samples accumulated across loops.
        positive_sample_count: Number of positive training labels.
        logger: Logger used for architecture-selection messages.

    Returns:
        Architecture name understood by ``CSFClassifier``: ``standard`` for
        low-coverage or low-positive-count data, otherwise ``cnn`` for
        CSF-sequence learning.
    """
    if current_loop_sample_count < 50_000 or positive_sample_count < 2_048:
        logger.info(
            "检测到小样本训练场景，使用 standard 架构以避免结构化模型在低覆盖率数据下塌缩"
        )
        return "standard"

    logger.info(
        "使用 cnn 架构进行训练（Bilous-style CSF序列卷积；本轮CSF数=%s, 累积样本数=%s）",
        current_loop_sample_count,
        accumulated_sample_count,
    )
    return "cnn"


def train_model(
    config: MLCalConfig,
    caled_csfs_descriptors: np.ndarray,
    correct_levels_ci: np.ndarray,
    logger: logging.Logger,
) -> tuple[CSFClassifier, np.ndarray, np.ndarray]:
    """Train the multi-label ANN classifier used for CSF selection.

    Args:
        config: ML calculation configuration.
        caled_csfs_descriptors: Descriptor matrix with label columns appended.
        correct_levels_ci: CI coefficients for the selected target levels.
        logger: Logger used for training diagnostics.

    Returns:
        Tuple of trained classifier, feature matrix used for training, and
        label matrix used for training.
    """

    # 数据提取：支持多标签分类
    # caled_csfs_descriptors 形状: (n_current_csfs, descriptor_features + n_correct_levels)
    # 前 descriptor_features 列是描述符特征，后 n_correct_levels 列是每个能级的标签

    # 从 correct_levels_ci 推断能级数量
    correct_levels_ci_2d = np.atleast_2d(correct_levels_ci)
    n_correct_levels = correct_levels_ci_2d.shape[0]
    current_loop_sample_count = correct_levels_ci_2d.shape[1]
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
    positive_sample_count = int(np.sum(np.any(y_train == 1, axis=1)))
    negative_sample_count = int(len(y_train) - positive_sample_count)

    # 计算每个能级的正样本比例
    per_level_positive_ratio = np.mean(y_train, axis=0)
    avg_positive_ratio = float(np.mean(per_level_positive_ratio))

    logger.info(f"多标签分类 - {n_correct_levels} 个能级")
    logger.info(
        f"训练集 - 正样本总数:{positive_count}, 负样本总数:{negative_count}, "
        f"平均正样本比例:{avg_positive_ratio:.4f}"
    )
    logger.info(
        "按CSF样本统计 - 正样本数:%s, 负样本数:%s, 样本总数:%s",
        positive_sample_count,
        negative_sample_count,
        len(X_train),
    )
    logger.info(
        f"各能级正样本比例: {np.array2string(per_level_positive_ratio, precision=4)}"
    )
    logger.info(f"描述符长度:{X_train.shape[1]}, 输出维度:{n_correct_levels}")
    if avg_positive_ratio > 0.35:
        logger.warning(
            "当前正样本比例过高(%.4f)，模型可能退化为几乎全正预测；建议提高 cutoff_value 或收紧正样本定义",
            avg_positive_ratio,
        )

    # 模型初始化
    # 计算正类权重（用于处理不平衡数据）
    pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
    class_weights = [1.0, pos_weight]  # [负样本权重, 正样本权重]
    desired_architecture = _select_model_architecture(
        current_loop_sample_count=current_loop_sample_count,
        accumulated_sample_count=len(X_train),
        positive_sample_count=positive_sample_count,
        logger=logger,
    )

    # CPU优化：减少hidden_size以降低计算量
    hidden_size = 96 if not torch.cuda.is_available() else 128
    model_params = getattr(config, "model_params", None)
    random_seed = getattr(model_params, "random_state", None)

    if config.cal_settings.cal_loop_num == 1:
        # 第一轮：直接创建新模型
        model: CSFClassifier = CSFClassifier(
            input_size=X_train.shape[1],
            output_size=n_correct_levels,  # 自动根据 output_size>1 启用多标签分类
            hidden_size=hidden_size,
            learning_rate=0.001,
            class_weights=class_weights,
            model_architecture=desired_architecture,
            random_seed=random_seed,
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
            model = CSFClassifier.load_model(str(model_path))
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
                model = CSFClassifier(
                    input_size=X_train.shape[1],
                    output_size=n_correct_levels,
                    hidden_size=hidden_size,
                    learning_rate=0.001,
                    class_weights=class_weights,
                    model_architecture=desired_architecture,
                    random_seed=random_seed,
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
                model = CSFClassifier(
                    input_size=X_train.shape[1],
                    output_size=n_correct_levels,
                    hidden_size=hidden_size,
                    learning_rate=0.001,
                    class_weights=class_weights,
                    model_architecture=desired_architecture,
                    random_seed=random_seed,
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
            model = CSFClassifier(
                input_size=X_train.shape[1],
                output_size=n_correct_levels,
                hidden_size=hidden_size,
                learning_rate=0.001,
                class_weights=class_weights,
                model_architecture=desired_architecture,
                random_seed=random_seed,
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
        model: CSFClassifier,
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
        CSFClassifier.model_evaluation(
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

    # -- 坍缩检测 --
    avg_positive_ratio = float(np.mean(y_labeled))
    is_collapsed = (
        abs(labeled_precision - avg_positive_ratio) < 0.02
        and labeled_recall > 0.95
    )
    if is_collapsed:
        logger.critical(
            "检测到模型坍缩（全正预测）: "
            "Precision=%.4f ≈ 正样本比例=%.4f, Recall=%.4f。"
            "建议在 config.toml [cal_settings] 中设置 use_regression_model = true。",
            labeled_precision, avg_positive_ratio, labeled_recall,
        )

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
    model: CSFClassifier,
    raw_csfs_descriptors: np.ndarray,
    caled_csfs_idxs_array: np.ndarray,
    correct_levels_ci_squared: np.ndarray,
    config: MLCalConfig,
    train_data_counts: MLDataCounts,
    logger: logging.Logger,
    selected_energy_data: pl.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MLDataCounts]:
    """Predict and select additional CSFs for the next calculation loop.

    Args:
        model: Trained ANN classifier.
        raw_csfs_descriptors: Descriptor matrix for all available CSFs.
        caled_csfs_idxs_array: Global indices already included in calculation.
        correct_levels_ci_squared: CI-square values for calculated CSFs and
            selected target levels.
        config: ML calculation configuration.
        train_data_counts: Mutable counters updated with selection statistics.
        logger: Logger used for prediction and selection diagnostics.
        selected_energy_data: Optional energy data used by hybrid reference
            ranking.

    Returns:
        Tuple containing newly sampled CSF indices, verified important CSF
        indices, current calculated CSF probabilities, and updated counters.
    """
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
    # 多能级情况：在所有能级中取最大概率值作为基础重要性分数
    importance_score = np.max(y_unselected_probability, axis=1)
    correction_score = np.zeros_like(importance_score, dtype=np.float64)
    final_score = importance_score.copy()
    dominant_pairs = np.full((len(unselected_idxs), 2), -1, dtype=np.int64)

    if _is_hybrid_reference_ranking_enabled(config, selected_energy_data):
        pairwise_gap_error_matrix = compute_pairwise_gap_error_matrix(
            selected_energy_data,
            config.cal_settings.spectral_term,
            config.cal_settings.reference_energy_levels,
        )
        correction_score, dominant_pairs = score_correction_candidates_from_ci(
            y_unselected_probability,
            pairwise_gap_error_matrix,
            pair_weighting=config.cal_settings.reference_gap_pair_weighting,
            top_pair_count=config.cal_settings.reference_energy_top_pair_count,
        )
        final_score, normalized_importance, normalized_correction = (
            combine_importance_and_reference_scores(
                importance_score,
                correction_score,
                importance_weight=config.cal_settings.reference_energy_importance_weight,
                correction_weight=config.cal_settings.reference_energy_score_weight,
            )
        )
        logger.info(
            "混合排序已启用: importance_weight=%.3f, correction_weight=%.3f",
            config.cal_settings.reference_energy_importance_weight,
            config.cal_settings.reference_energy_score_weight,
        )
        logger.info(
            "修正分数统计 - 最小值: %.4f, 最大值: %.4f, 平均值: %.4f",
            float(np.min(normalized_correction)),
            float(np.max(normalized_correction)),
            float(np.mean(normalized_correction)),
        )
        logger.info(
            "组合分数统计 - 最小值: %.4f, 最大值: %.4f, 平均值: %.4f",
            float(np.min(final_score)),
            float(np.max(final_score)),
            float(np.mean(final_score)),
        )
    else:
        normalized_importance = importance_score

    y_unselected_prediction = (final_score > 0.5).astype(int)

    logger.info(f"推理了 {len(y_unselected_probability)} 个未选择CSF组态")
    logger.info(
        "未选择CSF重要性分数统计 - 最小值: %.4f, 最大值: %.4f, 平均值: %.4f",
        float(np.min(importance_score)),
        float(np.max(importance_score)),
        float(np.mean(importance_score)),
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

    if getattr(config.cal_settings, "selection_mode", "fixed_ratio") == "fixed_ratio":
        ml_sampled_idxs, ml_predicted_count = _select_fixed_ratio_candidates_from_scores(
            unselected_idxs=unselected_idxs,
            final_score=final_score,
            predicted_mask=ml_predicted_important_mask,
            target_count=new_sampling_CSFs_num,
            logger=logger,
        )
    else:
        ml_sampled_idxs = select_ml_candidate_indices(
            unselected_idxs,
            per_level_scores=y_unselected_probability,
            final_score=final_score,
            target_count=new_sampling_CSFs_num,
            config=config,
            logger=logger,
            source=ContributionSource.CLASSIFIER_PROBABILITY,
            physical_contributions=None,
        )
        ml_predicted_count = int(ml_predicted_important_global_idxs.shape[0])

    train_data_counts.important_csfs_count = verified_important_idxs.shape[0]
    train_data_counts.ml_predicted_count = ml_predicted_count
    train_data_counts.ml_sampled_count = ml_sampled_idxs.shape[0]

    log_reference_level_order_diagnostics(
        selected_energy_data=selected_energy_data,
        spectral_terms=config.cal_settings.spectral_term,
        reference_energy_levels=config.cal_settings.reference_energy_levels,
        logger=logger,
    )

    _write_candidate_hybrid_scores(
        unselected_idxs=unselected_idxs,
        per_level_scores=y_unselected_probability,
        importance_scores=importance_score,
        correction_scores=correction_score,
        final_scores=final_score,
        dominant_pairs=dominant_pairs,
        selected_idxs=ml_sampled_idxs,
        config=config,
        logger=logger,
    )

    return (
        ml_sampled_idxs,
        verified_important_idxs,
        y_current_cal_probability,
        train_data_counts,
    )


def _push_top_scored_candidate(
    candidates: list[tuple[float, int, int]],
    score: float,
    global_idx: int,
    limit: int,
) -> None:
    if limit <= 0:
        return

    entry = (float(score), int(global_idx), int(global_idx))
    if len(candidates) < limit:
        heapq.heappush(candidates, entry)
    elif entry > candidates[0]:
        heapq.heapreplace(candidates, entry)


def _top_candidates_to_indices(candidates: list[tuple[float, int, int]]) -> np.ndarray:
    sorted_candidates = sorted(candidates, reverse=True)
    return np.array([global_idx for _, _, global_idx in sorted_candidates], dtype=np.int64)


def _select_fixed_ratio_candidates_from_scores(
    *,
    unselected_idxs: NDArray[np.int64],
    final_score: NDArray[np.float64],
    predicted_mask: NDArray[np.bool],
    target_count: int,
    logger: logging.Logger,
) -> tuple[NDArray[np.int64], int]:
    predicted_local_idxs = np.where(predicted_mask)[0]
    predicted_global_idxs = unselected_idxs[predicted_local_idxs]
    predicted_count = int(predicted_global_idxs.shape[0])

    if target_count <= 0:
        logger.info("目标新增组态数为0，本轮不新增ML采样组态")
        return np.array([], dtype=np.int64), predicted_count

    if len(predicted_local_idxs) >= target_count:
        logger.info(f"ML预测组态充足，按概率排序选择前{target_count}个")
        predicted_scores = final_score[predicted_local_idxs]
        sorted_idxs = np.argsort(predicted_scores)[::-1]
        top_k_local_idxs = predicted_local_idxs[sorted_idxs[:target_count]]
        selected = unselected_idxs[top_k_local_idxs]
        logger.info(f"从{len(predicted_local_idxs)}个ML预测重要组态中选择了{len(selected)}个")
        return selected.astype(np.int64, copy=False), predicted_count

    if predicted_count > 0:
        logger.info(f"ML预测组态不足，全部采用{predicted_count}个")
        return predicted_global_idxs.astype(np.int64, copy=False), predicted_count

    logger.warning(
        "固定阈值 0.5 下未预测到任何重要组态，回退为按概率排序选择 top-%s",
        target_count,
    )
    top_k_local_idxs = np.argsort(final_score)[::-1][:target_count]
    selected = unselected_idxs[top_k_local_idxs]
    logger.info(
        "回退采样完成：选取了 %s 个组合分数最高的未选择组态，最高分=%.4f，最低入选分=%.4f",
        len(selected),
        float(final_score[top_k_local_idxs[0]]) if len(top_k_local_idxs) > 0 else 0.0,
        float(final_score[top_k_local_idxs[-1]]) if len(top_k_local_idxs) > 0 else 0.0,
    )
    return selected.astype(np.int64, copy=False), predicted_count


def _iter_model_probability_batches(
    model: CSFClassifier, descriptor_batches: Iterator[NDArray[np.float32]]
) -> Iterator[NDArray[np.float64]]:
    if hasattr(model, "iter_predict_proba_batches"):
        yield from model.iter_predict_proba_batches(descriptor_batches)
        return

    for descriptor_batch in descriptor_batches:
        yield model.predict_proba(descriptor_batch)


def _predict_probability_batches_for_indices(
    model: CSFClassifier,
    raw_csfs_descriptors: pl.LazyFrame,
    indices: NDArray[np.int64],
    batch_size: int,
) -> Iterator[tuple[NDArray[np.int64], NDArray[np.float64]]]:
    pending_indices: deque[NDArray[np.int64]] = deque()

    def descriptor_batches() -> Iterator[NDArray[np.float32]]:
        for batch_indices, descriptor_batch in iter_indexed_descriptor_batches(
            raw_csfs_descriptors,
            indices=indices,
            batch_size=batch_size,
        ):
            pending_indices.append(batch_indices)
            yield descriptor_batch

    for probabilities in _iter_model_probability_batches(model, descriptor_batches()):
        if not pending_indices:
            raise ValueError("Model probability batch iterator yielded too many batches.")
        yield pending_indices.popleft(), np.asarray(probabilities)

    if pending_indices:
        raise ValueError("Model probability batch iterator yielded too few batches.")


def _iter_unselected_index_chunks(
    total_count: int,
    current_calc_idxs: NDArray[np.int64],
    chunk_size: int,
) -> Iterator[NDArray[np.int64]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    sorted_current_idxs = np.sort(current_calc_idxs.astype(np.int64, copy=False))
    sorted_current_idxs = sorted_current_idxs[
        (sorted_current_idxs >= 0) & (sorted_current_idxs < total_count)
    ]

    for start in range(0, total_count, chunk_size):
        stop = min(start + chunk_size, total_count)
        candidate_idxs = np.arange(start, stop, dtype=np.int64)

        left = np.searchsorted(sorted_current_idxs, start, side="left")
        right = np.searchsorted(sorted_current_idxs, stop, side="left")
        calculated_in_chunk = sorted_current_idxs[left:right]
        if calculated_in_chunk.size == 0:
            yield candidate_idxs
            continue

        keep_mask = np.ones(candidate_idxs.shape, dtype=bool)
        keep_mask[calculated_in_chunk - start] = False
        unselected_idxs = candidate_idxs[keep_mask]
        if unselected_idxs.size > 0:
            yield unselected_idxs


def _update_score_bounds(
    min_score: float | None,
    max_score: float | None,
    scores: NDArray[np.float64],
) -> tuple[float | None, float | None]:
    if scores.size == 0:
        return min_score, max_score
    batch_min = float(np.min(scores))
    batch_max = float(np.max(scores))
    return (
        batch_min if min_score is None else min(min_score, batch_min),
        batch_max if max_score is None else max(max_score, batch_max),
    )


def _normalize_scores_from_bounds(
    scores: NDArray[np.float64],
    min_score: float | None,
    max_score: float | None,
) -> NDArray[np.float64]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    if min_score is None or max_score is None:
        return np.zeros_like(scores, dtype=np.float64)
    if np.isclose(min_score, max_score):
        if max_score <= 0:
            return np.zeros_like(scores, dtype=np.float64)
        return np.ones_like(scores, dtype=np.float64)
    return (scores - min_score) / (max_score - min_score)


def _combine_scores_from_global_bounds(
    importance_scores: NDArray[np.float64],
    correction_scores: NDArray[np.float64],
    importance_min: float | None,
    importance_max: float | None,
    correction_min: float | None,
    correction_max: float | None,
    importance_weight: float,
    correction_weight: float,
) -> NDArray[np.float64]:
    total_weight = importance_weight + correction_weight
    if total_weight <= 0:
        raise ValueError("importance_weight 和 correction_weight 的和必须大于 0")

    normalized_importance = _normalize_scores_from_bounds(
        importance_scores,
        importance_min,
        importance_max,
    )
    normalized_correction = _normalize_scores_from_bounds(
        correction_scores,
        correction_min,
        correction_max,
    )
    return (
        importance_weight * normalized_importance
        + correction_weight * normalized_correction
    ) / total_weight


def _predict_model_streaming_hybrid(
    model: CSFClassifier,
    raw_csfs_descriptors: pl.LazyFrame,
    caled_csfs_idxs_array: np.ndarray,
    correct_levels_ci_squared: np.ndarray,
    config: MLCalConfig,
    train_data_counts: MLDataCounts,
    logger: logging.Logger,
    selected_energy_data: pl.DataFrame,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MLDataCounts]:
    total_csfs_count = int(raw_csfs_descriptors.select(pl.len()).collect().item())
    current_calc_idxs = caled_csfs_idxs_array.astype(np.int64, copy=False)

    cutoff_value = config.cal_settings.cutoff_value
    csfs_above_threshold_idxs = np.where(
        np.any(correct_levels_ci_squared >= np.float64(cutoff_value), axis=0)
    )[0]
    verified_important_idxs = caled_csfs_idxs_array[csfs_above_threshold_idxs]
    logger.info(f"已验证重要组态数: {len(verified_important_idxs)}")

    verified_important_ci_coefficients = correct_levels_ci_squared[
        :, csfs_above_threshold_idxs
    ]
    logger.info(
        f"已提取 {verified_important_ci_coefficients.shape[1]} 个重要组态的CI系数，维度: {verified_important_ci_coefficients.shape}"
    )

    current_important_count = len(verified_important_idxs)
    min_important_count = min(50, int(total_csfs_count * 0.01))
    if current_important_count <= min_important_count:
        current_important_count = min_important_count
        logger.info(f"重要组态数目小于等于最小值，调整为{min_important_count}")

    expansion_ratio = config.cal_settings.expansion_ratio
    new_sampling_CSFs_num = math.ceil(expansion_ratio * current_important_count)
    sampling_ratio = config.cal_settings.sampling_ratio
    max_sampling_CSFs_num = math.ceil(total_csfs_count * sampling_ratio)
    if new_sampling_CSFs_num + current_important_count > max_sampling_CSFs_num:
        new_sampling_CSFs_num = max_sampling_CSFs_num - current_important_count
        logger.info(f"目标新增组态数超过最大选择数，调整为{new_sampling_CSFs_num}")

    y_current_cal_probability_batches = [
        probabilities
        for _, probabilities in _predict_probability_batches_for_indices(
            model,
            raw_csfs_descriptors,
            indices=current_calc_idxs,
            batch_size=batch_size,
        )
    ]
    if y_current_cal_probability_batches:
        y_current_cal_probability = np.vstack(y_current_cal_probability_batches)
    else:
        output_size = int(getattr(model, "output_size", 1))
        y_current_cal_probability = np.empty((0, output_size), dtype=float)
    logger.info(
        f"当前计算CSF数量: {len(current_calc_idxs)}, 预测概率shape: {y_current_cal_probability.shape}"
    )

    pairwise_gap_error_matrix = compute_pairwise_gap_error_matrix(
        selected_energy_data,
        config.cal_settings.spectral_term,
        config.cal_settings.reference_energy_levels,
    )
    importance_weight = config.cal_settings.reference_energy_importance_weight
    correction_weight = config.cal_settings.reference_energy_score_weight
    if importance_weight + correction_weight <= 0:
        raise ValueError("importance_weight 和 correction_weight 的和必须大于 0")

    importance_min: float | None = None
    importance_max: float | None = None
    correction_min: float | None = None
    correction_max: float | None = None
    raw_importance_sum = 0.0
    raw_importance_min: float | None = None
    raw_importance_max: float | None = None
    normalized_correction_sum = 0.0
    normalized_final_sum = 0.0
    unselected_count = 0

    for unselected_idx_chunk in _iter_unselected_index_chunks(
        total_count=total_csfs_count,
        current_calc_idxs=current_calc_idxs,
        chunk_size=batch_size,
    ):
        for global_indices, probabilities in _predict_probability_batches_for_indices(
            model,
            raw_csfs_descriptors,
            indices=unselected_idx_chunk,
            batch_size=batch_size,
        ):
            if probabilities.ndim == 1:
                probabilities = probabilities.reshape(-1, 1)

            importance_scores = np.max(probabilities, axis=1)
            correction_scores, _ = score_correction_candidates_from_ci(
                probabilities,
                pairwise_gap_error_matrix,
                pair_weighting=config.cal_settings.reference_gap_pair_weighting,
                top_pair_count=config.cal_settings.reference_energy_top_pair_count,
            )
            importance_min, importance_max = _update_score_bounds(
                importance_min,
                importance_max,
                importance_scores,
            )
            correction_min, correction_max = _update_score_bounds(
                correction_min,
                correction_max,
                correction_scores,
            )
            raw_importance_min, raw_importance_max = _update_score_bounds(
                raw_importance_min,
                raw_importance_max,
                importance_scores,
            )
            raw_importance_sum += float(np.sum(importance_scores))
            unselected_count += len(global_indices)

    positive_candidates: list[tuple[float, int, int]] = []
    positive_chunks: list[NDArray[np.int64]] = []
    fallback_candidates: list[tuple[float, int, int]] = []
    ml_predicted_count = 0
    final_min_score: float | None = None
    final_max_score: float | None = None
    normalized_correction_min: float | None = None
    normalized_correction_max: float | None = None
    selection_limit = max(new_sampling_CSFs_num, 0)

    for unselected_idx_chunk in _iter_unselected_index_chunks(
        total_count=total_csfs_count,
        current_calc_idxs=current_calc_idxs,
        chunk_size=batch_size,
    ):
        for global_indices, probabilities in _predict_probability_batches_for_indices(
            model,
            raw_csfs_descriptors,
            indices=unselected_idx_chunk,
            batch_size=batch_size,
        ):
            if probabilities.ndim == 1:
                probabilities = probabilities.reshape(-1, 1)

            importance_scores = np.max(probabilities, axis=1)
            correction_scores, _ = score_correction_candidates_from_ci(
                probabilities,
                pairwise_gap_error_matrix,
                pair_weighting=config.cal_settings.reference_gap_pair_weighting,
                top_pair_count=config.cal_settings.reference_energy_top_pair_count,
            )
            final_scores = _combine_scores_from_global_bounds(
                importance_scores,
                correction_scores,
                importance_min,
                importance_max,
                correction_min,
                correction_max,
                importance_weight,
                correction_weight,
            )
            normalized_correction = _normalize_scores_from_bounds(
                correction_scores,
                correction_min,
                correction_max,
            )

            ml_predicted_mask = final_scores > 0.5
            batch_predicted_count = int(np.count_nonzero(ml_predicted_mask))
            ml_predicted_count += batch_predicted_count
            normalized_correction_sum += float(np.sum(normalized_correction))
            normalized_final_sum += float(np.sum(final_scores))
            normalized_correction_min, normalized_correction_max = _update_score_bounds(
                normalized_correction_min,
                normalized_correction_max,
                normalized_correction,
            )
            final_min_score, final_max_score = _update_score_bounds(
                final_min_score,
                final_max_score,
                final_scores,
            )

            if (
                selection_limit > 0
                and batch_predicted_count > 0
                and ml_predicted_count <= selection_limit
            ):
                positive_chunks.append(global_indices[ml_predicted_mask])
            elif ml_predicted_count > selection_limit:
                positive_chunks.clear()

            for global_idx, score in zip(global_indices, final_scores, strict=True):
                if score > 0.5 and selection_limit > 0:
                    _push_top_scored_candidate(
                        positive_candidates,
                        float(score),
                        int(global_idx),
                        selection_limit,
                    )
                if selection_limit > 0:
                    _push_top_scored_candidate(
                        fallback_candidates,
                        float(score),
                        int(global_idx),
                        selection_limit,
                    )

    logger.info(
        "混合排序已启用: importance_weight=%.3f, correction_weight=%.3f",
        importance_weight,
        correction_weight,
    )
    mean_correction = normalized_correction_sum / unselected_count if unselected_count else 0.0
    mean_final = normalized_final_sum / unselected_count if unselected_count else 0.0
    logger.info(
        "修正分数统计 - 最小值: %.4f, 最大值: %.4f, 平均值: %.4f",
        normalized_correction_min if normalized_correction_min is not None else 0.0,
        normalized_correction_max if normalized_correction_max is not None else 0.0,
        mean_correction,
    )
    logger.info(
        "组合分数统计 - 最小值: %.4f, 最大值: %.4f, 平均值: %.4f",
        final_min_score if final_min_score is not None else 0.0,
        final_max_score if final_max_score is not None else 0.0,
        mean_final,
    )

    logger.info(f"推理了 {unselected_count} 个未选择CSF组态")
    mean_importance = raw_importance_sum / unselected_count if unselected_count else 0.0
    logger.info(
        "未选择CSF重要性分数统计 - 最小值: %.4f, 最大值: %.4f, 平均值: %.4f",
        raw_importance_min if raw_importance_min is not None else 0.0,
        raw_importance_max if raw_importance_max is not None else 0.0,
        mean_importance,
    )

    logger.info("      组态采样")
    logger.info("更新重要组态索引")
    logger.info(f"开始选择组态，当前重要组态数为：{len(verified_important_idxs)}")
    logger.info(f"ML预测的重要组态数（在未选择中）：({ml_predicted_count},)")
    logger.info(f"目标新增组态数：{new_sampling_CSFs_num}")
    if getattr(config.cal_settings, "selection_mode", "fixed_ratio") == "cumulative_contribution":
        logger.warning(
            "classification probability is not a physical cumulative contribution; "
            "falling back to fixed_ratio/top-k selection"
        )

    if new_sampling_CSFs_num <= 0:
        logger.info("目标新增组态数为0，本轮不新增ML采样组态")
        ml_sampled_idxs = np.array([], dtype=np.int64)
    elif ml_predicted_count >= new_sampling_CSFs_num:
        logger.info(f"ML预测组态充足，按概率排序选择前{new_sampling_CSFs_num}个")
        ml_sampled_idxs = _top_candidates_to_indices(positive_candidates)
        logger.info(
            f"从{ml_predicted_count}个ML预测重要组态中选择了{len(ml_sampled_idxs)}个"
        )
    elif ml_predicted_count > 0:
        logger.info(f"ML预测组态不足，全部采用{ml_predicted_count}个")
        ml_sampled_idxs = np.concatenate(positive_chunks).astype(np.int64, copy=False)
    else:
        logger.warning(
            "固定阈值 0.5 下未预测到任何重要组态，回退为按概率排序选择 top-%s",
            new_sampling_CSFs_num,
        )
        ml_sampled_idxs = _top_candidates_to_indices(fallback_candidates)
        selected_scores = [score for score, _, _ in sorted(fallback_candidates, reverse=True)]
        logger.info(
            "回退采样完成：选取了 %s 个组合分数最高的未选择组态，最高分=%.4f，最低入选分=%.4f",
            len(ml_sampled_idxs),
            selected_scores[0] if selected_scores else 0.0,
            selected_scores[-1] if selected_scores else 0.0,
        )

    train_data_counts.important_csfs_count = verified_important_idxs.shape[0]
    train_data_counts.ml_predicted_count = ml_predicted_count
    train_data_counts.ml_sampled_count = ml_sampled_idxs.shape[0]
    logger.info(
        "流式混合排序跳过 candidate_hybrid_scores.csv: "
        "为保持内存有界，不缓存全部候选CSF评分"
    )

    return (
        ml_sampled_idxs,
        verified_important_idxs,
        y_current_cal_probability,
        train_data_counts,
    )


def predict_model_streaming(
    model: CSFClassifier,
    raw_csfs_descriptors: pl.LazyFrame,
    caled_csfs_idxs_array: np.ndarray,
    correct_levels_ci_squared: np.ndarray,
    config: MLCalConfig,
    train_data_counts: MLDataCounts,
    logger: logging.Logger,
    selected_energy_data: pl.DataFrame | None = None,
    batch_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MLDataCounts]:
    """Streaming variant of ``predict_model`` for lazy descriptor sources."""
    if _is_hybrid_reference_ranking_enabled(config, selected_energy_data):
        assert selected_energy_data is not None
        return _predict_model_streaming_hybrid(
            model,
            raw_csfs_descriptors,
            caled_csfs_idxs_array,
            correct_levels_ci_squared,
            config,
            train_data_counts,
            logger,
            selected_energy_data,
            batch_size,
        )

    total_csfs_count = int(raw_csfs_descriptors.select(pl.len()).collect().item())
    current_calc_idxs = caled_csfs_idxs_array.astype(np.int64, copy=False)

    cutoff_value = config.cal_settings.cutoff_value
    csfs_above_threshold_idxs = np.where(
        np.any(correct_levels_ci_squared >= np.float64(cutoff_value), axis=0)
    )[0]
    verified_important_idxs = caled_csfs_idxs_array[csfs_above_threshold_idxs]
    logger.info(f"已验证重要组态数: {len(verified_important_idxs)}")

    verified_important_ci_coefficients = correct_levels_ci_squared[
        :, csfs_above_threshold_idxs
    ]
    logger.info(
        f"已提取 {verified_important_ci_coefficients.shape[1]} 个重要组态的CI系数，维度: {verified_important_ci_coefficients.shape}"
    )

    current_important_count = len(verified_important_idxs)
    min_important_count = min(50, int(total_csfs_count * 0.01))
    if current_important_count <= min_important_count:
        current_important_count = min_important_count
        logger.info(f"重要组态数目小于等于最小值，调整为{min_important_count}")

    expansion_ratio = config.cal_settings.expansion_ratio
    new_sampling_CSFs_num = math.ceil(expansion_ratio * current_important_count)
    sampling_ratio = config.cal_settings.sampling_ratio
    max_sampling_CSFs_num = math.ceil(total_csfs_count * sampling_ratio)
    if new_sampling_CSFs_num + current_important_count > max_sampling_CSFs_num:
        new_sampling_CSFs_num = max_sampling_CSFs_num - current_important_count
        logger.info(f"目标新增组态数超过最大选择数，调整为{new_sampling_CSFs_num}")

    y_current_cal_probability_batches = [
        probabilities
        for _, probabilities in _predict_probability_batches_for_indices(
            model,
            raw_csfs_descriptors,
            indices=current_calc_idxs,
            batch_size=batch_size,
        )
    ]
    if y_current_cal_probability_batches:
        y_current_cal_probability = np.vstack(y_current_cal_probability_batches)
    else:
        output_size = int(getattr(model, "output_size", 1))
        y_current_cal_probability = np.empty((0, output_size), dtype=float)
    logger.info(
        f"当前计算CSF数量: {len(current_calc_idxs)}, 预测概率shape: {y_current_cal_probability.shape}"
    )

    positive_candidates: list[tuple[float, int, int]] = []
    positive_chunks: list[NDArray[np.int64]] = []
    fallback_candidates: list[tuple[float, int, int]] = []
    ml_predicted_count = 0
    unselected_count = 0
    predicted_score_sum = 0.0
    predicted_min_score: float | None = None
    predicted_max_score: float | None = None
    selection_limit = max(new_sampling_CSFs_num, 0)

    for unselected_idx_chunk in _iter_unselected_index_chunks(
        total_count=total_csfs_count,
        current_calc_idxs=current_calc_idxs,
        chunk_size=batch_size,
    ):
        for global_indices, probabilities in _predict_probability_batches_for_indices(
            model,
            raw_csfs_descriptors,
            indices=unselected_idx_chunk,
            batch_size=batch_size,
        ):
            if probabilities.ndim == 1:
                probabilities = probabilities.reshape(-1, 1)

            scores = np.max(probabilities, axis=1)
            ml_predicted_mask = scores > 0.5
            batch_predicted_count = int(np.count_nonzero(ml_predicted_mask))
            ml_predicted_count += batch_predicted_count
            unselected_count += len(global_indices)
            predicted_score_sum += float(np.sum(scores))
            if scores.size > 0:
                batch_min = float(np.min(scores))
                batch_max = float(np.max(scores))
                predicted_min_score = (
                    batch_min
                    if predicted_min_score is None
                    else min(predicted_min_score, batch_min)
                )
                predicted_max_score = (
                    batch_max
                    if predicted_max_score is None
                    else max(predicted_max_score, batch_max)
                )

            if (
                selection_limit > 0
                and batch_predicted_count > 0
                and ml_predicted_count <= selection_limit
            ):
                positive_chunks.append(global_indices[ml_predicted_mask])
            elif ml_predicted_count > selection_limit:
                positive_chunks.clear()

            for global_idx, score in zip(global_indices, scores, strict=True):
                if score > 0.5 and selection_limit > 0:
                    _push_top_scored_candidate(
                        positive_candidates,
                        float(score),
                        int(global_idx),
                        selection_limit,
                    )
                if selection_limit > 0:
                    _push_top_scored_candidate(
                        fallback_candidates,
                        float(score),
                        int(global_idx),
                        selection_limit,
                    )

    logger.info(f"推理了 {unselected_count} 个未选择CSF组态")
    mean_score = predicted_score_sum / unselected_count if unselected_count else 0.0
    logger.info(
        "未选择CSF重要性分数统计 - 最小值: %.4f, 最大值: %.4f, 平均值: %.4f",
        predicted_min_score if predicted_min_score is not None else 0.0,
        predicted_max_score if predicted_max_score is not None else 0.0,
        mean_score,
    )

    logger.info("      组态采样")
    logger.info("更新重要组态索引")
    logger.info(f"开始选择组态，当前重要组态数为：{len(verified_important_idxs)}")
    logger.info(f"ML预测的重要组态数（在未选择中）：({ml_predicted_count},)")
    logger.info(f"目标新增组态数：{new_sampling_CSFs_num}")
    if getattr(config.cal_settings, "selection_mode", "fixed_ratio") == "cumulative_contribution":
        logger.warning(
            "classification probability is not a physical cumulative contribution; "
            "falling back to fixed_ratio/top-k selection"
        )

    if new_sampling_CSFs_num <= 0:
        logger.info("目标新增组态数为0，本轮不新增ML采样组态")
        ml_sampled_idxs = np.array([], dtype=np.int64)
        train_data_counts.important_csfs_count = verified_important_idxs.shape[0]
        train_data_counts.ml_predicted_count = ml_predicted_count
        train_data_counts.ml_sampled_count = ml_sampled_idxs.shape[0]
        return (
            ml_sampled_idxs,
            verified_important_idxs,
            y_current_cal_probability,
            train_data_counts,
        )

    if ml_predicted_count >= new_sampling_CSFs_num:
        logger.info(f"ML预测组态充足，按概率排序选择前{new_sampling_CSFs_num}个")
        ml_sampled_idxs = _top_candidates_to_indices(positive_candidates)
        logger.info(
            f"从{ml_predicted_count}个ML预测重要组态中选择了{len(ml_sampled_idxs)}个"
        )
    elif ml_predicted_count > 0:
        logger.info(f"ML预测组态不足，全部采用{ml_predicted_count}个")
        ml_sampled_idxs = np.concatenate(positive_chunks).astype(np.int64, copy=False)
    else:
        logger.warning(
            "固定阈值 0.5 下未预测到任何重要组态，回退为按概率排序选择 top-%s",
            new_sampling_CSFs_num,
        )
        ml_sampled_idxs = _top_candidates_to_indices(fallback_candidates)
        selected_scores = [score for score, _, _ in sorted(fallback_candidates, reverse=True)]
        logger.info(
            "回退采样完成：选取了 %s 个组合分数最高的未选择组态，最高分=%.4f，最低入选分=%.4f",
            len(ml_sampled_idxs),
            selected_scores[0] if selected_scores else 0.0,
            selected_scores[-1] if selected_scores else 0.0,
        )

    train_data_counts.important_csfs_count = verified_important_idxs.shape[0]
    train_data_counts.ml_predicted_count = ml_predicted_count
    train_data_counts.ml_sampled_count = ml_sampled_idxs.shape[0]

    return (
        ml_sampled_idxs,
        verified_important_idxs,
        y_current_cal_probability,
        train_data_counts,
    )


def handle_calculation_error(config: MLCalConfig, logger: logging.Logger) -> None:
    """Record a calculation error and update continuation state.

    Args:
        config: ML calculation configuration.
        logger: Logger used for error-handling messages.
    """
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
