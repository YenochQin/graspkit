# -*- encoding: utf-8 -*-
"""
回归模型训练与预测模块

提供与 ml_trainer.py 接口一致的回归版训练/预测函数，直接预测 log₁₀(CI²) 值。

与分类版本的主要区别：
  - 标签：log₁₀(max(ci², min_clip_value))，连续值而非 0/1
  - 选择逻辑：按预测 log₁₀(CI²) 最大值排序，选 top-k
  - 无需 cutoff_value 阈值
"""

import logging
import math
import os
from typing import cast

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from ..data_IO import MLCalConfig
from ..utils.data_modules import MLDataCounts
from .ml_regression_model import ANNRegressor


def generate_regression_train_descriptors(
    raw_csfs_descriptors: np.ndarray,
    accumulated_idxs: np.ndarray,
    accumulated_ci_squared: np.ndarray,
    min_clip_value: float = 1e-15,
) -> np.ndarray:
    """
    生成回归用训练数据

    将 CI² 值转换为 log₁₀ 连续标签，保留完整的数值排序信息。

    Args:
        raw_csfs_descriptors: 原始 CSF 描述符，shape: (total_csfs, n_features)
        accumulated_idxs: 累积的 CSF 索引，shape: (n_accumulated,)
        accumulated_ci_squared: 对应的 CI 系数平方，shape: (n_levels, n_accumulated)
        min_clip_value: 对 CI² 进行下限截断，避免 log(0)，默认 1e-15

    Returns:
        训练数据矩阵，shape: (n_accumulated, n_features + n_levels)
        前 n_features 列为描述符特征，后 n_levels 列为 log₁₀(CI²) 标签
    """
    # 取对应索引的描述符
    sampled_csfs_descriptors = raw_csfs_descriptors[accumulated_idxs]

    # 截断 CI² 并取 log₁₀，转置为 (n_accumulated, n_levels)
    ci_squared_clipped = np.maximum(accumulated_ci_squared, min_clip_value)
    log_ci_labels = np.log10(ci_squared_clipped).T  # (n_accumulated, n_levels)

    # 拼接描述符和回归标签
    caled_csfs_descriptors = np.column_stack([sampled_csfs_descriptors, log_ci_labels])
    return caled_csfs_descriptors


def train_regression_model(
    train_data: np.ndarray,
    config: MLCalConfig,
    logger: logging.Logger | None = None,
) -> tuple[ANNRegressor, dict[str, float]]:
    """
    训练回归模型

    Args:
        train_data: 训练数据，shape: (n_samples, n_features + n_levels)
                   由 generate_regression_train_descriptors 生成
        config: ML 计算配置对象
        loop_num: 当前计算轮次（用于决定是否加载已有模型）
        logger: 日志记录器（可选）

    Returns:
        (model, metrics_dict)
        metrics_dict 包含: mae, rmse, spearman_rho, overfitting_gap,
                          train_mae, train_rmse, train_spearman_rho
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # 推断能级数量和特征维度
    n_levels = config.cal_settings.spectral_term.__len__()
    descriptor_features = train_data.shape[1] - n_levels

    X = train_data[:, :descriptor_features]
    y = train_data[:, descriptor_features:]  # (n_samples, n_levels)

    X_train, X_test, y_train, y_test = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        train_test_split(X, y, test_size=0.2, random_state=42),
    )

    logger.info(f"回归训练 - {n_levels} 个能级，描述符维度: {descriptor_features}")
    logger.info(
        f"训练集: {len(X_train)}, 测试集: {len(X_test)}"
    )
    logger.info(
        f"标签范围 - 最小: {y_train.min():.3f}, 最大: {y_train.max():.3f}, "
        f"均值: {y_train.mean():.3f}"
    )

    # CPU 优化
    hidden_size = 96 if not torch.cuda.is_available() else 128

    # 初始化模型（回归模型不支持加载历史分类模型，始终创建新模型）
    model = ANNRegressor(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        output_size=n_levels,
        learning_rate=0.001,
        huber_delta=1.0,
    )
    logger.info(
        f"创建 ANNRegressor（输出维度={n_levels}，hidden_size={hidden_size}，"
        f"{'CPU优化' if not torch.cuda.is_available() else 'GPU模式'}）"
    )

    # CPU 线程优化
    if not torch.cuda.is_available():
        cpu_count = os.cpu_count() or 4
        cpu_threads = getattr(config.server_settings, "cpu_threads", 16)
        if cpu_threads is not None:
            try:
                optimal_threads = min(int(cpu_threads), cpu_count)
            except (ValueError, TypeError):
                optimal_threads = min(32, cpu_count)
        else:
            optimal_threads = min(32, cpu_count)
        torch.set_num_threads(optimal_threads)
        os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
        os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
        logger.info(f"PyTorch 线程数: {optimal_threads}")
        batch_size_opt = 4096
        max_epochs_opt = 150
    else:
        batch_size_opt = 2048
        max_epochs_opt = 150

    # 从训练集切出验证集用于早停
    X_fit, X_val, y_fit, y_val = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        train_test_split(X_train, y_train, test_size=0.1, random_state=42),
    )

    logger.info(f"开始训练 - 拟合集: {len(X_fit)}, 验证集: {len(X_val)}")
    model.fit(
        X_fit,
        y_fit,
        X_val=X_val,
        y_val=y_val,
        batch_size=batch_size_opt,
        max_epochs=max_epochs_opt,
    )

    # 评估
    test_mae, test_rmse, test_rho = model.evaluate(X_test, y_test)
    train_mae, train_rmse, train_rho = model.evaluate(X_train, y_train)

    logger.info(f"测试集 - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, Spearman ρ: {test_rho:.4f}")
    logger.info(f"训练集 - MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, Spearman ρ: {train_rho:.4f}")

    overfitting_gap = train_rho - test_rho
    logger.info(f"过拟合检测（训练ρ - 测试ρ）: {overfitting_gap:.4f}")

    overfitting_threshold = getattr(config.ml_config, "overfitting_threshold", 0.1)
    if overfitting_gap > overfitting_threshold:
        logger.warning("检测到可能的过拟合现象（Spearman ρ 差异过大）")

    metrics: dict[str, float] = {
        "mae": test_mae,
        "rmse": test_rmse,
        "spearman_rho": test_rho,
        "overfitting_gap": overfitting_gap,
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_spearman_rho": train_rho,
    }

    return model, metrics


def evaluate_regression_model(
    model: ANNRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float]:
    """
    评估回归模型性能

    Args:
        model: 已训练的 ANNRegressor
        X_test: 测试描述符，shape: (n_samples, n_features)
        y_test: 真实 log₁₀(CI²) 标签，shape: (n_samples, n_levels)

    Returns:
        (mae, rmse, spearman_rho)
    """
    return model.evaluate(X_test, y_test)


def predict_regression_model(
    model: ANNRegressor,
    unselected_csf_descriptors: np.ndarray,
    unselected_idxs: np.ndarray,
    verified_important_idxs: np.ndarray,
    config: MLCalConfig,
    train_data_counts: MLDataCounts,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MLDataCounts]:
    """
    使用回归模型预测未计算 CSF 的 log₁₀(CI²)，按预测值排序选 top-k。

    接口与 predict_model() 保持一致，可直接替换使用。

    Args:
        model: 已训练的 ANNRegressor
        unselected_csf_descriptors: 未选择 CSF 的描述符，shape: (n_unselected, n_features)
        unselected_idxs: 未选择 CSF 在原始总池中的全局索引，shape: (n_unselected,)
        verified_important_idxs: 已验证重要 CSF 的全局索引，shape: (n_verified,)
        config: ML 计算配置
        train_data_counts: 训练数据计数对象（会更新并返回）
        logger: 日志记录器（可选）

    Returns:
        (ml_sampled_idxs, verified_important_idxs, y_predicted_log_ci, train_data_counts)
        - ml_sampled_idxs: ML 选出的新增 CSF 全局索引
        - verified_important_idxs: 已验证重要 CSF 全局索引（透传）
        - y_predicted_log_ci: 预测的 log₁₀(CI²) 值，shape: (n_unselected, n_levels)
        - train_data_counts: 更新后的计数对象
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # 对未选择的 CSF 进行预测（分批处理，避免内存溢出）
    y_predicted_log_ci = model.predict_batch(
        unselected_csf_descriptors,
        batch_size=10_000_000,
    )
    logger.info(f"完成对 {len(unselected_idxs)} 个未选择 CSF 的回归预测")

    # ============ 动态选择机制 ============
    current_important_count = len(verified_important_idxs)
    total_csfs_count = train_data_counts.total_csfs_count

    # 最小重要组态数量保护
    min_important_count = min(50, int(total_csfs_count * 0.01))
    if current_important_count <= min_important_count:
        current_important_count = min_important_count
        logger.info(f"重要组态数目小于最小值，调整为 {min_important_count}")

    expansion_ratio = getattr(config.cal_settings, "expansion_ratio", 2)
    new_target = math.ceil(expansion_ratio * current_important_count)

    # 设置上限
    sampling_ratio = getattr(config.cal_settings, "sampling_ratio", 0.085)
    max_sampling_num = math.ceil(total_csfs_count * sampling_ratio)
    if new_target + current_important_count > max_sampling_num:
        new_target = max_sampling_num - current_important_count
        logger.info(f"目标新增组态数超过最大选择数，调整为 {new_target}")

    # 按各能级预测 log₁₀(CI²) 的最大值降序排序
    # max 代表：只要在某个能级上预测值大，该 CSF 就值得被选入
    max_predicted_log_ci = np.max(y_predicted_log_ci, axis=1)  # (n_unselected,)
    ranked_local_idxs = np.argsort(max_predicted_log_ci)[::-1]  # 降序

    n_select = min(new_target, len(ranked_local_idxs))
    top_k_local_idxs = ranked_local_idxs[:n_select]
    ml_sampled_idxs = unselected_idxs[top_k_local_idxs]

    logger.info(f"当前已验证重要组态数: {current_important_count}")
    logger.info(f"目标新增 CSF 数: {new_target}")
    logger.info(f"实际选出 CSF 数: {len(ml_sampled_idxs)}")
    logger.info(
        f"预测 log₁₀(CI²) 范围 - 最大: {max_predicted_log_ci.max():.3f}, "
        f"最小: {max_predicted_log_ci.min():.3f}"
    )

    train_data_counts.important_csfs_count = current_important_count
    train_data_counts.ml_predicted_count = len(unselected_idxs)
    train_data_counts.ml_sampled_count = len(ml_sampled_idxs)

    return (
        ml_sampled_idxs,
        verified_important_idxs,
        y_predicted_log_ci,
        train_data_counts,
    )
