"""
神经网络模块 - 提供多种架构的人工神经网络分类器

本模块包含：
1. TensorNet: 张量网络架构，适合处理结构化特征
2. ANNClassifier: 优化的人工神经网络分类器
"""

import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _set_random_seed(seed: int) -> None:
    """设置所有相关库的随机种子以保证可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TensorNet(nn.Module):
    """
    张量网络架构 - 用于处理具有结构特征的输入

    这个架构特别适合处理原子物理中的组态函数数据，
    其中特征可以按照子壳层进行结构化排列。

    输入形状: (batch_size, seq_length, in_channels)
    其中 seq_length * in_channels = 总特征数
    """

    def __init__(
        self, input_shape: tuple[int, int], hidden_dim: int = 128, num_classes: int = 2
    ):
        """
        初始化 TensorNet

        Args:
            input_shape: (seq_length, in_channels) 输入张量形状
            hidden_dim: 隐藏层维度
            num_classes: 输出类别数
        """
        super(TensorNet, self).__init__()
        seq_length, in_channels = input_shape
        self.w1 = nn.Parameter(torch.zeros(seq_length, in_channels, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(hidden_dim, in_channels))
        self.w2 = nn.Parameter(torch.zeros(in_channels))
        self.b2 = nn.Parameter(torch.zeros(1))
        self.w3 = nn.Parameter(torch.zeros(hidden_dim, num_classes))
        self.b3 = nn.Parameter(torch.zeros(num_classes))
        self.relu = nn.ReLU()

        self.input_shape = input_shape

        # 自动初始化权重
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w3)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)
        nn.init.zeros_(self.b3)
        nn.init.normal_(self.w2, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量，形状 (batch_size, seq_length, in_channels)

        Returns:
            输出张量，形状 (batch_size, num_classes)
        """
        x = x.reshape(-1, self.input_shape[0], self.input_shape[1])
        a1 = torch.einsum("ijk,jkl->ilk", x, self.w1) + self.b1
        a2 = torch.einsum("ijk,k->ij", self.relu(a1), self.w2) + self.b2
        a3 = torch.matmul(self.relu(a2), self.w3) + self.b3
        return a3


class ANNClassifier:
    """
    优化的人工神经网络分类器

    主要改进：
    1. 支持多种模型架构（标准全连接、TensorNet张量网络）
    2. 添加类型提示和完整文档
    3. 改进错误处理和验证
    4. 支持早停和学习率调度
    5. 优化内存使用和性能（批处理预测）
    6. 增强日志记录
    7. 改进代码结构和可读性
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 150,
        output_size: int = 2,
        learning_rate: float = 0.001,
        class_weights: list[float] | None = None,
        device: str | None = None,
        use_dynamic_weights: bool = True,
        model_architecture: Literal["standard", "tensornet"] = "standard",
        tensor_channels: int = 3,
        random_seed: int | None = None,
        multi_label: bool | None = None,
    ):
        """
        初始化ANN分类器

        Args:
            input_size: 输入特征数量
            hidden_size: 隐藏层神经元数量（仅standard架构使用）
            output_size: 输出维度
                - output_size = 1: 单标签二分类
                - output_size > 1: 多标签分类（output_size 等于能级数）
            learning_rate: 学习率
            class_weights: 类别权重用于处理不平衡数据（如果use_dynamic_weights为True则忽略）
            device: 计算设备，如果为None则自动选择
            use_dynamic_weights: 是否使用动态权重计算（基于训练数据中正负样本比例）
            model_architecture: 模型架构类型 ("standard" 标准全连接 或 "tensornet" 张量网络)
            tensor_channels: TensorNet的通道数（必须能整除input_size）
            random_seed: 随机种子（用于可重复性）
            multi_label: 是否使用多标签分类（None时自动判断：output_size > 1 时启用）
        """
        # 设置随机种子
        if random_seed is not None:
            _set_random_seed(random_seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.use_dynamic_weights = use_dynamic_weights
        self.model_architecture = model_architecture
        self.tensor_channels = tensor_channels

        # 自动判断是否使用多标签分类
        # 对于本任务（CSF重要性检测），每个能级都是独立的二分类标签
        # 因此应该始终使用多标签分类（BCEWithLogitsLoss），无论 output_size 是多少
        # output_size = 1: 单个能级，但仍使用多标签模式（1个标签）
        # output_size > 1: 多个能级，使用多标签模式（多个标签）
        if multi_label is None:
            self.multi_label = True  # 始终使用多标签模式
        else:
            self.multi_label = multi_label

        # 验证 TensorNet 参数
        if model_architecture == "tensornet":
            if input_size % tensor_channels != 0:
                raise ValueError(
                    f"对于 TensorNet 架构，input_size ({input_size}) "
                    f"必须能被 tensor_channels ({tensor_channels}) 整除"
                )

        # 设备配置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 类别权重处理
        if not use_dynamic_weights:
            # 使用静态权重
            if class_weights is None:
                class_weights = [9.0, 1.0]  # 默认权重
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(
                self.device
            )
        else:
            self.class_weights = None  # 将在训练时动态计算

        # 构建模型
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # 初始化损失函数（权重将在训练时设置）
        if self.multi_label:
            # 多标签分类：使用 BCEWithLogitsLoss
            # pos_weight 可以用来处理不平衡数据（每个标签一个权重）
            if self.class_weights is not None:
                # 对于多标签，class_weights 应该是正类权重
                pos_weight = torch.tensor(
                    [self.class_weights[1]], dtype=torch.float32
                ).to(self.device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        else:
            # 单标签分类：使用 CrossEntropyLoss
            if self.class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()

        # 训练历史记录
        self.training_history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        self.logger = logging.getLogger(__name__)

    def _build_model(self) -> nn.Module:
        """构建神经网络模型"""
        if self.model_architecture == "tensornet":
            # TensorNet 架构
            seq_length = self.input_size // self.tensor_channels
            model = TensorNet(
                input_shape=(seq_length, self.tensor_channels),
                hidden_dim=self.hidden_size,
                num_classes=self.output_size,
            ).to(self.device)
        else:
            # 标准全连接架构
            model = nn.Sequential(
                nn.Linear(self.input_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.LayerNorm(self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_size // 2, self.output_size),
            ).to(self.device)
            # 初始化权重
            self._initialize_weights(model)

        return model

    def _initialize_weights(self, model: nn.Module) -> None:
        """初始化模型权重"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        batch_size: int = 2048,
        max_epochs: int = 150,
        early_stopping_patience: int = 20,
        min_delta: float = 1e-4,
    ) -> dict[str, list[float]]:
        """
        训练ANN模型

        Args:
            X_train: 训练数据
            y_train: 训练标签
            X_val: 验证数据（可选）
            y_val: 验证标签（可选）
            batch_size: 批次大小
            max_epochs: 最大训练轮数
            early_stopping_patience: 早停耐心值
            min_delta: 最小改进阈值

        Returns:
            训练历史记录字典
        """
        # 数据验证
        self._validate_input_data(X_train, y_train)

        # 动态计算权重（如果启用）
        if self.use_dynamic_weights:
            pos_weight_tensor = self._calculate_dynamic_weights(y_train)
            # 重新创建损失函数
            if self.multi_label:
                # 多标签分类：使用 BCEWithLogitsLoss
                # pos_weight_tensor 形状为 (N_roots,)，直接用于 BCEWithLogitsLoss
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            else:
                # 单标签分类：使用 CrossEntropyLoss
                # 从 pos_weight_tensor 提取权重构建 [负类权重, 正类权重]
                neg_weight = 1.0
                pos_weight = pos_weight_tensor[0].item()  # 取第一个能级的权重
                class_weights = torch.tensor(
                    [neg_weight, pos_weight], dtype=torch.float32
                ).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 数据转换
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        # 多标签分类需要 float32 类型，单标签需要 long 类型
        y_dtype = torch.float32 if self.multi_label else torch.long
        y_train_tensor = torch.tensor(y_train, dtype=y_dtype).to(self.device)

        # 验证数据处理
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=y_dtype).to(self.device)
        else:
            X_val_tensor = y_val_tensor = None

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        # 早停机制
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        # 训练循环
        self.model.train()
        val_loss = val_accuracy = 0.0

        for epoch in range(max_epochs):
            epoch_loss = self._train_epoch(X_train_tensor, y_train_tensor, batch_size)
            self.training_history["train_loss"].append(epoch_loss)

            # 验证评估
            if X_val_tensor is not None and y_val_tensor is not None:
                val_loss, val_accuracy = self._validate_epoch(
                    X_val_tensor, y_val_tensor
                )
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_accuracy"].append(val_accuracy)

                # 学习率调度
                scheduler.step(val_loss)

                # 早停检查
                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break

            # 日志记录
            if (epoch + 1) % 50 == 0:
                log_msg = f"Epoch [{epoch + 1}/{max_epochs}], Loss: {epoch_loss:.6f}"
                if X_val_tensor is not None:
                    log_msg += (
                        f", Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy:.4f}"
                    )
                self.logger.info(log_msg)

        return self.training_history

    def _train_epoch(
        self, X_train: torch.Tensor, y_train: torch.Tensor, batch_size: int
    ) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        permutation = torch.randperm(X_train.size(0))

        for i in range(0, X_train.size(0), batch_size):
            idxs = permutation[i : i + batch_size]
            batch_X, batch_y = X_train[idxs], y_train[idxs]

            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate_epoch(
        self, X_val: torch.Tensor, y_val: torch.Tensor
    ) -> tuple[float, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            outputs = self.model(X_val)
            loss = self.criterion(outputs, y_val)

            _, predicted = torch.max(outputs.data, 1)
            total = y_val.size(0)
            correct = (predicted == y_val).sum().item()

            total_loss = loss.item()
            accuracy = correct / total

        return total_loss, accuracy

    def _calculate_dynamic_weights(
        self,
        y_train: np.ndarray,
        focus_on_recall: bool = True,
        min_positive_weight: float = 3.0,
        adaptive_strength: float = 0.5,
    ) -> torch.Tensor:
        """
        针对"找出重要组态"任务优化的权重计算 (适配多能级/多输出 BCEWithLogitsLoss)

        修改说明：
        1. 移除了 bincount，改用 sum(dim=0) 以支持二维标签矩阵。
        2. 返回值调整为 pos_weight 向量，直接用于 BCEWithLogitsLoss。
        3. 所有逻辑改为向量化操作，为每个能级(Root)单独计算权重。

        Args:
            y_train: 训练标签 (N_samples, N_roots)
            focus_on_recall: 是否优先保证召回率
            min_positive_weight: 正类的最小权重倍数
            adaptive_strength: 自适应调整强度

        Returns:
            pos_weight: 张量，形状为 (N_roots,)，用于 BCEWithLogitsLoss
        """
        # 1. 数据转换与统计 (支持二维矩阵)
        # y_train shape: [Batch_Size, N_roots]
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        n_samples = y_tensor.size(0)
        n_roots = y_tensor.size(1) if y_tensor.dim() > 1 else 1

        # 容错处理：如果变成一维了，强制unsqueeze
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(1)

        # 按列求和：获得每个能级的正样本数
        # shape: [N_roots]
        pos_counts = y_tensor.sum(dim=0)
        neg_counts = n_samples - pos_counts

        # 防止除零
        pos_counts = torch.clamp(pos_counts, min=1.0)

        # 计算比例
        pos_ratios = pos_counts / n_samples
        neg_ratios = 1.0 - pos_ratios

        if focus_on_recall:
            # --- 向量化逻辑开始 ---

            # 创建结果容器
            calculated_weights = torch.zeros_like(pos_ratios)

            # 情况 A: 早期迭代/少数类 (Ratio <= 0.5)
            # 逻辑：pos_weight = 1/ratio, 然后应用 adaptive_strength
            mask_early = pos_ratios <= 0.5
            if mask_early.any():
                raw_weight = 1.0 / (pos_ratios[mask_early] + 1e-6)
                # 自适应调整: base + (target - base) * strength
                # 这里 base=1.0 (无加权), target=raw_weight
                calculated_weights[mask_early] = (
                    1.0 + (raw_weight - 1.0) * adaptive_strength
                )

            # 情况 B: 后期迭代/多数类 (Ratio > 0.5)
            # 逻辑：权重衰减，但保持 >= min_positive_weight
            mask_late = ~mask_early
            if mask_late.any():
                # 衰减因子: 0 (当ratio=0.5) -> 1 (当ratio=1.0)
                decay_factor = (pos_ratios[mask_late] - 0.5) / 0.5

                # 计算当前权重
                # 你的逻辑: pos_weight = base * (1 - 0.3 * decay * strength)
                current_weights = min_positive_weight * (
                    1.0 - 0.3 * decay_factor * adaptive_strength
                )

                # 确保不低于最小值
                calculated_weights[mask_late] = torch.max(
                    current_weights, torch.tensor(min_positive_weight)
                )

            # 最终赋予 pos_weight
            pos_weight_tensor = calculated_weights

        else:
            # 标准平衡模式: num_neg / num_pos
            pos_weight_tensor = neg_counts / pos_counts

        # 移动到设备
        pos_weight_tensor = pos_weight_tensor.to(self.device)

        # --- 详细日志 (针对多能级优化显示) ---
        # 计算一些统计量用于展示
        avg_pos_ratio = pos_ratios.mean().item()
        min_w = pos_weight_tensor.min().item()
        max_w = pos_weight_tensor.max().item()
        avg_w = pos_weight_tensor.mean().item()

        self.logger.info(
            f"\n{'=' * 70}\n"
            f"权重计算 v2 (BCE-MultiLabel) - 重要组态检测模式\n"
            f"{'=' * 70}\n"
            f"  数据统计:\n"
            f"    总样本数:     {n_samples:,}\n"
            f"    能级数量:     {n_roots}\n"
            f"    平均正类占比: {avg_pos_ratio:.1%}\n"
            f"\n"
            f"  权重配置 (pos_weight):\n"
            f"    Min 权重:     {min_w:.3f}\n"
            f"    Max 权重:     {max_w:.3f}\n"
            f"    Avg 权重:     {avg_w:.3f}\n"
            f"\n"
            f"  策略说明:\n"
            f"    {'✓ 高召回率模式' if focus_on_recall else '○ 平衡模式'}\n"
            f"    自适应强度:   {adaptive_strength:.1f}\n"
            f"    策略逻辑:     Ratio<=0.5用反比增强, Ratio>0.5用衰减保护\n"
            f"{'=' * 70}"
        )

        return pos_weight_tensor

    def _validate_input_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """验证输入数据的有效性"""
        if X.shape[0] != y.shape[0]:
            raise ValueError("X和y的样本数量不匹配")

        if X.shape[1] != self.input_size:
            raise ValueError(
                f"输入特征维度 {X.shape[1]} 与模型期望的 {self.input_size} 不匹配"
            )

        if self.multi_label:
            # 多标签分类：检查标签维度
            if y.ndim != 2:
                raise ValueError(f"多标签分类期望二维标签数组，但收到 {y.ndim} 维数组")
            if y.shape[1] != self.output_size:
                raise ValueError(
                    f"标签维度 {y.shape[1]} 与模型输出维度 {self.output_size} 不匹配"
                )
            # 检查标签值是否为 0/1
            unique_vals = np.unique(y)
            if not np.all((unique_vals == 0) | (unique_vals == 1)):
                raise ValueError(f"多标签分类期望标签值为 0/1，但收到 {unique_vals}")
        else:
            # 单标签分类：检查类别数
            if len(np.unique(y)) > self.output_size:
                raise ValueError(
                    f"标签类别数 {len(np.unique(y))} 超过模型输出维度 {self.output_size}"
                )

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """预测类别"""
        predictions = self.predict_proba(X)
        if self.multi_label:
            # 多标签分类：每个标签独立 sigmoid，返回 0/1 标签
            return (predictions > threshold).astype(int)
        else:
            # 单标签分类：softmax 概率，取正类（索引1）
            # 注意：当 output_size=1 时，也应该使用 multi_label 模式
            # 这里保留是为了兼容性
            if predictions.shape[1] == 1:
                # 只有 1 个输出，直接使用 sigmoid
                return (predictions > threshold).astype(int)
            return (predictions[:, 1] > threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        Args:
            X: 输入数据

        Returns:
            预测概率
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            model_outputs = self.model(X_tensor)
            if self.multi_label:
                # 多标签分类：每个标签独立的 sigmoid 概率
                outputs = torch.sigmoid(model_outputs)
            else:
                # 单标签分类：softmax 概率
                outputs = torch.softmax(model_outputs, dim=1)

        return outputs.cpu().numpy()

    def predict_proba_batch(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        批处理预测概率 - 适合处理大数据集避免内存溢出

        Args:
            X: 输入数据
            batch_size: 批次大小

        Returns:
            预测概率数组
        """
        self.model.eval()
        num_samples = X.shape[0]
        all_proba = []

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_X = X[i : min(i + batch_size, num_samples)]
                X_tensor = torch.tensor(batch_X, dtype=torch.float32).to(self.device)
                model_outputs = self.model(X_tensor)
                if self.multi_label:
                    outputs = torch.sigmoid(model_outputs)
                else:
                    outputs = torch.softmax(model_outputs, dim=1)
                all_proba.append(outputs.cpu().numpy())

        return np.vstack(all_proba) if len(all_proba) > 1 else all_proba[0]

    def predict_batch(
        self, X: np.ndarray, batch_size: int = 1024, threshold: float = 0.5
    ) -> np.ndarray:
        """
        批处理预测类别 - 适合处理大数据集避免内存溢出

        Args:
            X: 输入数据
            batch_size: 批次大小
            threshold: 分类阈值

        Returns:
            预测类别数组
        """
        proba = self.predict_proba_batch(X, batch_size)
        return (proba[:, 1] > threshold).astype(int)

    def evaluate(
        self, X: np.ndarray, y: np.ndarray, verbose: bool = True
    ) -> dict[str, float]:
        """
        评估模型性能

        Args:
            X: 输入数据
            y: 真实标签
            verbose: 是否打印评估结果

        Returns:
            评估指标字典
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            if self.multi_label:
                # 多标签分类：使用 sigmoid 和阈值
                y_probability = torch.sigmoid(outputs).cpu().numpy()
                y_pred = (y_probability > 0.5).astype(int)
            else:
                # 单标签分类：使用 argmax
                y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
                y_probability = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

        if self.multi_label:
            # 多标签评估指标（使用 average='samples' 计算每个样本的平均指标）
            metrics = {
                "accuracy": accuracy_score(y.flatten(), y_pred.flatten()),
                "f1_score": f1_score(y, y_pred, average="samples"),
                "precision": precision_score(y, y_pred, average="samples"),
                "recall": recall_score(y, y_pred, average="samples"),
            }
            # 注意：ROC AUC 在多标签情况下计算方式不同，这里简化处理
        else:
            # 单标签评估指标
            metrics = {
                "accuracy": accuracy_score(y, y_pred),
                "f1_score": f1_score(y, y_pred),
                "precision": precision_score(y, y_pred),
                "recall": recall_score(y, y_pred),
                "roc_auc": roc_auc_score(y, y_probability),
            }

        if verbose:
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")

        return metrics

    def plot_roc_curve(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        绘制ROC曲线

        Args:
            X: 输入数据
            y: 真实标签
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            if self.multi_label:
                # 多标签分类：使用 sigmoid
                y_probability = torch.sigmoid(outputs).cpu().numpy()
            else:
                # 单标签分类：使用 softmax
                y_probability = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

        if not self.multi_label:
            fpr, tpr, _ = roc_curve(y, y_probability)
            plt.figure()
            plt.plot(
                fpr,
                tpr,
                label=f"ROC Curve (AUC = {roc_auc_score(y, y_probability):.4f})",
            )
            plt.title("ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.show()
        else:
            # 多标签 ROC 曲线需要特殊处理，这里简化处理
            self.logger.warning("多标签分类的 ROC 曲线绘制尚未实现")

    def save_model(self, path: str) -> None:
        """保存模型完整状态"""
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_history": self.training_history,
            "hyperparameters": {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "output_size": self.output_size,
                "learning_rate": self.learning_rate,
                "model_architecture": self.model_architecture,
                "tensor_channels": self.tensor_channels,
                "multi_label": self.multi_label,
            },
        }
        torch.save(save_dict, path)

    def load_model(self, path: str, device: str | None = None) -> "ANNClassifier":
        """加载模型"""
        checkpoint = torch.load(path, map_location=device or self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_history = checkpoint["training_history"]
        self.model.to(self.device)
        return self

    @staticmethod
    def plot_curve(
        cal_mix_coeff_List: np.ndarray,
        y_probability_all: np.ndarray,
        y_test: np.ndarray,
        y_probability: np.ndarray,
        filename: str,
        level_title: str = "Ci Values vs Predicted Probability",
    ):
        """
        绘制评估曲线（支持多标签分类）

        Args:
            cal_mix_coeff_List: 混合系数列表
            y_probability_all: 所有概率预测
            y_test: 测试标签
            y_probability: 测试概率
            filename: 保存文件名
            level_title: 能级标题（用于多能级图表标题）

        Returns:
            ROC AUC和PR AUC
        """
        plt.figure(figsize=(10, 8))

        # 处理多标签分类：展平数据
        if y_test.ndim == 2:
            # 多标签分类：展平所有标签和概率
            y_test_flat = y_test.flatten()
            # 对于多标签，y_probability 形状为 (n_samples, n_labels)
            if y_probability.ndim == 2:
                y_prob_flat = y_probability.flatten()
            else:
                y_prob_flat = y_probability
        else:
            # 单标签分类
            y_test_flat = y_test
            y_prob_flat = y_probability

        # 绘制 ROC 曲线
        fpr, tpr, _ = roc_curve(y_test_flat, y_prob_flat)
        roc_auc = roc_auc_score(y_test_flat, y_prob_flat)
        plt.subplot(2, 2, 1)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        # 绘制 PR 曲线
        precision, recall, _ = precision_recall_curve(y_test_flat, y_prob_flat)
        pr_auc = auc(recall, precision)
        plt.subplot(2, 2, 2)
        plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
        plt.title("PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")

        # 绘制混淆矩阵
        # 对于多标签，需要使用展平的数据
        y_pred = np.where(y_prob_flat > 0.5, 1, 0)
        cm = confusion_matrix(y_test_flat, y_pred)
        plt.subplot(2, 2, 3)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        # 绘制可解释性曲线 - Ci值与预测概率的关系
        plt.subplot(2, 2, 4)

        # 数据维度检查和处理
        if len(cal_mix_coeff_List) == 0 or len(y_probability_all) == 0:
            plt.text(
                0.5,
                0.5,
                "No data to plot",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title(level_title)
        elif len(cal_mix_coeff_List) != len(y_probability_all):
            plt.text(
                0.5,
                0.5,
                f"Data length mismatch:\nMix coeff: {len(cal_mix_coeff_List)}\nProbability: {len(y_probability_all)}",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
                fontsize=10,
            )
            plt.title(f"{level_title} (Data Mismatch)")
        else:
            # 数据长度匹配，正常绘图
            if y_probability_all.ndim == 2 and y_probability_all.shape[1] >= 2:
                prob_values = y_probability_all[:, 1]
            elif y_probability_all.ndim == 1:
                prob_values = y_probability_all
            else:
                prob_values = y_probability_all.flatten()

            # 确保混合系数不为零，避免log(0)
            mix_coeff_values = cal_mix_coeff_List.copy()
            zero_mask = mix_coeff_values == 0
            mix_coeff_values = np.where(zero_mask, 1e-10, mix_coeff_values)

            # 过滤掉无效值
            valid_mask = (
                (prob_values >= 0)
                & (prob_values <= 1)
                & (~np.isnan(prob_values))
                & (~np.isnan(mix_coeff_values))
            )

            if np.sum(valid_mask) > 0:
                prob_valid = prob_values[valid_mask]
                mix_coeff_valid = mix_coeff_values[valid_mask]

                plt.scatter(
                    prob_valid, np.log(np.abs(mix_coeff_valid)), alpha=0.6, s=20
                )

                zero_count = np.sum(zero_mask)
                total_count = len(cal_mix_coeff_List)

                plt.xlabel("Predicted Probability")
                plt.ylabel("Log|Ci Values|")
                plt.title(f"{level_title}\n(n={total_count}, zeros={zero_count})")
                plt.grid(True, alpha=0.3)
            else:
                plt.text(
                    0.5,
                    0.5,
                    "No valid data points",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                )
                plt.title(f"{level_title} (No Valid Data)")

        plt.tight_layout()

        # 确保父目录存在
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)
        plt.close()

        return roc_auc, pr_auc

    @staticmethod
    def model_evaluation(
        y_test: np.ndarray, y_pred: np.ndarray, y_probability: np.ndarray
    ):
        """
        模型评估

        Args:
            y_test: 测试集真实标签
            y_pred: 预测标签
            y_probability: 预测概率

        Returns:
            f1, roc_auc, accuracy, precision, recall
        """
        try:
            # 根据标签维度选择合适的 average 参数
            if y_test.ndim == 2:
                # 多标签分类：使用 'micro' 或 'macro'
                avg = "micro"
                # 多标签ROC AUC：直接传入2D数组，指定average参数
                roc_auc = roc_auc_score(y_test, y_probability, average=avg)
            else:
                # 二元分类：使用 'binary'
                avg = "binary"
                roc_auc = roc_auc_score(y_test, y_probability)

            f1 = f1_score(y_test, y_pred, average=avg)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average=avg)
            recall = recall_score(y_test, y_pred, average=avg)
        except Exception as e:
            logging.getLogger(__name__).warning(f"评估指标计算出错: {e}")
            f1 = roc_auc = accuracy = precision = recall = 0.0
            if len(y_test) > 0:
                accuracy = accuracy_score(y_test, y_pred)

        return f1, roc_auc, accuracy, precision, recall

    @staticmethod
    def resampling(
        X_train: np.ndarray, y_train: np.ndarray, weight: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        数据重采样（上采样）- 增加少数类样本

        Args:
            X_train: 训练数据
            y_train: 训练标签
            weight: [正类权重, 负类权重] 控制采样比例

        Returns:
            重采样后的数据和标签
        """
        X_positive = X_train[y_train == 1]
        X_negative = X_train[y_train == 0]
        Y_positive = y_train[y_train == 1]
        Y_negative = y_train[y_train == 0]

        Num_min = max(X_positive.shape[0], X_negative.shape[0])

        N_sample_positive = int(weight[0] * Num_min)
        N_sample_negative = int(weight[1] * Num_min)
        id_positive = np.random.randint(0, X_positive.shape[0], N_sample_positive)
        id_negative = np.random.randint(0, X_negative.shape[0], N_sample_negative)

        X_resampled = np.concatenate(
            (X_positive[id_positive], X_negative[id_negative]), axis=0
        )
        y_resampled = np.concatenate(
            (Y_positive[id_positive], Y_negative[id_negative]), axis=0
        )

        # 打乱顺序
        shuffle_index = np.random.permutation(X_resampled.shape[0])
        X_resampled = X_resampled[shuffle_index]
        y_resampled = y_resampled[shuffle_index]
        return X_resampled, y_resampled

    @staticmethod
    def downsampling(
        X_train: np.ndarray, y_train: np.ndarray, weight: list[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        数据下采样 - 减少多数类样本

        Args:
            X_train: 训练数据
            y_train: 训练标签
            weight: [正类权重, 负类权重] 控制采样比例

        Returns:
            下采样后的数据和标签
        """
        X_positive = X_train[y_train == 1]
        X_negative = X_train[y_train == 0]
        Y_positive = y_train[y_train == 1]
        Y_negative = y_train[y_train == 0]

        Num_max = min(X_positive.shape[0], X_negative.shape[0])

        N_sample_positive = int(weight[0] * Num_max)
        N_sample_negative = int(weight[1] * Num_max)
        id_positive = np.random.choice(
            X_positive.shape[0], N_sample_positive, replace=False
        )
        id_negative = np.random.choice(
            X_negative.shape[0], N_sample_negative, replace=False
        )

        X_resampled = np.concatenate(
            (X_positive[id_positive], X_negative[id_negative]), axis=0
        )
        y_resampled = np.concatenate(
            (Y_positive[id_positive], Y_negative[id_negative]), axis=0
        )

        # 打乱顺序
        shuffle_index = np.random.permutation(X_resampled.shape[0])
        X_resampled = X_resampled[shuffle_index]
        y_resampled = y_resampled[shuffle_index]
        return X_resampled, y_resampled

    @staticmethod
    def print_gpu_memory() -> None:
        """打印GPU内存使用情况"""
        if not torch.cuda.is_available():
            print("CUDA not available")
            return

        # 总内存
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # 当前分配的内存
        allocated_memory = torch.cuda.memory_allocated(0)
        # 缓存的内存
        cached_memory = torch.cuda.memory_reserved(0)

        # 转换为 GiB 并打印
        print(f"GPU 总内存: {total_memory / 1024**3:.2f} GiB")
        print(f"已分配内存: {allocated_memory / 1024**3:.2f} GiB")
        print(f"缓存内存: {cached_memory / 1024**3:.2f} GiB")
        print(f"可用内存: {(total_memory - allocated_memory) / 1024**3:.2f} GiB")

    @staticmethod
    def predict_in_batches(
        model: "ANNClassifier",
        X: np.ndarray,
        batch_size: int = 4000000,
        predict_proba: bool = False,
        proba_index: int = 1,
    ) -> np.ndarray:
        """
        超快速分批预测函数，支持普通预测和概率预测

        Args:
            model: 训练好的模型
            X: 待预测数据
            batch_size: 每批处理的数据量（较大值通常更快）
            predict_proba: 是否使用概率预测
            proba_index: 概率预测中要提取的列索引（默认是1，表示正类概率）

        Returns:
            完整的预测结果
        """
        n_samples = X.shape[0]
        result = []

        # 确定预测方法
        predict_func = model.predict_proba if predict_proba else model.predict

        # 一次性处理小数据集
        if n_samples <= batch_size:
            pred = predict_func(X)
            return pred[:, proba_index] if predict_proba else pred

        # 分批处理大数据集
        for i in range(0, n_samples, batch_size):
            batch_X = X[i : i + batch_size]
            batch_pred = predict_func(batch_X)

            # 如果是概率预测且需要提取特定列
            if predict_proba and proba_index is not None:
                batch_pred = batch_pred[:, proba_index]

            result.append(batch_pred)

        # 合并结果
        if len(result[0].shape) > 1:  # 多维数组
            return np.vstack(result)
        else:  # 一维数组
            return np.concatenate(result)

    def get_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        method: Literal["permutation", "gradient"] = "permutation",
    ) -> np.ndarray:
        """
        计算特征重要性

        Args:
            X: 输入数据
            y: 真实标签（仅 permutation 方法需要，推荐提供以获得准确结果）
            method: 计算方法 ('permutation' 或 'gradient')

        Returns:
            特征重要性数组
        """
        if method == "permutation":
            return self._permutation_importance(X, y)
        elif method == "gradient":
            return self._gradient_importance(X)
        else:
            raise ValueError("方法必须是 'permutation' 或 'gradient'")

    def _permutation_importance(
        self, X: np.ndarray, y: np.ndarray | None = None, n_repeats: int = 10
    ) -> np.ndarray:
        """置换重要性 - 通过随机打乱特征来评估其重要性"""
        y_pred = self.predict(X)
        # 如果没有提供 y，使用预测结果作为基准（这不是最佳实践，但保持向后兼容）
        if y is None:
            self.logger.warning(
                "未提供真实标签 y，使用预测结果计算置换重要性。"
                "建议提供真实标签以获得准确的重要性评估。"
            )
            baseline_score = self.evaluate(X, y_pred, verbose=False)["accuracy"]
        else:
            baseline_score = self.evaluate(X, y, verbose=False)["accuracy"]

        importance_scores = []

        for feature_idx in range(X.shape[1]):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                X_permuted[:, feature_idx] = np.random.permutation(
                    X_permuted[:, feature_idx]
                )
                y_pred_permuted = self.predict(X_permuted)
                if y is None:
                    permuted_score = self.evaluate(
                        X_permuted, y_pred_permuted, verbose=False
                    )["accuracy"]
                else:
                    permuted_score = self.evaluate(X_permuted, y, verbose=False)[
                        "accuracy"
                    ]
                scores.append(baseline_score - permuted_score)
            importance_scores.append(np.mean(scores))

        return np.array(importance_scores)

    def _gradient_importance(self, X: np.ndarray) -> np.ndarray:
        """梯度重要性"""
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(
            self.device
        )

        outputs = self.model(X_tensor)
        # 对于二分类，使用正类的输出
        if self.output_size == 2:
            target_output = outputs[:, 1].sum()
        else:
            target_output = outputs.sum()

        target_output.backward()

        # 计算梯度的绝对值作为重要性
        if X_tensor.grad is not None:
            gradients = X_tensor.grad.abs().mean(dim=0)
            return gradients.cpu().numpy()
        else:
            return np.zeros(X.shape[1])
