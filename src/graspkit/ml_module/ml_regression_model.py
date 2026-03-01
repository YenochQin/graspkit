# -*- encoding: utf-8 -*-
"""
回归神经网络模块 - 直接预测 log₁₀(CI²) 值用于 CSF 重要性排序

与 ANNClassifier 的主要区别：
- 输出层无激活（linear output）
- 损失函数：HuberLoss(delta=1.0)，对异常值鲁棒
- 评估指标：MAE / RMSE / Spearman ρ
- 标签范围：[-15, 0]（log₁₀ CI²），保留完整数值排序信息
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split


def _set_random_seed(seed: int) -> None:
    """设置所有相关库的随机种子以保证可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ANNRegressor:
    """
    人工神经网络回归器，直接预测 log₁₀(CI²) 值

    与 ANNClassifier 相比：
    1. 输出层无激活函数（线性输出）
    2. 使用 HuberLoss 代替 BCEWithLogitsLoss
    3. 评估指标为 MAE、RMSE 和 Spearman 排名相关系数
    4. 无需预先设定 cutoff_value 阈值即可对 CSF 排名
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 150,
        output_size: int = 1,
        learning_rate: float = 0.001,
        huber_delta: float = 1.0,
        device: str | None = None,
        random_seed: int | None = None,
    ):
        """
        初始化 ANN 回归器

        Args:
            input_size: 输入特征数量
            hidden_size: 隐藏层神经元数量
            output_size: 输出维度（等于能级数量）
            learning_rate: 学习率
            huber_delta: HuberLoss 的 delta 参数，控制对异常值的敏感度
            device: 计算设备，None 时自动选择
            random_seed: 随机种子（用于可重复性）
        """
        if random_seed is not None:
            _set_random_seed(random_seed)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.huber_delta = huber_delta

        # 设备配置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 构建模型（与 ANNClassifier standard 架构相同，输出层无激活）
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # HuberLoss 对异常值鲁棒（delta=1.0 时，误差<1用MSE，误差>=1用MAE）
        self.criterion = nn.HuberLoss(delta=huber_delta)

        # 训练历史记录
        self.training_history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_mae": [],
        }

        self.logger = logging.getLogger(__name__)

    def _build_model(self) -> nn.Module:
        """构建标准全连接回归网络（输出层无激活函数）"""
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
            # 无激活：直接输出 log₁₀(CI²) 预测值
        ).to(self.device)
        self._initialize_weights(model)
        return model

    def _initialize_weights(self, model: nn.Module) -> None:
        """初始化模型权重（Xavier 均匀分布）"""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_input_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """验证输入数据的有效性"""
        if X.shape[0] != y.shape[0]:
            raise ValueError("X 和 y 的样本数量不匹配")
        if X.shape[1] != self.input_size:
            raise ValueError(
                f"输入特征维度 {X.shape[1]} 与模型期望的 {self.input_size} 不匹配"
            )
        if y.ndim != 2:
            raise ValueError(
                f"回归标签期望二维数组 (n_samples, n_levels)，但收到 {y.ndim} 维数组"
            )
        if y.shape[1] != self.output_size:
            raise ValueError(
                f"标签维度 {y.shape[1]} 与模型输出维度 {self.output_size} 不匹配"
            )

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
        训练回归模型

        Args:
            X_train: 训练数据，shape: (n_samples, n_features)
            y_train: 训练标签（log₁₀ CI²），shape: (n_samples, n_levels)
            X_val: 验证数据（可选）
            y_val: 验证标签（可选）
            batch_size: 批次大小
            max_epochs: 最大训练轮数
            early_stopping_patience: 早停耐心值
            min_delta: 最小改进阈值

        Returns:
            训练历史记录字典
        """
        self._validate_input_data(X_train, y_train)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        else:
            X_val_tensor = y_val_tensor = None

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        val_loss = val_mae = 0.0

        for epoch in range(max_epochs):
            epoch_loss = self._train_epoch(X_train_tensor, y_train_tensor, batch_size)
            self.training_history["train_loss"].append(epoch_loss)

            if X_val_tensor is not None and y_val_tensor is not None:
                val_loss, val_mae = self._validate_epoch(X_val_tensor, y_val_tensor)
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_mae"].append(val_mae)

                scheduler.step(val_loss)

                if val_loss < best_val_loss - min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {
                        k: v.clone() for k, v in self.model.state_dict().items()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"早停触发，在第 {epoch + 1} 轮停止训练")
                    if best_model_state is not None:
                        self.model.load_state_dict(best_model_state)
                    break

            if (epoch + 1) % 50 == 0:
                log_msg = f"Epoch [{epoch + 1}/{max_epochs}], Loss: {epoch_loss:.6f}"
                if X_val_tensor is not None:
                    log_msg += f", Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.4f}"
                self.logger.info(log_msg)

        return self.training_history

    def _train_epoch(
        self, X_train: torch.Tensor, y_train: torch.Tensor, batch_size: int
    ) -> float:
        """训练一个 epoch，返回平均损失"""
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _validate_epoch(
        self, X_val: torch.Tensor, y_val: torch.Tensor
    ) -> tuple[float, float]:
        """验证一个 epoch，返回 (huber_loss, mae)"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_val)
            loss = self.criterion(outputs, y_val)
            mae = torch.mean(torch.abs(outputs - y_val)).item()
        return loss.item(), mae

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测 log₁₀(CI²) 值

        Args:
            X: 输入描述符，shape: (n_samples, n_features)

        Returns:
            预测的 log₁₀(CI²) 值，shape: (n_samples, n_levels)
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.cpu().numpy()

    def predict_batch(self, X: np.ndarray, batch_size: int = 1_000_000) -> np.ndarray:
        """
        分批预测，适合处理大数据集

        Args:
            X: 输入描述符
            batch_size: 每批样本数

        Returns:
            预测的 log₁₀(CI²) 值，shape: (n_samples, n_levels)
        """
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch_size):
                batch_X = X[i : i + batch_size]
                X_tensor = torch.tensor(batch_X, dtype=torch.float32).to(self.device)
                outputs = self.model(X_tensor)
                all_preds.append(outputs.cpu().numpy())
        return np.vstack(all_preds) if len(all_preds) > 1 else all_preds[0]

    def evaluate(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, float, float]:
        """
        评估回归模型性能

        Args:
            X: 输入数据，shape: (n_samples, n_features)
            y: 真实 log₁₀(CI²) 标签，shape: (n_samples, n_levels)

        Returns:
            (mae, rmse, spearman_rho)
            - mae: 平均绝对误差
            - rmse: 均方根误差
            - spearman_rho: Spearman 排名相关系数（衡量排序质量）
        """
        y_pred = self.predict(X)

        y_true_flat = y.flatten()
        y_pred_flat = y_pred.flatten()

        mae = float(np.mean(np.abs(y_pred_flat - y_true_flat)))
        rmse = float(np.sqrt(np.mean((y_pred_flat - y_true_flat) ** 2)))
        rho, _ = spearmanr(y_true_flat, y_pred_flat)
        spearman_rho = float(rho)

        self.logger.info(
            f"回归评估结果 - MAE: {mae:.4f}, RMSE: {rmse:.4f}, Spearman ρ: {spearman_rho:.4f}"
        )
        return mae, rmse, spearman_rho

    def plot_curve(
        self,
        X: np.ndarray,
        y: np.ndarray,
        filename: str,
        level_titles: list[str] | None = None,
    ) -> None:
        """
        绘制预测值 vs 真实值散点图

        Args:
            X: 输入数据
            y: 真实 log₁₀(CI²) 标签，shape: (n_samples, n_levels)
            filename: 保存文件名
            level_titles: 每个能级的标题列表（可选）
        """
        y_pred = self.predict(X)
        n_levels = self.output_size

        cols = min(n_levels, 3)
        rows = (n_levels + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

        for level_idx in range(n_levels):
            row, col = divmod(level_idx, cols)
            ax = axes[row][col]

            y_true_level = y[:, level_idx]
            y_pred_level = y_pred[:, level_idx]

            rho, _ = spearmanr(y_true_level, y_pred_level)
            mae = float(np.mean(np.abs(y_pred_level - y_true_level)))

            ax.scatter(y_true_level, y_pred_level, alpha=0.4, s=15)
            lim_min = min(y_true_level.min(), y_pred_level.min())
            lim_max = max(y_true_level.max(), y_pred_level.max())
            ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=1)
            ax.set_xlabel("True log₁₀(CI²)")
            ax.set_ylabel("Predicted log₁₀(CI²)")
            title = level_titles[level_idx] if level_titles else f"Level {level_idx + 1}"
            ax.set_title(f"{title}\nρ={rho:.3f}, MAE={mae:.3f}")
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for extra_idx in range(n_levels, rows * cols):
            row, col = divmod(extra_idx, cols)
            axes[row][col].set_visible(False)

        plt.tight_layout()
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filename)
        plt.close()

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
                "huber_delta": self.huber_delta,
            },
        }
        torch.save(save_dict, path)

    def load_model(self, path: str, device: str | None = None) -> "ANNRegressor":
        """加载模型"""
        checkpoint = torch.load(path, map_location=device or self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_history = checkpoint["training_history"]
        self.model.to(self.device)
        return self
