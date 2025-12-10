import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
import logging
############################
#####  ANN Classifier  #####
############################
class TensorNet(nn.Module):
    def __init__(self, input_shape, hidden_dim=128, num_classes=2):
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
        
        # 自动初始化
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w3)
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)
        nn.init.zeros_(self.b3)
        nn.init.normal_(self.w2, mean=0.0, std=0.01)

    def forward(self, x):
        x= x.reshape(-1,self.input_shape[0],self.input_shape[1])
        a1 = torch.einsum('ijk,jkl->ilk', x, self.w1) + self.b1
        a2 = torch.einsum('ijk,k->ij', self.relu(a1), self.w2) + self.b2
        a3 = torch.matmul(self.relu(a2), self.w3) + self.b3
        return a3

class ANNClassifier:
    def __init__(self, input_size, hidden_size=150, output_size=2, learning_rate=0.001):
        """
        Initialize the ANN classifier.
        :param input_size: Number of input features.
        :param hidden_size: Number of neurons in the hidden layer.
        :param output_size: Number of output classes (default: 2 for binary classification).
        :param learning_rate: Learning rate for the optimizer.
        """
        torch.set_num_threads(12)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, output_size)
        # ).to(self.device)
        # self.model = StaticLiquidNN(input_dim=input_size).to(self.device)
        self.model = TensorNet(input_shape=(int(input_size/3), 3)).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=2048, max_epochs=150, weight = [1,1]):
        """
        Train the ANN model.
        :param X_train: Training data (numpy array).
        :param y_train: Training labels (numpy array).
        :param X_val: Validation data (optional, numpy array).
        :param y_val: Validation labels (optional, numpy array).
        :param batch_size: Size of each training batch.
        :param max_epochs: Maximum number of epochs.
        """

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)
        weights = [y_train_tensor.sum()/len(y_train_tensor), 1-y_train_tensor.sum()/len(y_train_tensor)]
        self.criterion = nn.CrossEntropyLoss(weight=torch.Tensor(weights).to(self.device))
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(self.device)

        # Training loop
        self.model.train()
        for epoch in range(max_epochs):
            permutation = torch.randperm(X_train_tensor.size(0))
            epoch_loss = 0.0

            for i in range(0, X_train_tensor.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]

                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 50 == 0:
                logging.getLogger().info(f"Epoch [{epoch + 1}/{max_epochs}], Loss: {epoch_loss:.8f}")

                # Optionally evaluate on validation set
                if X_val is not None and y_val is not None:
                    self.evaluate(X_val, y_val, verbose=True)

    def predict(self, X):
        """
        Predict the labels for the given input data.
        :param X: Input data (numpy array).
        :return: Predicted labels (numpy array).
        """
        predictions = self.predict_proba(X)
        predictions = predictions[:, 1]>0.5

        return predictions
    
    def predict_proba(self, X):
        """
        Predict the labels for the given input data.
        :param X: Input data (numpy array).
        :return: Predicted labels (numpy array).
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = torch.softmax(self.model(X_tensor), dim=1)

        return outputs.cpu().numpy()
    
    def predict_batch(self, X, batch_size=1024):
        """
        Predict the labels for the given input data with batch processing.
        :param X: Input data (numpy array).
        :param batch_size: Number of samples per batch (adjust based on GPU memory).
        :return: Predicted labels (numpy array).
        """
        # Get probability predictions in batches
        proba = self.predict_proba_batch(X, batch_size=batch_size)
        # Convert probabilities to binary labels
        return (proba[:, 1] > 0.5).astype(int)


    def predict_proba_batch(self, X, batch_size=1024):
        """
        Predict probabilities for the given input data with batch processing.
        :param X: Input data (numpy array).
        :param batch_size: Number of samples per batch (adjust based on GPU memory).
        :return: Predicted probabilities (numpy array).
        """
        self.model.eval()
        num_samples = X.shape[0]
        all_proba = []
        
        # Process in batches to avoid OOM
        with torch.no_grad():  # Disable gradient computation for memory efficiency
            for i in range(0, num_samples, batch_size):
                # Get batch data (handle last batch which may be smaller)
                batch_X = X[i:min(i + batch_size, num_samples)]
                
                # Convert to tensor and move to device
                X_tensor = torch.tensor(batch_X, dtype=torch.float32).to(self.device)
                
                # Model inference
                outputs = torch.softmax(self.model(X_tensor), dim=1)
                
                # Move results to CPU and convert to numpy
                all_proba.append(outputs.cpu().numpy())
        
        # Concatenate all batches
        return np.vstack(all_proba)

    def evaluate(self, X, y, verbose=True):
        """
        Evaluate the model on the given data and labels.
        :param X: Input data (numpy array).
        :param y: True labels (numpy array).
        :param verbose: If True, print the evaluation metrics.
        :return: A dictionary containing evaluation metrics.
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
            y_proba = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "f1_score": f1_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_proba)
        }

        if verbose:
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")

        return metrics

    def plot_roc_curve(self, X, y):
        """
        Plot the ROC curve for the model.
        :param X: Input data (numpy array).
        :param y: True labels (numpy array).
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            y_proba = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

        fpr, tpr, _ = roc_curve(y, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y, y_proba):.4f})")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

    def save_model(self, path):
        """
        Save the model parameters to a file.
        :param path: File path to save the model.
        """
        torch.save(self.model.state_dict(), path)
        joblib.dump(self, path + '_ann.pkl')

    def load_model(self, path):
        """
        Load the model parameters from a file.
        :param path: File path to load the model.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        return joblib.load(path + '_ann.pkl')
    
    def resampling(X_train, y_train, weight):
        X_positive = X_train[y_train==1]
        X_negtive = X_train[y_train==0]
        Y_positive = y_train[y_train==1]
        Y_negtive = y_train[y_train==0]

        Num_min = max(X_positive.shape[0], X_negtive.shape[0])

        N_sample_positive, N_sample_negtive = int(weight[0]*Num_min), int(weight[1]*Num_min)
        id_positive = np.random.randint(0, X_positive.shape[0], N_sample_positive)
        id_negtive = np.random.randint(0, X_negtive.shape[0], N_sample_negtive)

        X_resampled = np.concatenate((X_positive[id_positive], X_negtive[id_negtive]), axis=0)
        y_resampled = np.concatenate((Y_positive[id_positive], Y_negtive[id_negtive]), axis=0)

        # 打乱顺序
        shuffle_index = np.random.permutation(X_resampled.shape[0])
        X_resampled = X_resampled[shuffle_index]
        y_resampled = y_resampled[shuffle_index]
        return X_resampled, y_resampled

    def downsampling(X_train, y_train, weight):
        X_positive = X_train[y_train==1]
        X_negative = X_train[y_train==0]
        Y_positive = y_train[y_train==1]
        Y_negative = y_train[y_train==0]

        Num_max = min(X_positive.shape[0], X_negative.shape[0])

        N_sample_positive, N_sample_negative = int(weight[0]*Num_max), int(weight[1]*Num_max)
        id_positive = np.random.choice(X_positive.shape[0], N_sample_positive, replace=False)
        id_negative = np.random.choice(X_negative.shape[0], N_sample_negative, replace=False)

        X_resampled = np.concatenate((X_positive[id_positive], X_negative[id_negative]), axis=0)
        y_resampled = np.concatenate((Y_positive[id_positive], Y_negative[id_negative]), axis=0)

        # 打乱顺序
        shuffle_index = np.random.permutation(X_resampled.shape[0])
        X_resampled = X_resampled[shuffle_index]
        y_resampled = y_resampled[shuffle_index]
        return X_resampled, y_resampled
    
    def plot_curve(ci_temp, y_proba_all, y_test, y_proba, filename):
        # 绘制 ROC 曲线
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        # 绘制 PR 曲线
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        plt.subplot(2, 2, 2)
        plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
        plt.title("PR Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")

        # 绘制混淆矩阵
        y_pred = np.where(y_proba > 0.5, 1, 0)
        cm = confusion_matrix(y_test, y_pred)
        plt.subplot(2, 2, 3)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # 绘制可解释性曲线
        plt.subplot(2, 2, 4)
        plt.scatter(y_proba_all[:, 1], np.log(abs(ci_temp)))
        plt.xlabel('Predicted Probability')
        plt.ylabel('Ci Values')
        plt.title('Ci with Predicted Probability')
        # 绘制校准曲线
        # prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
        # plt.subplot(2, 2, 4)
        # plt.plot(prob_pred, prob_true, 's-')
        # plt.plot([0, 1], [0, 1], '--', color='gray')
        # plt.title('Calibration Curve')
        # plt.xlabel('Mean Predicted Probability')
        # plt.ylabel('Fraction of Positives')

        plt.tight_layout()
        
        result_file = f"roc_curves/{filename}.png"
        plt.savefig(result_file)
        plt.close()

        return roc_auc, pr_auc
    
    def model_evaluation(y_test, y_pred, y_proba):

        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        return f1, roc_auc, accuracy, precision, recall
    
    def print_gpu_memory():
        # 总内存
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # 当前分配的内存
        allocated_memory = torch.cuda.memory_allocated(0)
        # 缓存的内存（PyTorch 保留但未分配给张量的内存）
        cached_memory = torch.cuda.memory_reserved(0)
        
        # 转换为 GiB 并打印
        print(f"GPU 总内存: {total_memory/1024**3:.2f} GiB")
        print(f"已分配内存: {allocated_memory/1024**3:.2f} GiB")
        print(f"缓存内存: {cached_memory/1024**3:.2f} GiB")
        print(f"可用内存: {(total_memory-allocated_memory)/1024**3:.2f} GiB")
        
    def predict_in_batches(model, X, batch_size=4000000, predict_proba=False, proba_index=1):
        """
        超快速分批预测函数，支持普通预测和概率预测
        
        参数:
        model: 训练好的模型
        X: 待预测数据
        batch_size: 每批处理的数据量（较大值通常更快）
        predict_proba: 是否使用概率预测
        proba_index: 概率预测中要提取的列索引（默认是1，表示正类概率）
        
        返回:
        y_pred: 完整的预测结果
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
            batch_X = X[i:i+batch_size]
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