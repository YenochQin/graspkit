from typing import TypedDict

import numpy as np


class PredictionOutputs(TypedDict):
    y_prediction_test: np.ndarray
    y_prediction_train: np.ndarray


class ProbabilityOutputs(TypedDict):
    y_probability_test: np.ndarray
    y_probability_train: np.ndarray
    y_probability_all: np.ndarray


class LabelOutputs(TypedDict):
    y_test: np.ndarray
    y_train: np.ndarray


class MetricsOutputs(TypedDict):
    f1: float
    roc_auc: float
    accuracy: float
    precision: float
    recall: float


class MetadataOutputs(TypedDict):
    eval_time: float
    test_samples: int
    train_samples: int


class EvaluationResults(TypedDict):
    probabilities: ProbabilityOutputs
    true_labels: LabelOutputs
    test_metrics: MetricsOutputs
    train_metrics: MetricsOutputs
    metadata: MetadataOutputs
