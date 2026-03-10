from typing import Literal, TypedDict

import numpy as np


class PredictionOutputs(TypedDict):
    y_prediction_labeled: np.ndarray


class ProbabilityOutputs(TypedDict):
    y_probability_labeled: np.ndarray
    y_probability_all: np.ndarray


class LabelOutputs(TypedDict):
    y_labeled: np.ndarray


class MetricsOutputs(TypedDict):
    f1: float
    roc_auc: float
    accuracy: float
    precision: float
    recall: float


class MetadataOutputs(TypedDict):
    eval_time: float
    labeled_samples: int
    metric_scope: Literal["in_sample"]


class EvaluationResults(TypedDict):
    probabilities: ProbabilityOutputs
    true_labels: LabelOutputs
    labeled_metrics: MetricsOutputs
    metadata: MetadataOutputs
