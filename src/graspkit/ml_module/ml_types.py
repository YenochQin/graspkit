from typing import Literal, TypedDict

import numpy as np


class PredictionOutputs(TypedDict):
    """Predicted labels for the currently labeled training subset.

    Attributes:
        y_prediction_labeled: Predicted labels for labeled samples.
    """

    y_prediction_labeled: np.ndarray


class ProbabilityOutputs(TypedDict):
    """Model probabilities for labeled samples and all evaluated samples.

    Attributes:
        y_probability_labeled: Probabilities for labeled samples.
        y_probability_all: Probabilities for all evaluated samples.
    """

    y_probability_labeled: np.ndarray
    y_probability_all: np.ndarray


class LabelOutputs(TypedDict):
    """Ground-truth labels used for model evaluation.

    Attributes:
        y_labeled: Ground-truth labels for labeled samples.
    """

    y_labeled: np.ndarray


class MetricsOutputs(TypedDict):
    """Classification metrics produced by in-sample diagnostics.

    Attributes:
        f1: F1 score.
        roc_auc: ROC AUC score.
        accuracy: Accuracy score.
        precision: Precision score.
        recall: Recall score.
    """

    f1: float
    roc_auc: float
    accuracy: float
    precision: float
    recall: float


class MetadataOutputs(TypedDict):
    """Metadata describing the scope and cost of a model evaluation.

    Attributes:
        eval_time: Evaluation runtime in seconds.
        labeled_samples: Number of labeled samples used for diagnostics.
        metric_scope: Scope represented by the metrics.
    """

    eval_time: float
    labeled_samples: int
    metric_scope: Literal["in_sample"]


class EvaluationResults(TypedDict):
    """Complete evaluation payload returned by ML classification diagnostics.

    Attributes:
        probabilities: Model probability outputs.
        true_labels: Ground-truth label outputs.
        labeled_metrics: Metrics computed on labeled samples.
        metadata: Evaluation metadata.
    """

    probabilities: ProbabilityOutputs
    true_labels: LabelOutputs
    labeled_metrics: MetricsOutputs
    metadata: MetadataOutputs
