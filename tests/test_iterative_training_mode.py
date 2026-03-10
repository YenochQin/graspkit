import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graspkit.ml_module import ml_results_analyzer, ml_trainer
from graspkit.ml_module.neural_network import ANNClassifier


def make_config(tmp_path: Path, loop_num: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        cal_path=SimpleNamespace(
            training_results=tmp_path / "training_results.csv",
            results_path=tmp_path / "results",
            models_path=tmp_path / "models",
            roc_curves_path=tmp_path / "roc",
            loop_file_name="demo_loop",
        ),
        cal_settings=SimpleNamespace(
            cal_loop_num=loop_num,
            spectral_term=["2P", "2D"],
        ),
        ml_config=SimpleNamespace(
            overfitting_threshold=0.3,
            underfitting_threshold=-0.3,
        ),
        server_settings=SimpleNamespace(cpu_threads=1),
        target=SimpleNamespace(conf="demo"),
    )


class DummyClassifier:
    def __init__(self, input_size: int, output_size: int, **kwargs: object) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.multi_label = output_size > 1
        self.fit_kwargs: dict[str, object] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: object) -> dict[str, list[float]]:
        self.fit_kwargs = {
            "X_shape": X.shape,
            "y_shape": y.shape,
            **kwargs,
        }
        return {"train_loss": [0.1], "val_loss": [], "val_accuracy": []}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones((len(X), self.output_size), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X), self.output_size), 0.75, dtype=float)

    def save_model(self, path: str) -> None:
        Path(path).write_text("dummy-checkpoint", encoding="utf-8")

    @staticmethod
    def model_evaluation(
        y_true: np.ndarray, y_pred: np.ndarray, y_probability: np.ndarray
    ) -> tuple[float, float, float, float, float]:
        return (0.8, 0.9, 0.85, 0.82, 0.81)


def test_train_model_uses_all_labeled_samples_without_validation_split(
    monkeypatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path)
    logger = logging.getLogger("train-model-test")

    monkeypatch.setattr(ml_trainer, "ANNClassifier", DummyClassifier)
    monkeypatch.setattr(ml_trainer.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ml_trainer.torch, "set_num_threads", lambda n: None)

    caled_csfs_descriptors = np.array(
        [
            [0.1, 0.2, 1, 0],
            [0.3, 0.4, 0, 1],
            [0.5, 0.6, 1, 1],
            [0.7, 0.8, 0, 0],
        ],
        dtype=float,
    )
    correct_levels_ci = np.array([[0.2, 0.1, 0.5, 0.0], [0.0, 0.3, 0.4, 0.1]])

    model, X_labeled, y_labeled = ml_trainer.train_model(
        config,
        caled_csfs_descriptors,
        correct_levels_ci,
        logger,
    )

    assert X_labeled.shape == (4, 2)
    assert y_labeled.shape == (4, 2)
    assert model.fit_kwargs is not None
    assert model.fit_kwargs["X_shape"] == (4, 2)
    assert model.fit_kwargs["y_shape"] == (4, 2)
    assert model.fit_kwargs["X_val"] is None
    assert model.fit_kwargs["y_val"] is None


def test_evaluate_model_reports_labeled_metrics_only(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    logger = logging.getLogger("evaluate-model-test")
    model = DummyClassifier(input_size=2, output_size=2)
    X_labeled = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    y_labeled = np.array([[1, 0], [0, 1]], dtype=int)

    evaluation_results, prediction_outputs = ml_trainer.evaluate_model(
        model,
        X_labeled,
        y_labeled,
        config,
        logger,
    )

    assert "labeled_metrics" in evaluation_results
    assert "test_metrics" not in evaluation_results
    assert evaluation_results["metadata"]["labeled_samples"] == 2
    assert evaluation_results["metadata"]["metric_scope"] == "in_sample"
    assert prediction_outputs["y_prediction_labeled"].shape == (2, 2)

    csv_text = config.cal_path.training_results.read_text(encoding="utf-8")
    assert "labeled_f1" in csv_text
    assert "metric_scope" in csv_text


def test_save_and_plot_results_writes_labeled_parquet_only(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    logger = logging.getLogger("save-results-test")
    config.cal_path.results_path.mkdir(parents=True, exist_ok=True)
    config.cal_path.models_path.mkdir(parents=True, exist_ok=True)
    config.cal_path.roc_curves_path.mkdir(parents=True, exist_ok=True)

    evaluation_results = {
        "probabilities": {
            "y_probability_labeled": np.array([[0.8, 0.2], [0.1, 0.9]]),
            "y_probability_all": np.array([[0.8, 0.2], [0.1, 0.9]]),
        },
        "true_labels": {"y_labeled": np.array([[1, 0], [0, 1]])},
        "labeled_metrics": {
            "f1": 0.8,
            "roc_auc": 0.9,
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.81,
        },
        "metadata": {
            "eval_time": 0.01,
            "labeled_samples": 2,
            "metric_scope": "in_sample",
        },
    }
    prediction_outputs = {
        "y_prediction_labeled": np.array([[1, 0], [0, 1]]),
    }
    model = DummyClassifier(input_size=2, output_size=2)

    saved_files = ml_results_analyzer.save_and_plot_results(
        config,
        logger,
        evaluation_results,
        prediction_outputs,
        model,
        correct_levels_ci=np.array([[0.2, 0.1], [0.3, 0.4]]),
        y_current_cal_probability=np.array([[0.8, 0.2], [0.1, 0.9]]),
        save_model=True,
        save_data=True,
        plot_curves=True,
    )

    labeled_file = config.cal_path.results_path / "demo_loop_labeled_results.parquet"
    model_file = config.cal_path.models_path / "demo_loop.pt"
    assert labeled_file.exists()
    assert model_file.exists()
    assert "labeled_data" in saved_files
    assert saved_files["model"].endswith("demo_loop.pt")

    frame = pl.read_parquet(labeled_file)
    assert frame.columns == ["y_true", "y_prediction", "y_proba"]
    assert frame.shape == (2, 3)


def test_annclassifier_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = ANNClassifier(
        input_size=4,
        hidden_size=8,
        output_size=2,
        learning_rate=0.002,
        class_weights=[1.0, 3.0],
        use_dynamic_weights=False,
        model_architecture="standard",
        tensor_channels=1,
        random_seed=7,
    )
    model.training_history["train_loss"].append(0.123)
    checkpoint_path = tmp_path / "model.pt"

    model.save_model(str(checkpoint_path))
    loaded = ANNClassifier.load_model(str(checkpoint_path), device="cpu")

    assert loaded.input_size == 4
    assert loaded.hidden_size == 8
    assert loaded.output_size == 2
    assert loaded.learning_rate == 0.002
    assert loaded.model_architecture == "standard"
    assert loaded.use_dynamic_weights is False
    assert loaded.training_history["train_loss"] == [0.123]
    assert loaded.class_weights is not None
    assert loaded.class_weights.detach().cpu().tolist() == [1.0, 3.0]

    original_state = model.model.state_dict()
    loaded_state = loaded.model.state_dict()
    assert original_state.keys() == loaded_state.keys()
    for key in original_state:
        assert torch.equal(original_state[key].cpu(), loaded_state[key].cpu())
