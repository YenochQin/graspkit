import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graspkit.ml_module import ml_results_analyzer, ml_trainer
from graspkit.ml_module.neural_network import ANNClassifier, CSFClassifier
from graspkit.utils.data_modules import MLDataCounts


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

    monkeypatch.setattr(ml_trainer, "CSFClassifier", DummyClassifier)
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


def test_predict_model_streaming_matches_numpy_selection(
    monkeypatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path)
    config.cal_settings.cutoff_value = 0.1
    config.cal_settings.expansion_ratio = 1.0
    config.cal_settings.sampling_ratio = 1.0
    config.cal_settings.reference_energy_levels = []
    config.cal_settings.reference_energy_mode = "monitor"
    config.cal_settings.reference_energy_score_weight = 0.0
    config.cal_settings.reference_energy_importance_weight = 1.0
    config.cal_settings.reference_gap_pair_weighting = "uniform"
    config.cal_settings.reference_energy_top_pair_count = 1
    config.cal_path.results_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("streaming-predict-test")

    raw_np = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    raw_lazy = pl.DataFrame(raw_np, schema=["col_0", "col_1", "col_2"]).lazy()
    caled_idxs = np.array([0, 2, 4], dtype=np.int64)
    correct_levels_ci_squared = np.array([[0.2, 0.01, 0.3]], dtype=np.float64)

    class ScoreByFirstColumn:
        output_size = 1
        multi_label = True

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return (X[:, [0]] / 10.0).astype(float)

        def predict_proba_batch(
            self, X: np.ndarray, batch_size: int = 1024
        ) -> np.ndarray:
            return self.predict_proba(X)

        def iter_predict_proba_batches(self, batches):
            for batch in batches:
                yield self.predict_proba(batch)

    monkeypatch.setattr(ml_trainer, "_is_hybrid_reference_ranking_enabled", lambda *_: False)

    numpy_result = ml_trainer.predict_model(
        ScoreByFirstColumn(),
        raw_np,
        caled_idxs,
        correct_levels_ci_squared,
        config,
        MLDataCounts(total_csfs_count=6, cal_csfs_count=3),
        logger,
    )
    streaming_result = ml_trainer.predict_model_streaming(
        ScoreByFirstColumn(),
        raw_lazy,
        caled_idxs,
        correct_levels_ci_squared,
        config,
        MLDataCounts(total_csfs_count=6, cal_csfs_count=3),
        logger,
        batch_size=2,
    )

    np.testing.assert_array_equal(streaming_result[0], numpy_result[0])
    np.testing.assert_array_equal(streaming_result[1], numpy_result[1])
    np.testing.assert_allclose(streaming_result[2], numpy_result[2])
    assert streaming_result[3].ml_sampled_count == numpy_result[3].ml_sampled_count


def test_predict_model_streaming_falls_back_to_cross_batch_top_k(
    monkeypatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path)
    config.cal_settings.cutoff_value = 0.1
    config.cal_settings.expansion_ratio = 3.0
    config.cal_settings.sampling_ratio = 1.0
    config.cal_settings.reference_energy_levels = []
    config.cal_settings.reference_energy_mode = "monitor"
    config.cal_settings.reference_energy_score_weight = 0.0
    logger = logging.getLogger("streaming-predict-fallback-test")

    raw_np = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
        ],
        dtype=np.float32,
    )
    raw_lazy = pl.DataFrame(raw_np, schema=["col_0", "col_1"]).lazy()
    caled_idxs = np.array([0, 2], dtype=np.int64)
    correct_levels_ci_squared = np.array([[0.2, 0.01]], dtype=np.float64)

    class LowScoreByFirstColumn:
        output_size = 1
        multi_label = True

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return (X[:, [0]] / 100.0).astype(float)

    monkeypatch.setattr(ml_trainer, "_is_hybrid_reference_ranking_enabled", lambda *_: False)

    streaming_result = ml_trainer.predict_model_streaming(
        LowScoreByFirstColumn(),
        raw_lazy,
        caled_idxs,
        correct_levels_ci_squared,
        config,
        MLDataCounts(total_csfs_count=7, cal_csfs_count=2),
        logger,
        batch_size=2,
    )

    np.testing.assert_array_equal(streaming_result[0], np.array([6, 5, 4]))
    assert streaming_result[3].ml_predicted_count == 0
    assert streaming_result[3].ml_sampled_count == 3


def test_predict_model_streaming_preserves_order_when_positive_count_below_target(
    monkeypatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path)
    config.cal_settings.cutoff_value = 0.1
    config.cal_settings.expansion_ratio = 4.0
    config.cal_settings.sampling_ratio = 1.0
    config.cal_settings.reference_energy_levels = []
    config.cal_settings.reference_energy_mode = "monitor"
    config.cal_settings.reference_energy_score_weight = 0.0
    logger = logging.getLogger("streaming-predict-positive-order-test")

    raw_np = np.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
        ],
        dtype=np.float32,
    )
    raw_lazy = pl.DataFrame(raw_np, schema=["col_0"]).lazy()
    caled_idxs = np.array([0, 2], dtype=np.int64)
    correct_levels_ci_squared = np.array([[0.2, 0.01]], dtype=np.float64)
    score_by_index = {
        0: 0.0,
        1: 0.9,
        2: 0.0,
        3: 0.2,
        4: 0.7,
        5: 0.1,
        6: 0.8,
    }

    class SparsePositiveScores:
        output_size = 1
        multi_label = True

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.array([[score_by_index[int(row[0])]] for row in X], dtype=float)

    monkeypatch.setattr(ml_trainer, "_is_hybrid_reference_ranking_enabled", lambda *_: False)

    streaming_result = ml_trainer.predict_model_streaming(
        SparsePositiveScores(),
        raw_lazy,
        caled_idxs,
        correct_levels_ci_squared,
        config,
        MLDataCounts(total_csfs_count=7, cal_csfs_count=2),
        logger,
        batch_size=2,
    )

    np.testing.assert_array_equal(streaming_result[0], np.array([1, 4, 6]))
    assert streaming_result[3].ml_predicted_count == 3


def test_predict_model_streaming_counts_predictions_when_sampling_target_is_zero(
    monkeypatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path)
    config.cal_settings.cutoff_value = 0.1
    config.cal_settings.expansion_ratio = 1.0
    config.cal_settings.sampling_ratio = 0.25
    config.cal_settings.reference_energy_levels = []
    config.cal_settings.reference_energy_mode = "monitor"
    config.cal_settings.reference_energy_score_weight = 0.0
    logger = logging.getLogger("streaming-predict-zero-target-test")

    raw_np = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    raw_lazy = pl.DataFrame(raw_np, schema=["col_0"]).lazy()
    caled_idxs = np.array([0, 2], dtype=np.int64)
    correct_levels_ci_squared = np.array([[0.2, 0.01]], dtype=np.float64)

    class TwoPositiveScores:
        output_size = 1
        multi_label = True

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return (X[:, [0]] > 0.0).astype(float) * 0.8

    monkeypatch.setattr(ml_trainer, "_is_hybrid_reference_ranking_enabled", lambda *_: False)

    streaming_result = ml_trainer.predict_model_streaming(
        TwoPositiveScores(),
        raw_lazy,
        caled_idxs,
        correct_levels_ci_squared,
        config,
        MLDataCounts(total_csfs_count=4, cal_csfs_count=2),
        logger,
        batch_size=1,
    )

    assert streaming_result[0].size == 0
    assert streaming_result[3].ml_predicted_count == 2
    assert streaming_result[3].ml_sampled_count == 0


def test_predict_model_streaming_uses_iter_batches_for_current_descriptors(
    monkeypatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path)
    config.cal_settings.cutoff_value = 0.1
    config.cal_settings.expansion_ratio = 1.0
    config.cal_settings.sampling_ratio = 1.0
    config.cal_settings.reference_energy_levels = []
    config.cal_settings.reference_energy_mode = "monitor"
    config.cal_settings.reference_energy_score_weight = 0.0
    logger = logging.getLogger("streaming-predict-current-iter-test")

    raw_np = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    raw_lazy = pl.DataFrame(raw_np, schema=["col_0"]).lazy()

    class IterOnlyCurrentModel:
        output_size = 1
        multi_label = True

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            if {0.0, 2.0}.issuperset(set(X[:, 0].tolist())):
                raise AssertionError("current descriptors should use iterator batches")
            return (X[:, [0]] / 10.0).astype(float)

        def iter_predict_proba_batches(self, batches):
            for batch in batches:
                yield (batch[:, [0]] / 10.0).astype(float)

    monkeypatch.setattr(ml_trainer, "_is_hybrid_reference_ranking_enabled", lambda *_: False)

    streaming_result = ml_trainer.predict_model_streaming(
        IterOnlyCurrentModel(),
        raw_lazy,
        np.array([0, 2], dtype=np.int64),
        np.array([[0.2, 0.01]], dtype=np.float64),
        config,
        MLDataCounts(total_csfs_count=4, cal_csfs_count=2),
        logger,
        batch_size=1,
    )

    np.testing.assert_allclose(streaming_result[2], np.array([[0.0], [0.2]]))


def test_iter_unselected_index_chunks_skips_current_indices_without_full_diff() -> None:
    chunks = list(
        ml_trainer._iter_unselected_index_chunks(
            total_count=9,
            current_calc_idxs=np.array([6, 1, 3], dtype=np.int64),
            chunk_size=4,
        )
    )

    assert [chunk.tolist() for chunk in chunks] == [[0, 2], [4, 5, 7], [8]]


def test_predict_model_streaming_rejects_hybrid_reference_ranking(
    monkeypatch, tmp_path: Path
) -> None:
    config = make_config(tmp_path)
    logger = logging.getLogger("streaming-predict-hybrid-test")
    raw_lazy = pl.DataFrame({"col_0": [0.0, 1.0]}).lazy()

    class AnyModel:
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.full((len(X), 1), 0.1)

    monkeypatch.setattr(ml_trainer, "_is_hybrid_reference_ranking_enabled", lambda *_: True)

    with pytest.raises(NotImplementedError, match="Hybrid reference ranking"):
        ml_trainer.predict_model_streaming(
            AnyModel(),
            raw_lazy,
            np.array([0], dtype=np.int64),
            np.array([[0.2]], dtype=np.float64),
            config,
            MLDataCounts(total_csfs_count=2, cal_csfs_count=1),
            logger,
            selected_energy_data=pl.DataFrame({"level": [0]}),
            batch_size=1,
        )


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


def test_csfclassifier_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = CSFClassifier(
        input_size=6,
        output_size=2,
        hidden_size=12,
        model_architecture="standard",
        random_seed=123,
    )
    checkpoint_path = tmp_path / "csf_checkpoint.pt"

    model.save_model(str(checkpoint_path))
    loaded = CSFClassifier.load_model(str(checkpoint_path), device="cpu")

    assert isinstance(loaded, CSFClassifier)
    assert loaded.model_architecture == "standard"
