import logging

import numpy as np
import pytest
import torch

from graspkit.ml_module import ANNClassifier
from graspkit.ml_module.cnn import BilousCNN
from graspkit.ml_module.ml_trainer import _select_model_architecture


def test_bilous_cnn_forward_accepts_flat_descriptor() -> None:
    model = BilousCNN(input_size=18, output_size=3)
    x = torch.randn(5, 18)

    logits = model(x)

    assert logits.shape == (5, 3)


def test_bilous_cnn_forward_accepts_structured_descriptor() -> None:
    model = BilousCNN(input_size=18, output_size=2)
    x = torch.randn(5, 6, 3)

    logits = model(x)

    assert logits.shape == (5, 2)


def test_bilous_cnn_rejects_non_three_channel_input_size() -> None:
    with pytest.raises(ValueError, match="input_size"):
        BilousCNN(input_size=17, output_size=2)


def test_ann_classifier_builds_cnn_architecture() -> None:
    classifier = ANNClassifier(
        input_size=18,
        output_size=3,
        model_architecture="cnn",
        random_seed=7,
    )

    assert isinstance(classifier.model, BilousCNN)
    assert classifier.multi_label is True


def test_ann_classifier_cnn_predict_proba_shape() -> None:
    classifier = ANNClassifier(
        input_size=18,
        output_size=3,
        model_architecture="cnn",
        random_seed=7,
    )
    x = np.random.default_rng(3).normal(size=(4, 18)).astype(np.float32)

    proba = classifier.predict_proba(x)

    assert proba.shape == (4, 3)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_ann_classifier_cnn_fit_and_batch_predict() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(size=(24, 18)).astype(np.float32)
    y = np.zeros((24, 2), dtype=np.float32)
    y[:8, 0] = 1.0
    y[8:14, 1] = 1.0

    classifier = ANNClassifier(
        input_size=18,
        output_size=2,
        model_architecture="cnn",
        random_seed=11,
    )
    history = classifier.fit(x, y, batch_size=8, max_epochs=2)
    proba = classifier.predict_proba_batch(x, batch_size=7)

    assert len(history["train_loss"]) == 2
    assert proba.shape == (24, 2)
    assert np.all(np.isfinite(proba))


def test_ann_classifier_cnn_save_and_load(tmp_path) -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=(10, 18)).astype(np.float32)

    classifier = ANNClassifier(
        input_size=18,
        output_size=2,
        model_architecture="cnn",
        random_seed=13,
    )
    before = classifier.predict_proba(x)
    model_path = tmp_path / "cnn_model.pt"

    classifier.save_model(str(model_path))
    restored = ANNClassifier.load_model(str(model_path))
    after = restored.predict_proba(x)

    assert restored.model_architecture == "cnn"
    assert isinstance(restored.model, BilousCNN)
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)


def test_select_model_architecture_keeps_standard_for_early_iterations() -> None:
    architecture = _select_model_architecture(
        cal_loop_num=2,
        current_loop_sample_count=100_000,
        accumulated_sample_count=150_000,
        positive_sample_count=10_000,
        logger=logging.getLogger("test"),
    )

    assert architecture == "standard"


def test_select_model_architecture_uses_cnn_for_large_late_csf_level_data() -> None:
    architecture = _select_model_architecture(
        cal_loop_num=4,
        current_loop_sample_count=100_000,
        accumulated_sample_count=150_000,
        positive_sample_count=10_000,
        logger=logging.getLogger("test"),
    )

    assert architecture == "cnn"


def test_select_model_architecture_keeps_standard_for_late_small_positive_data() -> None:
    architecture = _select_model_architecture(
        cal_loop_num=4,
        current_loop_sample_count=100_000,
        accumulated_sample_count=150_000,
        positive_sample_count=500,
        logger=logging.getLogger("test"),
    )

    assert architecture == "standard"
