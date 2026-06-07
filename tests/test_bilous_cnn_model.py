import logging

import numpy as np
import pytest
import torch

from graspkit.ml_module import ANNClassifier
from graspkit.ml_module.cnn import CSFConv1DBackbone
from graspkit.ml_module.ml_trainer import _select_model_architecture


def test_csf_conv1d_backbone_forward_accepts_flat_descriptor() -> None:
    model = CSFConv1DBackbone(input_size=18, output_size=3)
    x = torch.randn(5, 18)

    logits = model(x)

    assert logits.shape == (5, 3)


def test_csf_conv1d_backbone_forward_accepts_structured_descriptor() -> None:
    model = CSFConv1DBackbone(input_size=18, output_size=2)
    x = torch.randn(5, 6, 3)

    logits = model(x)

    assert logits.shape == (5, 2)


def test_csf_conv1d_backbone_rejects_non_three_channel_input_size() -> None:
    with pytest.raises(ValueError, match="input_size"):
        CSFConv1DBackbone(input_size=17, output_size=2)


def test_ann_classifier_builds_cnn_architecture() -> None:
    classifier = ANNClassifier(
        input_size=18,
        output_size=3,
        model_architecture="cnn",
        random_seed=7,
    )

    assert isinstance(classifier.model, CSFConv1DBackbone)
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


def test_ann_classifier_iter_predict_proba_batches_matches_batch_prediction() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(13, 18)).astype(np.float32)
    classifier = ANNClassifier(
        input_size=18,
        output_size=2,
        model_architecture="cnn",
        random_seed=17,
    )

    expected = classifier.predict_proba_batch(x, batch_size=5)
    actual = np.vstack(list(classifier.iter_predict_proba_batches([x[:5], x[5:9], x[9:]])))

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


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
    assert isinstance(restored.model, CSFConv1DBackbone)
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)


def test_select_model_architecture_uses_cnn_for_early_iterations_with_enough_data() -> None:
    architecture = _select_model_architecture(
        current_loop_sample_count=100_000,
        accumulated_sample_count=150_000,
        positive_sample_count=10_000,
        logger=logging.getLogger("test"),
    )

    assert architecture == "cnn"


def test_select_model_architecture_uses_cnn_for_large_late_csf_level_data() -> None:
    architecture = _select_model_architecture(
        current_loop_sample_count=100_000,
        accumulated_sample_count=150_000,
        positive_sample_count=10_000,
        logger=logging.getLogger("test"),
    )

    assert architecture == "cnn"


def test_select_model_architecture_keeps_standard_for_late_small_positive_data() -> None:
    architecture = _select_model_architecture(
        current_loop_sample_count=100_000,
        accumulated_sample_count=150_000,
        positive_sample_count=500,
        logger=logging.getLogger("test"),
    )

    assert architecture == "standard"
