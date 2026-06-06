import torch

from graspkit.ml_module import (
    ANNClassifier,
    CSFClassifier,
    CSFMLPBackbone,
)
from graspkit.ml_module.ann import CSFMLPBackbone as DirectCSFMLPBackbone
from graspkit.ml_module.neural_network import ANNClassifier as LegacyANNClassifier
from graspkit.ml_module.neural_network import CSFClassifier as DirectCSFClassifier


def test_csf_mlp_backbone_is_available_from_direct_and_package_entrypoints() -> None:
    assert CSFMLPBackbone is DirectCSFMLPBackbone

    model = CSFMLPBackbone(input_size=18, hidden_size=12, output_size=3)
    logits = model(torch.randn(4, 18))

    assert logits.shape == (4, 3)


def test_csf_classifier_is_primary_classifier_entrypoint() -> None:
    assert CSFClassifier is DirectCSFClassifier
    assert ANNClassifier is CSFClassifier
    assert LegacyANNClassifier is ANNClassifier


def test_csf_classifier_load_model_returns_primary_class(tmp_path) -> None:
    model = CSFClassifier(input_size=18, output_size=2, random_seed=17)
    model_path = tmp_path / "csf_classifier.pt"

    model.save_model(str(model_path))
    restored = CSFClassifier.load_model(str(model_path), device="cpu")

    assert isinstance(restored, CSFClassifier)
    assert type(restored).__name__ == "CSFClassifier"
