# -*- encoding: utf-8 -*-
"""
@Id :test_load_config.py
@date :2025/02/19
@author :YenochQin (秦毅)
"""

from pathlib import Path

import pytest

from graspkit.data_IO.processing_data_loader import load_config


class TestLoadConfig:
    """Test suite for load_config function"""

    def test_load_valid_config(self, fixtures_dir: Path):
        """Test loading a valid TOML configuration file"""
        config_path = fixtures_dir / "configs" / "realistic.toml"
        config = load_config(config_path)

        # Test target section
        assert hasattr(config, "target")
        assert config.target.atom == "Ni_I"
        assert config.target.conf == "3d8_4s2"
        assert config.target.full_CSFs_set_file == "3d8_4s2_j2as6raw.c"

        # Test cal_settings section
        assert hasattr(config, "cal_settings")
        assert config.cal_settings.cal_loop_num == 1
        assert config.cal_settings.cutoff_value == 0.00000000001
        assert config.cal_settings.sampling_ratio == 0.025
        assert isinstance(config.cal_settings.root_path, Path)

        # Test ml_config section
        assert hasattr(config, "ml_config")
        assert config.ml_config.overfitting_threshold == 0.1
        assert config.ml_config.underfitting_threshold == -0.05

    def test_load_config_with_optional_sections(self, fixtures_dir: Path):
        """Test loading config with optional rnucleus, server_settings, model_params"""
        config_path = fixtures_dir / "configs" / "optional_sections.toml"
        config = load_config(config_path)

        # Test rnucleus (optional)
        assert hasattr(config, "rnucleus")
        assert config.rnucleus.atomic_number == 28
        assert config.rnucleus.mass_number == 58

        # Test server_settings (optional)
        assert hasattr(config, "server_settings")
        assert config.server_settings.tasks_per_node == 46
        assert config.server_settings.cpu_threads == 32

        # Test model_params (optional)
        assert hasattr(config, "model_params")
        assert config.model_params.n_estimators == 100
        assert config.model_params.random_state == 42

    def test_load_config_missing_target_section(self, fixtures_dir: Path):
        """Test that missing target section raises ValueError"""
        config_path = fixtures_dir / "configs" / "missing_target.toml"

        with pytest.raises(ValueError, match="缺少必需的节"):
            load_config(config_path)

    def test_class_weight_string_to_int_keys(self, fixtures_dir: Path):
        """Test that class_weight dict keys are converted from string to int"""
        config_path = fixtures_dir / "configs" / "class_weight.toml"
        config = load_config(config_path)

        assert hasattr(config, "model_params")
        assert hasattr(config.model_params, "class_weight")

        # Keys should be integers, not strings
        class_weight = config.model_params.class_weight
        assert 0 in class_weight
        assert 1 in class_weight
        assert class_weight[0] == 1.0
        assert class_weight[1] == 2.5

    def test_cal_path_dynamic_attributes(self, fixtures_dir: Path):
        """Test that cal_path can have dynamic attributes added"""
        config_path = fixtures_dir / "configs" / "realistic.toml"
        config = load_config(config_path)

        # cal_path should be initialized
        assert hasattr(config, "cal_path")

        # Test adding dynamic attributes
        config.cal_path.custom_field = "test_value"
        assert config.cal_path.custom_field == "test_value"

        config.cal_path.test_number = 123
        assert config.cal_path.test_number == 123

    def test_spectral_term_validation(self, fixtures_dir: Path):
        """Test that spectral_term is properly loaded as a list"""
        config_path = fixtures_dir / "configs" / "optional_sections.toml"
        config = load_config(config_path)

        assert hasattr(config.cal_settings, "spectral_term")
        spectral_term = config.cal_settings.spectral_term
        assert isinstance(spectral_term, list)
        assert len(spectral_term) > 0
        assert "2s(2).2p(6).3s(2).3p(6).3d(8)1D.4s(2)_1D" in spectral_term


# Fixture for tests directory
@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory"""
    return Path(__file__).parent / "fixtures"
