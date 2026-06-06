import pytest
from pydantic import ValidationError

from graspkit_config.ml_config_models import ServerSettings


def _base_server_settings() -> dict[str, object]:
    return {
        "slurm_partition": "batch",
        "tasks_per_node": 46,
        "python_source": "uv",
        "uv_env_path": "/work/graspkit-tools/.venv",
        "graspkit_tools_path": "/work/graspkit-tools",
    }


def test_server_settings_only_exposes_uv_environment_fields() -> None:
    assert "uv_env_path" in ServerSettings.model_fields
    assert "conda_path" not in ServerSettings.model_fields
    assert "conda_env_name" not in ServerSettings.model_fields


def test_server_settings_rejects_conda_python_source() -> None:
    payload = {
        **_base_server_settings(),
        "python_source": "conda",
        "uv_env_path": None,
    }

    with pytest.raises(ValidationError, match="python_source 必须是 'uv'"):
        ServerSettings.model_validate(payload)
