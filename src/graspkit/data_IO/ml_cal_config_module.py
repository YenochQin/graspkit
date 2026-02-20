# -*- encoding: utf-8 -*-
'''
@Id :ml_cal_config_module.py
@date :2026/02/20 15:48:54
@author :YenochQin (秦毅)
'''

from pathlib import Path
from typing import ClassVar
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


# Target model
class Target(BaseModel):
    """Target atom configuration"""
    atom: str
    conf: str
    full_CSFs_set_file: str
    presampled_csfs_file: str | None
    presampled_csfs_mix_file: str | None


# CalSettings model
class CalSettings(BaseModel):
    """Calculation settings"""
    continue_cal: bool = True
    cal_loop_num: int
    cal_error_num: int = 0
    backward_loop_needed: bool = False
    target_backward_loop: int = 0
    cutoff_value: float
    sampling_ratio: float
    expansion_ratio: float
    energy_std_threshold: float = 1e-5
    csfs_num_relative_std_threshold: float = 0.002
    cal_method: str = "rci"
    cal_rwfn_file: str
    rci_n_quantum_for_self_energy: int
    root_path: Path
    cal_levels: str
    spectral_term: list[str]

    @field_validator("root_path", mode="before")
    @classmethod
    def convert_root_path_to_path(cls, v: str | Path) -> Path:
        """自动将字符串路径转换为Path对象"""
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("cutoff_value")
    @classmethod
    def validate_cutoff_value(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("cutoff_value 必须大于 0")
        return v

    @field_validator("sampling_ratio")
    @classmethod
    def validate_sampling_ratio(cls, v: float) -> float:
        if not (0 < v <= 1):
            raise ValueError("sampling_ratio 必须在 (0, 1] 范围内")
        return v

    @field_validator("spectral_term")
    @classmethod
    def validate_spectral_term(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and len(v) == 0:
            raise ValueError("spectral_term 必须是非空列表")
        return v

# ServerSettings model
class ServerSettings(BaseModel):
    """Server settings"""
    slurm_partition: str
    tasks_per_node: int
    mpi_tmp_path: str | None = None
    cpu_threads: int | None = None
    python_source: str  # "uv" or "conda"
    uv_env_path: str | None = None
    conda_path: str | None = None
    conda_env_name: str | None = None
    graspkit_tools_path: str
    module_load: list[str] | None = None

    @field_validator("tasks_per_node")
    @classmethod
    def validate_tasks_per_node(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("tasks_per_node 必须大于 0")
        return v

    @model_validator(mode="after")
    def validate_python_source_config(self) -> "ServerSettings":
        """验证 python_source 与对应路径的配置一致性"""
        if self.python_source == "uv":
            if self.uv_env_path is None:
                raise ValueError("当 python_source='uv' 时，必须提供 uv_env_path")
            # conda 相关字段应该为空
            if self.conda_path is not None or self.conda_env_name is not None:
                raise ValueError("当 python_source='uv' 时，conda_path 和 conda_env_name 应该为空")
        elif self.python_source == "conda":
            if self.conda_path is None:
                raise ValueError("当 python_source='conda' 时，必须提供 conda_path")
            if self.conda_env_name is None:
                raise ValueError("当 python_source='conda' 时，必须提供 conda_env_name")
            # uv 相关字段应该为空
            if self.uv_env_path is not None:
                raise ValueError("当 python_source='conda' 时，uv_env_path 应该为空")
        else:
            raise ValueError(f"python_source 必须是 'uv' 或 'conda'，当前值: {self.python_source}")
        return self


# Rnucleus model
class Rnucleus(BaseModel):
    """Nucleus configuration"""
    atomic_number: int
    mass_number: int
    atomic_mass: float
    nuclear_spin: int = 1
    nuclear_dipole: int = 1
    nuclear_quadrupole: int = 1

# Step Control model
class StepControl(BaseModel):
    """Step Control"""
    enable_step_control: bool = False
    target_loop: int = 0
    start_step: str = "auto"
    end_step: str = "auto"
    skip_completed_steps: bool = False


# MlConfig model
class MlConfig(BaseModel):
    """Machine learning configuration"""
    use_rcsfs: bool = True
    high_prob_percentile: int = 95
    overfitting_threshold: float
    underfitting_threshold: float
    include_wrong_level_negatives: bool = True
    feature_selection: bool = True


# ModelParams model (optional)
class ModelParams(BaseModel):
    """Model parameters (optional)"""
    n_estimators: int = 1000
    random_state: int = 42


# CalPath model - supports dynamic attributes
class CalPath(BaseModel):
    """Calculation paths with dynamic attribute support"""
    # 显式添加类型注解
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")


# Root MLCalConfig model
class MLCalConfig(BaseModel):
    """Root ML calculation configuration model"""
    target: Target
    cal_settings: CalSettings
    ml_config: MlConfig
    cal_path: CalPath = CalPath()  # Always initialized
    rnucleus: Rnucleus 
    server_settings: ServerSettings
    model_params: ModelParams 
    step_control: StepControl

    @model_validator(mode="before")
    @classmethod
    def validate_required_sections(cls, data: dict[str, int | float | str | bool | Path | None]) -> dict[str, int | float | str | bool | Path | None]:
        """Validate required sections exist"""
        required_sections = ["target", "cal_settings", "ml_config", "rnucleus", "server_settings", "model_params"]
        missing_sections = [s for s in required_sections if s not in data]
        if missing_sections:
            raise ValueError(f"配置文件缺少必需的节: {missing_sections}")
        return data

# Export all models
__all__ = [
    "MLCalConfig",
    "Target",
    "CalSettings",
    "Rnucleus",
    "StepControl",
    "ServerSettings",
    "ModelParams",
    "MlConfig",
    "CalPath",
]
