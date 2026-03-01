# -*- encoding: utf-8 -*-
'''
@Id :ml_cal_config_module.py
@date :2026/02/20 15:48:54
@author :YenochQin (秦毅)
'''
from pathlib import Path
from dataclasses import dataclass, field as dc_field
from pydantic import BaseModel, Field, field_validator, model_validator


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


# CalPath dataclass - 所有路径属性（由 MLCalConfig.setup_paths() 计算，不来自 TOML）
@dataclass
class CalPath:
    """Calculation paths populated by MLCalConfig.setup_paths() after config load."""

    # 全量CSF集合相关路径
    full_CSFs_set_file_path: Path = dc_field(init=False)
    full_CSFs_set_parquet_path: Path = dc_field(init=False)
    full_CSFs_set_desc_path: Path = dc_field(init=False)
    full_CSFs_set_header_path: Path = dc_field(init=False)

    # 循环相关路径
    loop_file_name: str = dc_field(init=False)
    cal_loop_path: Path = dc_field(init=False)

    # 结果目录路径
    results_path: Path = dc_field(init=False)
    test_data_path: Path = dc_field(init=False)
    models_path: Path = dc_field(init=False)
    roc_curves_path: Path = dc_field(init=False)
    log_dir: Path = dc_field(init=False)

    # 结果文件路径
    iteration_results: Path = dc_field(init=False)
    training_results: Path = dc_field(init=False)
    accumulated_idxs_ci_path: Path = dc_field(init=False)

    # 历史数据路径（仅 cal_loop_num > 1 时赋值）
    previous_important_idxs_file: Path | None = dc_field(init=False, default=None)
    ml_results_path: Path | None = dc_field(init=False, default=None)


# Root MLCalConfig model
class MLCalConfig(BaseModel):
    """Root ML calculation configuration model"""

    model_config = {"arbitrary_types_allowed": True}

    target: Target
    cal_settings: CalSettings
    ml_config: MlConfig
    rnucleus: Rnucleus
    server_settings: ServerSettings
    model_params: ModelParams
    step_control: StepControl

    # cal_path 不来自 TOML，由 model_validator(mode="after") 在所有字段验证完成后自动计算
    cal_path: CalPath = Field(default_factory=CalPath, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def validate_required_sections(cls, data: dict[str, int | float | str | bool | Path | None]) -> dict[str, int | float | str | bool | Path | None]:
        """Validate required sections exist"""
        required_sections = ["target", "cal_settings", "ml_config", "rnucleus", "server_settings", "model_params"]
        missing_sections = [s for s in required_sections if s not in data]
        if missing_sections:
            raise ValueError(f"配置文件缺少必需的节: {missing_sections}")
        return data

    @model_validator(mode="after")
    def _setup_cal_path(self) -> "MLCalConfig":
        """所有字段验证完成后自动初始化所有文件路径"""
        self.cal_path = CalPath()
        self.setup_paths()
        return self

    def setup_paths(self) -> None:
        """
        为配置对象设置所有必需的文件路径

        该函数设置以下路径：
        1. 全量CSF集合相关文件路径
        2. 压缩的二进制CSF文件路径
        3. 当前计算循环的路径
        4. 结果文件存储路径
        5. 如果是后续循环，设置前一轮的重要索引和ML结果路径
        """
        root_path: Path = Path(self.cal_settings.root_path)

        # 设置全量CSF集合文件的完整路径
        full_CSFs_set_path: Path = root_path / self.target.full_CSFs_set_file
        full_CSFs_path_without_suffix: Path = full_CSFs_set_path.with_suffix("")
        self.cal_path.full_CSFs_set_file_path = full_CSFs_set_path
        # 设置CSF二进制和头文件的路径
        self.cal_path.full_CSFs_set_parquet_path = full_CSFs_path_without_suffix.with_suffix(".parquet")
        self.cal_path.full_CSFs_set_desc_path = root_path / f"{self.target.conf}_desc"
        self.cal_path.full_CSFs_set_header_path = full_CSFs_set_path.with_stem(
            f"{full_CSFs_set_path.stem}_header"
        ).with_suffix(".toml")

        self.cal_path.loop_file_name = (
            f"{self.target.conf}_{self.cal_settings.cal_loop_num}"
        )

        # 设置当前计算循环的工作目录路径，格式：{配置名}_{循环编号}
        self.cal_path.cal_loop_path = root_path / self.cal_path.loop_file_name

        # 设置计算结果文件的存储路径
        self.cal_path.results_path = root_path / "results"
        self.cal_path.test_data_path = root_path / "test_data"
        self.cal_path.models_path = root_path / "models"
        self.cal_path.roc_curves_path = root_path / "roc_curves"
        self.cal_path.log_dir = root_path / "logs"

        self.cal_path.iteration_results = (
            self.cal_path.results_path / "iteration_results.csv"
        )
        self.cal_path.training_results = (
            self.cal_path.results_path / "training_results.csv"
        )
        self.cal_path.accumulated_idxs_ci_path = (
            self.cal_path.results_path / f"{self.target.conf}_merged_ci_squared.npz"
        )

        # 如果是第二轮及之后的计算循环，需要设置前一轮的相关文件路径
        if self.cal_settings.cal_loop_num > 1:
            # 前一轮计算保存的重要索引文件路径
            self.cal_path.previous_important_idxs_file = (
                self.cal_path.results_path
                / f"{self.target.conf}_{self.cal_settings.cal_loop_num - 1}_important_idxs"
            )
            # 前一轮机器学习生成的最终采样索引文件路径
            self.cal_path.ml_results_path = (
                self.cal_path.results_path
                / f"{self.target.conf}_{self.cal_settings.cal_loop_num - 1}_final_sampled_idxs"
            )

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
