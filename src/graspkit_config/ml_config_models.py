# -*- encoding: utf-8 -*-
"""Lightweight Pydantic configuration models for ML-driven CSF selection."""

from dataclasses import dataclass, field as dc_field
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class Target(BaseModel):
    """Target atom configuration."""

    atom: str
    conf: str
    full_CSFs_set_file: str
    presampled_csfs_file: str | None
    presampled_csfs_mix_file: str | None


class CalSettings(BaseModel):
    """Calculation settings."""

    continue_cal: bool = True
    cal_loop_num: int = 1
    cal_error_num: int = 0
    backward_loop_needed: bool = False
    target_backward_loop: int = 0
    cutoff_value: float
    sampling_ratio: float = 0.1
    expansion_ratio: float = 2.0
    energy_std_threshold: float = 1e-5
    csfs_num_relative_std_threshold: float = 0.002
    cal_method: str = "rci"
    cal_rwfn_file: str
    rci_n_quantum_for_self_energy: int
    root_path: Path
    cal_levels: str
    spectral_term: list[str]
    reference_energy_levels: list[float] = []
    reference_energy_mode: str = "monitor"
    reference_energy_score_weight: float = 0.3
    reference_energy_importance_weight: float = 0.7
    reference_gap_pair_weighting: str = "error_magnitude"
    reference_energy_hard_error_threshold: float | None = None
    reference_energy_top_pair_count: int | None = None
    reference_energy_threshold: float = 0.0
    reference_energy_threshold_base: float | None = None
    reference_energy_threshold_min: float = 300.0
    reference_energy_threshold_tighten_start_loop: int = 4
    reference_energy_threshold_decay: float = 0.85
    diff_ci_cutoff: float = 0.0
    use_regression_model: bool = False
    regression_min_clip: float = 1e-15

    @field_validator("root_path", mode="before")
    @classmethod
    def convert_root_path_to_path(cls, v: str | Path) -> Path:
        """Convert string paths into Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v

    @field_validator("cutoff_value")
    @classmethod
    def validate_cutoff_value(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("cutoff_value 必须大于 0")
        return v

    @field_validator("regression_min_clip")
    @classmethod
    def validate_regression_min_clip(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("regression_min_clip 必须大于 0")
        return v

    @field_validator("sampling_ratio")
    @classmethod
    def validate_sampling_ratio(cls, v: float) -> float:
        if not (0 < v <= 1):
            raise ValueError("sampling_ratio 必须在 (0, 1] 范围内")
        return v

    @field_validator("spectral_term")
    @classmethod
    def validate_spectral_term(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("spectral_term 必须是非空列表")
        return v

    @field_validator(
        "reference_energy_threshold",
        "reference_energy_threshold_min",
        "reference_energy_score_weight",
        "reference_energy_importance_weight",
        mode="before",
    )
    @classmethod
    def validate_nonnegative_reference_thresholds(cls, v: float) -> float:
        if v < 0:
            raise ValueError("reference_energy_threshold 相关参数必须大于等于 0")
        return v

    @field_validator("reference_energy_hard_error_threshold", mode="before")
    @classmethod
    def validate_reference_energy_hard_error_threshold(
        cls, v: float | None
    ) -> float | None:
        if v is not None and v < 0:
            raise ValueError("reference_energy_hard_error_threshold 必须大于等于 0")
        return v

    @field_validator("reference_energy_top_pair_count")
    @classmethod
    def validate_reference_energy_top_pair_count(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("reference_energy_top_pair_count 必须大于 0")
        return v

    @field_validator("reference_energy_mode")
    @classmethod
    def validate_reference_energy_mode(cls, v: str) -> str:
        allowed_modes = {"monitor", "hybrid_rank", "gate", "rollback"}
        if v not in allowed_modes:
            raise ValueError(
                "reference_energy_mode 必须是以下值之一: "
                f"{sorted(allowed_modes)}，当前值: {v}"
            )
        return v

    @field_validator("reference_gap_pair_weighting")
    @classmethod
    def validate_reference_gap_pair_weighting(cls, v: str) -> str:
        allowed_weighting = {"uniform", "error_magnitude", "error_squared"}
        if v not in allowed_weighting:
            raise ValueError(
                "reference_gap_pair_weighting 必须是以下值之一: "
                f"{sorted(allowed_weighting)}，当前值: {v}"
            )
        return v

    @field_validator("reference_energy_threshold_tighten_start_loop")
    @classmethod
    def validate_reference_threshold_start_loop(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("reference_energy_threshold_tighten_start_loop 必须大于 0")
        return v

    @field_validator("reference_energy_threshold_decay")
    @classmethod
    def validate_reference_threshold_decay(cls, v: float) -> float:
        if not (0 < v <= 1):
            raise ValueError("reference_energy_threshold_decay 必须在 (0, 1] 范围内")
        return v

    @model_validator(mode="after")
    def validate_reference_energy_levels(self) -> "CalSettings":
        if self.reference_energy_levels:
            if len(self.reference_energy_levels) != len(self.spectral_term):
                raise ValueError(
                    f"reference_energy_levels 长度({len(self.reference_energy_levels)}) "
                    f"必须与 spectral_term 长度({len(self.spectral_term)}) 相同"
                )
        if (
            self.reference_energy_score_weight == 0
            and self.reference_energy_importance_weight == 0
        ):
            raise ValueError(
                "reference_energy_score_weight 和 "
                "reference_energy_importance_weight 不能同时为零"
            )
        if self.reference_energy_threshold_base is None:
            self.reference_energy_threshold_base = self.reference_energy_threshold

        if self.reference_energy_threshold_base > 0:
            effective_floor = min(
                self.reference_energy_threshold_base,
                self.reference_energy_threshold_min,
            )
            if self.cal_loop_num >= self.reference_energy_threshold_tighten_start_loop:
                tighten_steps = (
                    self.cal_loop_num
                    - self.reference_energy_threshold_tighten_start_loop
                    + 1
                )
                tightened_threshold = self.reference_energy_threshold_base * (
                    self.reference_energy_threshold_decay**tighten_steps
                )
                self.reference_energy_threshold = max(
                    effective_floor, tightened_threshold
                )
            else:
                self.reference_energy_threshold = self.reference_energy_threshold_base
        return self


class ServerSettings(BaseModel):
    """Server settings."""

    slurm_partition: str
    tasks_per_node: int
    mpi_tmp_path: str | None = None
    cpu_threads: int = 16
    python_source: str
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
        """Validate consistency between python_source and environment fields."""
        if self.python_source == "uv":
            if self.uv_env_path is None:
                raise ValueError("当 python_source='uv' 时，必须提供 uv_env_path")
            if self.conda_path is not None or self.conda_env_name is not None:
                raise ValueError("当 python_source='uv' 时，conda_path 和 conda_env_name 应该为空")
        elif self.python_source == "conda":
            if self.conda_path is None:
                raise ValueError("当 python_source='conda' 时，必须提供 conda_path")
            if self.conda_env_name is None:
                raise ValueError("当 python_source='conda' 时，必须提供 conda_env_name")
            if self.uv_env_path is not None:
                raise ValueError("当 python_source='conda' 时，uv_env_path 应该为空")
        else:
            raise ValueError(
                f"python_source 必须是 'uv' 或 'conda'，当前值: {self.python_source}"
            )
        return self


class Rnucleus(BaseModel):
    """Nucleus configuration."""

    atomic_number: int
    mass_number: int
    atomic_mass: float
    nuclear_spin: int = 1
    nuclear_dipole: int = 1
    nuclear_quadrupole: int = 1


class StepControl(BaseModel):
    """Step Control."""

    enable_step_control: bool = False
    target_loop: int = 0
    start_step: str = "auto"
    end_step: str = "auto"
    skip_completed_steps: bool = False


class MlConfig(BaseModel):
    """Machine learning configuration."""

    high_prob_percentile: int = 95
    overfitting_threshold: float = 0.1
    underfitting_threshold: float = -0.05
    include_wrong_level_negatives: bool = True
    feature_selection: bool = True


class ModelParams(BaseModel):
    """Model parameters."""

    n_estimators: int = 1000
    random_state: int = 42


@dataclass
class CalPath:
    """Calculation paths populated by ``MLCalConfig.setup_paths()`` after config load."""

    full_CSFs_set_file_path: Path = dc_field(init=False)
    full_CSFs_set_parquet_path: Path = dc_field(init=False)
    full_CSFs_set_desc_path: Path = dc_field(init=False)
    full_CSFs_set_header_path: Path = dc_field(init=False)
    loop_file_name: str = dc_field(init=False)
    cal_loop_path: Path = dc_field(init=False)
    results_path: Path = dc_field(init=False)
    test_data_path: Path = dc_field(init=False)
    models_path: Path = dc_field(init=False)
    roc_curves_path: Path = dc_field(init=False)
    log_dir: Path = dc_field(init=False)
    iteration_results: Path = dc_field(init=False)
    training_results: Path = dc_field(init=False)
    accumulated_idxs_ci_path: Path = dc_field(init=False)
    previous_important_idxs_file: Path | None = dc_field(init=False, default=None)
    ml_results_path: Path | None = dc_field(init=False, default=None)


class MLCalConfig(BaseModel):
    """Root ML calculation configuration model."""

    model_config = {"arbitrary_types_allowed": True}

    target: Target
    cal_settings: CalSettings
    ml_config: MlConfig
    rnucleus: Rnucleus
    server_settings: ServerSettings
    model_params: ModelParams
    step_control: StepControl
    cal_path: CalPath = Field(default_factory=CalPath, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def validate_required_sections(
        cls, data: dict[str, int | float | str | bool | Path | None]
    ) -> dict[str, int | float | str | bool | Path | None]:
        """Validate required sections exist."""
        required_sections = [
            "target",
            "cal_settings",
            "ml_config",
            "rnucleus",
            "server_settings",
            "model_params",
        ]
        missing_sections = [
            section for section in required_sections if section not in data
        ]
        if missing_sections:
            raise ValueError(f"配置文件缺少必需的节: {missing_sections}")
        return data

    @model_validator(mode="after")
    def _setup_cal_path(self) -> "MLCalConfig":
        """Initialize derived paths after validation."""
        self.cal_path = CalPath()
        self.setup_paths()
        return self

    def setup_paths(self) -> None:
        """Populate all derived filesystem paths from the validated config."""
        root_path = Path(self.cal_settings.root_path)

        full_csfs_set_path = root_path / self.target.full_CSFs_set_file
        full_csfs_path_without_suffix = full_csfs_set_path.with_suffix("")
        self.cal_path.full_CSFs_set_file_path = full_csfs_set_path
        self.cal_path.full_CSFs_set_parquet_path = full_csfs_path_without_suffix.with_suffix(
            ".parquet"
        )
        self.cal_path.full_CSFs_set_desc_path = root_path / f"{self.target.conf}_desc"
        self.cal_path.full_CSFs_set_header_path = full_csfs_set_path.with_stem(
            f"{full_csfs_set_path.stem}_header"
        ).with_suffix(".toml")

        self.cal_path.loop_file_name = f"{self.target.conf}_{self.cal_settings.cal_loop_num}"
        self.cal_path.cal_loop_path = root_path / self.cal_path.loop_file_name
        self.cal_path.results_path = root_path / "results"
        self.cal_path.test_data_path = root_path / "test_data"
        self.cal_path.models_path = root_path / "models"
        self.cal_path.roc_curves_path = root_path / "roc_curves"
        self.cal_path.log_dir = root_path / "logs"
        self.cal_path.iteration_results = self.cal_path.results_path / "iteration_results.csv"
        self.cal_path.training_results = self.cal_path.results_path / "training_results.csv"
        self.cal_path.accumulated_idxs_ci_path = (
            self.cal_path.results_path / f"{self.target.conf}_merged_ci_squared.npz"
        )

        if self.cal_settings.cal_loop_num > 1:
            self.cal_path.previous_important_idxs_file = (
                self.cal_path.results_path
                / f"{self.target.conf}_{self.cal_settings.cal_loop_num - 1}_important_idxs"
            )
            self.cal_path.ml_results_path = (
                self.cal_path.results_path
                / f"{self.target.conf}_{self.cal_settings.cal_loop_num - 1}_final_sampled_idxs"
            )


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
