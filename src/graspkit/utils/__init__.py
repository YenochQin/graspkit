# -*- encoding: utf-8 -*-
"""
@Id :__init__.py
@date :2025/06/16 15:59:13
@author :YenochQin (秦毅)
"""

from .data_modules import CSFs, MixCoefficientData, MLDataCounts
from .environment_config import (
    get_environment_config,
    is_debug_mode,
    is_production_mode,
    is_slurm_environment,
)
from .plot_functions import (
    auto_plot_wavefunction_comparison,
    inter_coupling_channel_bar,
)

from .quadrupole_deformation import (
    calculate_deformation,
)
from .tool_function import (
    LS_shell_full_charged,
    chunk_string,
    doubleJ_to_J,
    str_subshell_2_kappa,
)

__all__ = [
    # 数据类
    "MixCoefficientData",
    "CSFs",
    "MLDataCounts",
    # 工具函数
    "str_subshell_2_kappa",
    "doubleJ_to_J",
    "chunk_string",
    "LS_shell_full_charged",
    # 环境配置
    "get_environment_config",
    "is_slurm_environment",
    "is_debug_mode",
    "is_production_mode",
    # 四极形变
    "calculate_deformation",
    # 作图
    "inter_coupling_channel_bar",
    "auto_plot_wavefunction_comparison",
]
