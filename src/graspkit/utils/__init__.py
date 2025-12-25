#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :__init__.py
@date :2025/06/16 15:59:13
@author :YenochQin (秦毅)
'''

from .data_modules import (
    MixCoefficientData,
    CSFs,
    MLDataCounts
)

from .tool_function import (
    level_print_title,
    level_J_value,
    level_parity,
    energy_au_cm,
    align_2d_list_columns,
    int_nl_2_str_nl,
    str_subshell_2_kappa,
    doubleJ_to_J,
    read_fortran_record,
    chunk_string,
    level_data_compare,
    LS_shell_full_charged,
)

from .environment_config import (
    get_environment_config,
    is_slurm_environment,
    is_debug_mode,
    is_production_mode
)

from .progress_manager import (
    create_progress_bar,
    wrap_iterator,
    progress_range,
    progress_context,
    log_stage_start,
    log_stage_end
)

from .quadrupole_deformation import (
    calculate_deformation,
)

from .plot_functions import (
    inter_coupling_channel_bar,
    auto_plot_wavefunction_comparison
)

__all__ = [
    # 数据类
    'MixCoefficientData',
    'CSFs',
    "MLDataCounts",
    
    # 工具函数
    'level_print_title',
    'level_J_value',
    'level_parity',
    'energy_au_cm',
    'align_2d_list_columns',
    'int_nl_2_str_nl',
    'str_subshell_2_kappa',
    'doubleJ_to_J',
    'read_fortran_record',
    'chunk_string',
    'level_data_compare',
    'LS_shell_full_charged',
    
    # 环境配置
    'get_environment_config',
    'is_slurm_environment',
    'is_debug_mode',
    'is_production_mode',
    
    # 进度管理
    'create_progress_bar',
    'wrap_iterator',
    'progress_range',
    'progress_context',
    'log_stage_start',
    'log_stage_end',
    
    # 四极形变
    'calculate_deformation',

    # 作图
    "inter_coupling_channel_bar",
    "auto_plot_wavefunction_comparison"
]