# -*- encoding: utf-8 -*-
"""
@Id :__init__.py
@date :2025/06/16 15:59:13
@author :YenochQin (秦毅)
"""

__author__ = "YenochQin (秦毅)"

from .CSFs_processor import (
    CSF_item_2_dict,
    CSFs_block_get_CSF,
    CSFs_sort_by_mix_coefficient,
    J_to_doubleJ,
    batch_asfs_mix_square_above_threshold,
    batch_blocks_csfs_final_coupling_J_collection,
    batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum,
    batch_process_csfs_parquet_to_descriptors,
    batch_process_csfs_to_descriptors,
    csf_J,
    generate_unique_random_numbers,
    parse_csf_2_descriptor,
    radom_choose_csfs,
    single_asf_csfs_final_coupling_J_mix_coefficient_sum,
    single_block_batch_asfs_CSFs_final_coupling_J_collection,
    union_lists_with_order,
)
from .data_IO import (
    BaseLoader,
    BinaryFileLoader,
    CSFLoader,
    EnergyFileLoader,
    LSJCompLoader,
    MixCoefLoader,
    RWFNFileLoader,
    TransitionLoader,
    MLCalConfig,
    CalPath,
    load_config,
    csfs_idxs_ci_loader,
    csfs_idxs_ci_storage,
    save_descriptors,
    save_descriptors_with_multi_block,
    scan_descriptors_polars,
    update_config,
    write_CSFs_pl_to_cfile,
    write_sorted_CSFs_to_cfile,
)
from .grasp_data_extractor import (
    format_compositions,
    format_configuration,
    format_energy_configurations,
    iterative_levels_collection,
    level_energy_collector,
    level_transition_data_processing,
    lsj_transition_data_level_location,
    merge_lsj_compositions,
    transition_data_level_location,
)
from .ml_module import (
    ANNClassifier,
    check_configuration_coupling,
    check_energy_convergence,
    ci_idx_data_processor,
    evaluate_calculation_convergence,
    evaluate_model,
    generate_train_csfs_descriptors,
    get_stay_descriptors,
    handle_calculation_error,
    ml_results_statistics,
    predict_model,
    save_and_plot_results,
    save_iteration_results,
    select_csfs_for_coverage,
    setup_directories,
    setup_logging,
    train_model,
    training_data_loader,
    validate_csf_desc_coverage,
    ANNRegressor,
    generate_regression_train_descriptors,
    train_regression_model,
    evaluate_regression_model,
    predict_regression_model,
)
from .utils import (
    CSFs,
    LS_shell_full_charged,
    MixCoefficientData,
    MLDataCounts,
    auto_plot_wavefunction_comparison,
    calculate_deformation,
    chunk_string,
    doubleJ_to_J,
    fig_settings,
    get_environment_config,
    inter_coupling_channel_bar,
    is_debug_mode,
    is_production_mode,
    is_slurm_environment,
    str_subshell_2_kappa,
)
from .version import __version__

__all__ = [
    # 版本信息
    "__author__",
    "__version__",
    # data_IO
    ## new loaders
    "BaseLoader",
    "BinaryFileLoader",
    "CSFLoader",
    "EnergyFileLoader",
    "LSJCompLoader",
    "MixCoefLoader",
    "RWFNFileLoader",
    "TransitionLoader",
    "MLCalConfig",
    "CalPath",
    ## produced_data_write
    "write_sorted_CSFs_to_cfile",
    "write_CSFs_pl_to_cfile",
    "update_config",
    "csfs_idxs_ci_storage",
    "save_descriptors",
    "save_descriptors_with_multi_block",
    ## processing_data_load
    "csfs_idxs_ci_loader",
    "load_config",
    "scan_descriptors_polars",
    # utils
    "MixCoefficientData",
    "CSFs",
    "MLDataCounts",
    # 工具函数
    "str_subshell_2_kappa",
    "doubleJ_to_J",
    "chunk_string",
    "calculate_deformation",
    "LS_shell_full_charged",
    ## CSFs_processor
    "batch_asfs_mix_square_above_threshold",
    "CSFs_block_get_CSF",
    "batch_blocks_csfs_final_coupling_J_collection",
    "single_asf_csfs_final_coupling_J_mix_coefficient_sum",
    "single_block_batch_asfs_CSFs_final_coupling_J_collection",
    "batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum",
    "union_lists_with_order",
    "CSFs_sort_by_mix_coefficient",
    "generate_unique_random_numbers",
    "radom_choose_csfs",
    ## CSFs_compress_extract
    "csf_J",
    "J_to_doubleJ",
    "CSF_item_2_dict",
    "parse_csf_2_descriptor",
    "batch_process_csfs_to_descriptors",
    "batch_process_csfs_parquet_to_descriptors",
    # grasp_data_extractor
    "format_configuration",
    "format_energy_configurations",
    "format_compositions",
    "iterative_levels_collection",
    "level_energy_collector",
    "merge_lsj_compositions",
    "lsj_transition_data_level_location",
    "transition_data_level_location",
    "level_transition_data_processing",
    # ml_module
    # neural_network
    "ANNClassifier",
    # ml_initializer
    "setup_logging",
    "setup_directories",
    "training_data_loader",
    "ci_idx_data_processor",
    "check_configuration_coupling",
    "check_energy_convergence",
    "evaluate_calculation_convergence",
    "generate_train_csfs_descriptors",
    "get_stay_descriptors",
    # ml_trainer
    "train_model",
    "evaluate_model",
    "predict_model",
    "handle_calculation_error",
    # ml_results_analyzer
    "validate_csf_desc_coverage",
    "select_csfs_for_coverage",
    "save_iteration_results",
    "save_and_plot_results",
    "ml_results_statistics",
    # ml_regression_model
    "ANNRegressor",
    # ml_regression_trainer
    "generate_regression_train_descriptors",
    "train_regression_model",
    "evaluate_regression_model",
    "predict_regression_model",
    # 环境配置和进度管理
    "get_environment_config",
    "is_slurm_environment",
    "is_debug_mode",
    "is_production_mode",
    # 作图
    "inter_coupling_channel_bar",
    "auto_plot_wavefunction_comparison",
]
