# -*- encoding: utf-8 -*-
"""
Package root for GraspKit.

Stage 0 of the package-splitting plan keeps the root import lightweight while
preserving legacy flat exports via lazy compatibility shims.
"""

from importlib import import_module
from typing import Any
from warnings import warn

from .version import __version__

__author__ = "YenochQin (秦毅)"

__all__ = [
    "__author__",
    "__version__",
]

_REMOVAL_VERSION = "GraspKit 3.4"

_LEGACY_EXPORTS: dict[str, tuple[str, str | None, str]] = {
    # data_IO
    "BaseLoader": ("graspkit.data_IO", "BaseLoader", "graspkit.data_IO.BaseLoader"),
    "BinaryFileLoader": (
        "graspkit.data_IO",
        "BinaryFileLoader",
        "graspkit.data_IO.BinaryFileLoader",
    ),
    "CSFLoader": ("graspkit.data_IO", "CSFLoader", "graspkit.data_IO.CSFLoader"),
    "EnergyFileLoader": (
        "graspkit.data_IO",
        "EnergyFileLoader",
        "graspkit.data_IO.EnergyFileLoader",
    ),
    "LSJCompLoader": (
        "graspkit.data_IO",
        "LSJCompLoader",
        "graspkit.data_IO.LSJCompLoader",
    ),
    "MixCoefLoader": (
        "graspkit.data_IO",
        "MixCoefLoader",
        "graspkit.data_IO.MixCoefLoader",
    ),
    "RWFNFileLoader": (
        "graspkit.data_IO",
        "RWFNFileLoader",
        "graspkit.data_IO.RWFNFileLoader",
    ),
    "TransitionLoader": (
        "graspkit.data_IO",
        "TransitionLoader",
        "graspkit.data_IO.TransitionLoader",
    ),
    "MLCalConfig": (
        "graspkit.data_IO",
        "MLCalConfig",
        "graspkit.data_IO.MLCalConfig",
    ),
    "CalPath": ("graspkit.data_IO", "CalPath", "graspkit.data_IO.CalPath"),
    "load_config": (
        "graspkit.data_IO",
        "load_config",
        "graspkit.data_IO.load_config",
    ),
    "csfs_idxs_ci_loader": (
        "graspkit.data_IO",
        "csfs_idxs_ci_loader",
        "graspkit.data_IO.csfs_idxs_ci_loader",
    ),
    "csfs_idxs_ci_storage": (
        "graspkit.data_IO",
        "csfs_idxs_ci_storage",
        "graspkit.data_IO.csfs_idxs_ci_storage",
    ),
    "save_descriptors": (
        "graspkit.data_IO",
        "save_descriptors",
        "graspkit.data_IO.save_descriptors",
    ),
    "save_descriptors_with_multi_block": (
        "graspkit.data_IO",
        "save_descriptors_with_multi_block",
        "graspkit.data_IO.save_descriptors_with_multi_block",
    ),
    "scan_descriptors_polars": (
        "graspkit.data_IO",
        "scan_descriptors_polars",
        "graspkit.data_IO.scan_descriptors_polars",
    ),
    "update_config": (
        "graspkit.data_IO",
        "update_config",
        "graspkit.data_IO.update_config",
    ),
    "write_CSFs_pl_to_cfile": (
        "graspkit.data_IO",
        "write_CSFs_pl_to_cfile",
        "graspkit.data_IO.write_CSFs_pl_to_cfile",
    ),
    "write_sorted_CSFs_to_cfile": (
        "graspkit.data_IO",
        "write_sorted_CSFs_to_cfile",
        "graspkit.data_IO.write_sorted_CSFs_to_cfile",
    ),
    # grasp_data_extractor
    "format_compositions": (
        "graspkit.grasp_data_extractor",
        "format_compositions",
        "graspkit.grasp_data_extractor.format_compositions",
    ),
    "format_configuration": (
        "graspkit.grasp_data_extractor",
        "format_configuration",
        "graspkit.grasp_data_extractor.format_configuration",
    ),
    "format_energy_configurations": (
        "graspkit.grasp_data_extractor",
        "format_energy_configurations",
        "graspkit.grasp_data_extractor.format_energy_configurations",
    ),
    "iterative_levels_collection": (
        "graspkit.grasp_data_extractor",
        "iterative_levels_collection",
        "graspkit.grasp_data_extractor.iterative_levels_collection",
    ),
    "level_energy_collector": (
        "graspkit.grasp_data_extractor",
        "level_energy_collector",
        "graspkit.grasp_data_extractor.level_energy_collector",
    ),
    "level_transition_data_processing": (
        "graspkit.grasp_data_extractor",
        "level_transition_data_processing",
        "graspkit.grasp_data_extractor.level_transition_data_processing",
    ),
    "lsj_transition_data_level_location": (
        "graspkit.grasp_data_extractor",
        "lsj_transition_data_level_location",
        "graspkit.grasp_data_extractor.lsj_transition_data_level_location",
    ),
    "merge_lsj_compositions": (
        "graspkit.grasp_data_extractor",
        "merge_lsj_compositions",
        "graspkit.grasp_data_extractor.merge_lsj_compositions",
    ),
    "transition_data_level_location": (
        "graspkit.grasp_data_extractor",
        "transition_data_level_location",
        "graspkit.grasp_data_extractor.transition_data_level_location",
    ),
    # CSFs_processor
    "CSF_item_2_dict": (
        "graspkit.CSFs_processor",
        "CSF_item_2_dict",
        "graspkit.CSFs_processor.CSF_item_2_dict",
    ),
    "CSFs_block_get_CSF": (
        "graspkit.CSFs_processor",
        "CSFs_block_get_CSF",
        "graspkit.CSFs_processor.CSFs_block_get_CSF",
    ),
    "CSFs_sort_by_mix_coefficient": (
        "graspkit.CSFs_processor",
        "CSFs_sort_by_mix_coefficient",
        "graspkit.CSFs_processor.CSFs_sort_by_mix_coefficient",
    ),
    "J_to_doubleJ": (
        "graspkit.CSFs_processor",
        "J_to_doubleJ",
        "graspkit.CSFs_processor.J_to_doubleJ",
    ),
    "batch_asfs_mix_square_above_threshold": (
        "graspkit.CSFs_processor",
        "batch_asfs_mix_square_above_threshold",
        "graspkit.CSFs_processor.batch_asfs_mix_square_above_threshold",
    ),
    "batch_blocks_csfs_final_coupling_J_collection": (
        "graspkit.CSFs_processor",
        "batch_blocks_csfs_final_coupling_J_collection",
        "graspkit.CSFs_processor.batch_blocks_csfs_final_coupling_J_collection",
    ),
    "batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum": (
        "graspkit.CSFs_processor",
        "batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum",
        "graspkit.CSFs_processor.batch_blocks_CSFs_final_coupling_J_mix_coefficient_sum",
    ),
    "batch_process_csfs_parquet_to_descriptors": (
        "graspkit.CSFs_processor",
        "batch_process_csfs_parquet_to_descriptors",
        "graspkit.CSFs_processor.batch_process_csfs_parquet_to_descriptors",
    ),
    "batch_process_csfs_to_descriptors": (
        "graspkit.CSFs_processor",
        "batch_process_csfs_to_descriptors",
        "graspkit.CSFs_processor.batch_process_csfs_to_descriptors",
    ),
    "csf_J": ("graspkit.CSFs_processor", "csf_J", "graspkit.CSFs_processor.csf_J"),
    "generate_unique_random_numbers": (
        "graspkit.CSFs_processor",
        "generate_unique_random_numbers",
        "graspkit.CSFs_processor.generate_unique_random_numbers",
    ),
    "parse_csf_2_descriptor": (
        "graspkit.CSFs_processor",
        "parse_csf_2_descriptor",
        "graspkit.CSFs_processor.parse_csf_2_descriptor",
    ),
    "radom_choose_csfs": (
        "graspkit.CSFs_processor",
        "radom_choose_csfs",
        "graspkit.CSFs_processor.radom_choose_csfs",
    ),
    "single_asf_csfs_final_coupling_J_mix_coefficient_sum": (
        "graspkit.CSFs_processor",
        "single_asf_csfs_final_coupling_J_mix_coefficient_sum",
        "graspkit.CSFs_processor.single_asf_csfs_final_coupling_J_mix_coefficient_sum",
    ),
    "single_block_batch_asfs_CSFs_final_coupling_J_collection": (
        "graspkit.CSFs_processor",
        "single_block_batch_asfs_CSFs_final_coupling_J_collection",
        "graspkit.CSFs_processor.single_block_batch_asfs_CSFs_final_coupling_J_collection",
    ),
    "union_lists_with_order": (
        "graspkit.CSFs_processor",
        "union_lists_with_order",
        "graspkit.CSFs_processor.union_lists_with_order",
    ),
    # ml_module
    "ANNClassifier": (
        "graspkit.ml_module",
        "ANNClassifier",
        "graspkit.ml_module.ANNClassifier",
    ),
    "check_configuration_coupling": (
        "graspkit.ml_module",
        "check_configuration_coupling",
        "graspkit.ml_module.check_configuration_coupling",
    ),
    "check_reference_energy_agreement": (
        "graspkit.ml_module",
        "check_reference_energy_agreement",
        "graspkit.ml_module.check_reference_energy_agreement",
    ),
    "check_energy_convergence": (
        "graspkit.ml_module",
        "check_energy_convergence",
        "graspkit.ml_module.check_energy_convergence",
    ),
    "ci_idx_data_processor": (
        "graspkit.ml_module",
        "ci_idx_data_processor",
        "graspkit.ml_module.ci_idx_data_processor",
    ),
    "evaluate_calculation_convergence": (
        "graspkit.ml_module",
        "evaluate_calculation_convergence",
        "graspkit.ml_module.evaluate_calculation_convergence",
    ),
    "evaluate_model": (
        "graspkit.ml_module",
        "evaluate_model",
        "graspkit.ml_module.evaluate_model",
    ),
    "generate_train_csfs_descriptors": (
        "graspkit.ml_module",
        "generate_train_csfs_descriptors",
        "graspkit.ml_module.generate_train_csfs_descriptors",
    ),
    "generate_regression_descriptors_from_config": (
        "graspkit.ml_module",
        "generate_regression_descriptors_from_config",
        "graspkit.ml_module.generate_regression_descriptors_from_config",
    ),
    "get_stay_descriptors": (
        "graspkit.ml_module",
        "get_stay_descriptors",
        "graspkit.ml_module.get_stay_descriptors",
    ),
    "handle_calculation_error": (
        "graspkit.ml_module",
        "handle_calculation_error",
        "graspkit.ml_module.handle_calculation_error",
    ),
    "ml_results_statistics": (
        "graspkit.ml_module",
        "ml_results_statistics",
        "graspkit.ml_module.ml_results_statistics",
    ),
    "predict_model": (
        "graspkit.ml_module",
        "predict_model",
        "graspkit.ml_module.predict_model",
    ),
    "save_and_plot_results": (
        "graspkit.ml_module",
        "save_and_plot_results",
        "graspkit.ml_module.save_and_plot_results",
    ),
    "save_iteration_results": (
        "graspkit.ml_module",
        "save_iteration_results",
        "graspkit.ml_module.save_iteration_results",
    ),
    "select_csfs_for_coverage": (
        "graspkit.ml_module",
        "select_csfs_for_coverage",
        "graspkit.ml_module.select_csfs_for_coverage",
    ),
    "setup_directories": (
        "graspkit.ml_module",
        "setup_directories",
        "graspkit.ml_module.setup_directories",
    ),
    "setup_logging": (
        "graspkit.ml_module",
        "setup_logging",
        "graspkit.ml_module.setup_logging",
    ),
    "train_model": (
        "graspkit.ml_module",
        "train_model",
        "graspkit.ml_module.train_model",
    ),
    "training_data_loader": (
        "graspkit.ml_module",
        "training_data_loader",
        "graspkit.ml_module.training_data_loader",
    ),
    "validate_csf_desc_coverage": (
        "graspkit.ml_module",
        "validate_csf_desc_coverage",
        "graspkit.ml_module.validate_csf_desc_coverage",
    ),
    "ANNRegressor": (
        "graspkit.ml_module",
        "ANNRegressor",
        "graspkit.ml_module.ANNRegressor",
    ),
    "generate_regression_train_descriptors": (
        "graspkit.ml_module",
        "generate_regression_train_descriptors",
        "graspkit.ml_module.generate_regression_train_descriptors",
    ),
    "train_regression_model": (
        "graspkit.ml_module",
        "train_regression_model",
        "graspkit.ml_module.train_regression_model",
    ),
    "evaluate_regression_model": (
        "graspkit.ml_module",
        "evaluate_regression_model",
        "graspkit.ml_module.evaluate_regression_model",
    ),
    "predict_regression_model": (
        "graspkit.ml_module",
        "predict_regression_model",
        "graspkit.ml_module.predict_regression_model",
    ),
    # lightweight utils
    "CSFs": ("graspkit.utils", "CSFs", "graspkit.utils.CSFs"),
    "LS_shell_full_charged": (
        "graspkit.utils",
        "LS_shell_full_charged",
        "graspkit.utils.LS_shell_full_charged",
    ),
    "MixCoefficientData": (
        "graspkit.utils",
        "MixCoefficientData",
        "graspkit.utils.MixCoefficientData",
    ),
    "MLDataCounts": ("graspkit.utils", "MLDataCounts", "graspkit.utils.MLDataCounts"),
    "calculate_deformation": (
        "graspkit.utils",
        "calculate_deformation",
        "graspkit.utils.calculate_deformation",
    ),
    "chunk_string": (
        "graspkit.utils",
        "chunk_string",
        "graspkit.utils.chunk_string",
    ),
    "doubleJ_to_J": (
        "graspkit.utils",
        "doubleJ_to_J",
        "graspkit.utils.doubleJ_to_J",
    ),
    "get_environment_config": (
        "graspkit.utils",
        "get_environment_config",
        "graspkit.utils.get_environment_config",
    ),
    "is_debug_mode": (
        "graspkit.utils",
        "is_debug_mode",
        "graspkit.utils.is_debug_mode",
    ),
    "is_production_mode": (
        "graspkit.utils",
        "is_production_mode",
        "graspkit.utils.is_production_mode",
    ),
    "is_slurm_environment": (
        "graspkit.utils",
        "is_slurm_environment",
        "graspkit.utils.is_slurm_environment",
    ),
    "str_subshell_2_kappa": (
        "graspkit.utils",
        "str_subshell_2_kappa",
        "graspkit.utils.str_subshell_2_kappa",
    ),
    # plotting compatibility
    "inter_coupling_channel_bar": (
        "graspkit.utils.plot_functions",
        "inter_coupling_channel_bar",
        "graspkit.utils.plot_functions.inter_coupling_channel_bar",
    ),
    "auto_plot_wavefunction_comparison": (
        "graspkit.utils.plot_functions",
        "auto_plot_wavefunction_comparison",
        "graspkit.utils.plot_functions.auto_plot_wavefunction_comparison",
    ),
    "fig_settings": (
        "graspkit.utils.fig_settings",
        None,
        "graspkit.utils.fig_settings",
    ),
}


def __getattr__(name: str) -> Any:
    legacy_export = _LEGACY_EXPORTS.get(name)
    if legacy_export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attribute_name, replacement = legacy_export
    warn(
        (
            f"`graspkit.{name}` is deprecated and will be removed no earlier than "
            f"{_REMOVAL_VERSION}. Import from `{replacement}` instead."
        ),
        FutureWarning,
        stacklevel=2,
    )

    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_LEGACY_EXPORTS))
