# -*- encoding: utf-8 -*-
"""
@Id :__init__.py
@date :2025/06/16 15:59:17
@author :YenochQin (秦毅)
"""

from .neural_network import ANNClassifier
from .ml_initializer import (
    setup_config,
    setup_logging,
    setup_directories,
    initialize_iteration_results_csv,
    training_data_loader,
    check_configuration_coupling,
    check_energy_convergence,
    evaluate_calculation_convergence,
    generate_train_csfs_descriptors,
    get_stay_descriptors,
)


from .ml_trainer import (
    train_model,
    evaluate_model,
    handle_calculation_error,
)

from .ml_results_analyzer import (
    validate_csf_descriptors_coverage,
    select_csfs_for_coverage,
    save_iteration_results,
    save_and_plot_results,
)

__all__ = [
    # neural_network
    "ANNClassifier",
    # ml_initializer
    "setup_config",
    "setup_logging",
    "setup_directories",
    "initialize_iteration_results_csv",
    "training_data_loader",
    "check_configuration_coupling",
    "check_energy_convergence",
    "evaluate_calculation_convergence",
    "generate_train_csfs_descriptors",
    "get_stay_descriptors",
    # ml_trainer
    "train_model",
    "evaluate_model",
    "handle_calculation_error",
    # ml_results_analyzer
    "validate_csf_descriptors_coverage",
    "select_csfs_for_coverage",
    "save_iteration_results",
    "save_and_plot_results",
]
