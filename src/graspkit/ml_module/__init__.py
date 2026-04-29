# -*- encoding: utf-8 -*-
from .neural_network import ANNClassifier
from .ml_initializer import (
    setup_logging,
    setup_directories,
    training_data_loader,
    ci_idx_data_processor,
    check_configuration_coupling,
    check_reference_energy_agreement,
    check_energy_convergence,
    evaluate_calculation_convergence,
    generate_train_csfs_descriptors,
    generate_regression_descriptors_from_config,
    get_stay_descriptors,
)


from .ml_trainer import (
    train_model,
    evaluate_model,
    handle_calculation_error,
    predict_model,
)

from .ml_results_analyzer import (
    validate_csf_desc_coverage,
    select_csfs_for_coverage,
    save_iteration_results,
    save_and_plot_results,
    ml_results_statistics
)

from .ml_regression_model import ANNRegressor
from .ml_regression_trainer import (
    generate_regression_train_descriptors,
    train_regression_model,
    evaluate_regression_model,
    predict_regression_model,
)

__all__ = [
    # neural_network
    "ANNClassifier",
    # ml_initializer
    "setup_logging",
    "setup_directories",
    "training_data_loader",
    "ci_idx_data_processor",
    "check_configuration_coupling",
    "check_reference_energy_agreement",
    "check_energy_convergence",
    "evaluate_calculation_convergence",
    "generate_train_csfs_descriptors",
    "generate_regression_descriptors_from_config",
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
]
