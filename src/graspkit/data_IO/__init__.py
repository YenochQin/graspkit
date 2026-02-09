# -*- encoding: utf-8 -*-
"""
@Id :__init__.py
@date :2025/06/16 15:59:13
@author :YenochQin (秦毅)
"""

# New loader classes
from .cpp_descriptor_wrapper import (
    CppDescriptorGenerator,
    batch_process_csfs_with_multi_block_cpp,
)
from .h5_descriptor_loader import load_hdf5_descriptors
from .loaders.base_loader import BaseLoader  # noqa: F401
from .loaders.binary_file_loader import BinaryFileLoader  # noqa: F401
from .loaders.csf_loader import CSFLoader  # noqa: F401
from .loaders.energy_file_loader import EnergyFileLoader  # noqa: F401
from .loaders.lsj_comp_loader import LSJCompLoader  # noqa: F401
from .loaders.mix_coef_loader import MixCoefLoader  # noqa: F401
from .loaders.radial_wavefunction_loader import RWFNFileLoader  # noqa: F401
from .loaders.transition_loader import TransitionLoader  # noqa: F401
from .processing_data_loader import (
    load_config,
    load_csf_metadata,
    load_csfs_binary,
    load_descriptors,
    load_descriptors_with_multi_block,
    load_large_hash,
    pkl_loader,
    scan_descriptors_polars,
)
from .produced_data_writor import (
    continue_calculate,
    pkl_storage,
    precompute_large_hash,
    save_csf_metadata,
    save_csfs_binary,
    save_descriptors,
    save_descriptors_with_multi_block,
    update_config,
    write_CSFs_pl_to_cfile,
    write_sorted_CSFs_to_cfile,
)

# 显式导出所有需要的函数
__all__ = [
    # New loader classes
    "BaseLoader",
    "BinaryFileLoader",
    "CSFLoader",
    "EnergyFileLoader",
    "LSJCompLoader",
    "MixCoefLoader",
    "RWFNFileLoader",
    "TransitionLoader",
    # produced_data_write
    "write_sorted_CSFs_to_cfile",
    "write_CSFs_pl_to_cfile",
    "save_csf_metadata",
    "save_csfs_binary",
    "continue_calculate",
    "update_config",
    "pkl_storage",
    "precompute_large_hash",
    "save_descriptors",
    "save_descriptors_with_multi_block",
    # processing_data_load
    "load_csf_metadata",
    "load_csfs_binary",
    "pkl_loader",
    "load_large_hash",
    "load_config",
    "load_descriptors",
    "load_descriptors_with_multi_block",
    "scan_descriptors_polars",
    # cpp_descriptor_wrapper
    "CppDescriptorGenerator",
    "batch_process_csfs_with_multi_block_cpp",
    # h5_descriptor_load
    "load_hdf5_descriptors",
]
