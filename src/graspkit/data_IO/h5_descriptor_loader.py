#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :h5_descriptor_load.py
@date :2025/08/18 23:29:40
@author :YenochQin (秦毅)
'''

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import h5py
import numpy as np

class CSFDescriptorError(Exception):
    """Custom exception for CSF descriptor related errors."""
    pass

def load_hdf5_descriptors(hdf5_file: str) -> Dict[str, np.ndarray]:
    """
    Load descriptors from C++ generated HDF5 file.
    
    Args:
        hdf5_file: Path to the HDF5 file
        
    Returns:
        Dictionary with 'descriptors' and 'labels' arrays
        
    Raises:
        CSFDescriptorError: If file cannot be read
    """
    if not os.path.isfile(hdf5_file):
        raise CSFDescriptorError(f"HDF5 file not found: {hdf5_file}")
        
    try:
        with h5py.File(hdf5_file, 'r') as f:
            results = {}
            if 'descriptors' in f:
                dataset = f['descriptors']
                if isinstance(dataset, h5py.Dataset):
                    results['descriptors'] = dataset[:]
            if 'labels' in f:
                dataset = f['labels']  
                if isinstance(dataset, h5py.Dataset):
                    results['labels'] = dataset[:]
            return results
    except Exception as e:
        raise CSFDescriptorError(f"Error reading HDF5 file: {e}")