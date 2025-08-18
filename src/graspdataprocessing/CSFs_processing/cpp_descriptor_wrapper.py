#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :cpp_descriptor_wrapper.py
@date :2025/08/17
@author :YenochQin (秦毅)
@description : Python wrapper for C++ CSF descriptor generator with parallel processing
'''

import os
import subprocess
import tempfile
import shutil
import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CppDescriptorGenerator:
    """Wrapper class for C++ CSF descriptor generator"""
    
    def __init__(self, cpp_executable_path: Optional[str] = None):
        """
        Initialize the C++ descriptor generator wrapper
        
        Args:
            cpp_executable_path: Path to the csf_descriptor executable
        """
        if cpp_executable_path is None:
            # Try to find the executable in common locations
            possible_paths = [
                "/usr/bin/csf_descriptor",
                "csf_descriptor"
            ]
            
            for path in possible_paths:
                if os.path.isfile(path) and os.access(path, os.X_OK):
                    cpp_executable_path = path
                    break
            else:
                raise FileNotFoundError(
                    "Could not find csf_descriptor executable. Please install CSF descriptor or provide explicit path."
                )
        
        self.cpp_executable = cpp_executable_path
        logger.info(f"Using C++ descriptor generator: {self.cpp_executable}")
    
    def generate_descriptors(self, 
                           csf_file_path: str, 
                           with_subshell_info: bool = False,
                           cpu_threads: Optional[int] = None,
                           output_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate descriptors using C++ program
        
        Args:
            csf_file_path: Path to the input CSF file
            with_subshell_info: Whether to use extended descriptor format
            cpu_threads: Number of threads to use (None for auto-detect)
            output_dir: Directory to save output files (temp dir if None)
            
        Returns:
            Tuple of (descriptors_array, labels_array)
        """
        csf_path = Path(csf_file_path)
        if not csf_path.exists():
            raise FileNotFoundError(f"CSF file not found: {csf_path}")
        
        # Create temporary directory if not provided
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="csf_descriptor_")
            cleanup_temp = True
        else:
            temp_dir = output_dir
            cleanup_temp = False
            os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Build command in the format: csf_descriptor -t [threads] -e input.csf
            cmd = [self.cpp_executable]
            
            if cpu_threads is not None and cpu_threads > 0:
                cmd.extend(["-t", str(cpu_threads)])
            
            if with_subshell_info:
                cmd.append("-e")
            
            cmd.append(str(csf_path))
            
            # Run C++ program
            logger.info(f"Running C++ descriptor generator: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(
                    f"C++ descriptor generator failed with return code {result.returncode}.\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                )
            
            # Find output file - C++ program creates files with .h5 extension
            expected_output = csf_path.with_suffix('.h5')
            if expected_output.exists():
                output_file = expected_output
            else:
                # Try alternative naming patterns
                alt_output = csf_path.with_name(f"{csf_path.stem}_descriptors.h5")
                if alt_output.exists():
                    output_file = alt_output
                else:
                    # Look for any .h5 file in the same directory
                    h5_files = list(csf_path.parent.glob("*.h5"))
                    if h5_files:
                        output_file = h5_files[0]
                    else:
                        raise FileNotFoundError(f"No HDF5 output file found after C++ processing")
            
            descriptors, labels = self._read_hdf5_output(output_file)
            
            logger.info(f"Successfully generated {len(descriptors)} descriptors using C++")
            return descriptors, labels
            
        finally:
            # Clean up temporary directory
            if cleanup_temp and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def _read_hdf5_output(self, hdf5_file: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Read descriptors and labels from HDF5 file"""
        try:
            with h5py.File(hdf5_file, 'r') as f:
                if '/descriptors' not in f:
                    raise KeyError("'/descriptors' dataset not found in HDF5 file")
                
                descriptors_ds = f['/descriptors']
                if not isinstance(descriptors_ds, h5py.Dataset):
                    raise TypeError("'/descriptors' is not a valid HDF5 dataset")
                
                descriptors = np.array(descriptors_ds[:])
                
                if '/labels' in f:
                    labels_ds = f['/labels']
                    if not isinstance(labels_ds, h5py.Dataset):
                        raise TypeError("'/labels' is not a valid HDF5 dataset")
                    labels = np.array(labels_ds[:])
                else:
                    # Generate sequential labels if not provided
                    labels = np.arange(len(descriptors))
                return descriptors, labels
        except Exception as e:
            raise RuntimeError(f"Failed to read HDF5 output: {e}")
    
    def batch_process_with_multi_block(self, 
            csfs_file_data,
            label_type: str = 'sequential',
            with_subshell_info: bool = False,
            cpu_threads: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper function compatible with existing Python API
        
        Args:
            csfs_file_data: CSFs file data object (from GraspFileLoad)
            label_type: Label type ('sequential', 'block', 'global_sequential')
            with_subshell_info: Whether to use extended descriptor format
            cpu_threads: Number of threads to use
            
        Returns:
            Tuple of (descriptors_array, labels_array)
        """
        # Save CSF data to temporary file for C++ processing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csf', delete=False) as temp_file:
            temp_csf_path = temp_file.name
            
            # Write CSF data in GRASP format
            self._write_csf_file(csfs_file_data, temp_csf_path)
        
        try:
            # Generate descriptors using C++
            descriptors, _ = self.generate_descriptors(
                temp_csf_path,
                with_subshell_info=with_subshell_info,
                cpu_threads=cpu_threads
            )
            
            # Generate labels according to label_type
            labels = self._generate_labels(csfs_file_data, label_type)
            
            # Ensure labels match descriptors
            if len(labels) != len(descriptors):
                raise RuntimeError(
                    f"Label count ({len(labels)}) doesn't match descriptor count ({len(descriptors)})"
                )
            
            return descriptors, np.array(labels)
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_csf_path):
                os.unlink(temp_csf_path)
    
    def _write_csf_file(self, csfs_file_data, output_path: str):
        """Write CSF data to file in GRASP format"""
        with open(output_path, 'w') as f:
            # Write subshell information
            for line in csfs_file_data.subshell_info_raw:
                f.write(line + '\n')
            
            # Write CSF data
            for block_idx, block in enumerate(csfs_file_data.CSFs_block_data):
                for csf_item in block:
                    for line in csf_item:
                        f.write(line + '\n')
                # Add separator between blocks
                if block_idx < len(csfs_file_data.CSFs_block_data) - 1:
                    f.write('*\n')
    
    def _generate_labels(self, csfs_file_data, label_type: str) -> list:
        """Generate labels based on label_type"""
        labels = []
        global_counter = 0
        
        for block_idx, block in enumerate(csfs_file_data.CSFs_block_data):
            for csf_idx, _ in enumerate(block):
                if label_type == 'block':
                    labels.append(block_idx)
                elif label_type == 'sequential':
                    labels.append(csf_idx)
                elif label_type == 'global_sequential':
                    labels.append(global_counter)
                else:  # custom
                    labels.append(f"block_{block_idx}_csf_{csf_idx}")
                global_counter += 1
        
        return labels

# Convenience functions for backward compatibility
def batch_process_csfs_with_multi_block_cpp(
                csfs_file_data,
                label_type: str = 'sequential',
                with_subshell_info: bool = False,
                cpu_threads: Optional[int] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop-in replacement for batch_process_csfs_with_multi_block using C++ backend
    """
    generator = CppDescriptorGenerator()
    return generator.batch_process_with_multi_block(
        csfs_file_data, label_type, with_subshell_info, cpu_threads
)

def batch_process_csfs_to_descriptors_cpp(
                csfs_file_data,
                with_subshell_info: bool = False,
                cpu_threads: Optional[int] = None
            ) -> np.ndarray:
    """
    Drop-in replacement for batch_process_csfs_to_descriptors using C++ backend
    """
    generator = CppDescriptorGenerator()
    descriptors, _ = generator.batch_process_with_multi_block(
        csfs_file_data, label_type='sequential', 
        with_subshell_info=with_subshell_info, cpu_threads=cpu_threads
    )
    return descriptors