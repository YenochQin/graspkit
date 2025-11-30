# -*- encoding: utf-8 -*-
"""
@Id :cpp_descriptor_wrapper.py
@date :2025/08/17
@author :YenochQin (秦毅)

Python wrapper for CSF Descriptor C++ program.

This module provides a convenient Python interface to the CSF descriptor
program, including support for running the program, reading results,
and batch processing.

"""

import os
import subprocess
import tempfile
import shutil
import numpy as np
import h5py
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class CSFDescriptorError(Exception):
    """Custom exception for CSF descriptor related errors."""

    pass


class CppDescriptorGenerator:
    """
    Python wrapper for the CSF descriptor C++ program.

    This class provides methods to run the CSF descriptor program,
    handle input/output operations, and process results.
    """

    def __init__(self, executable_path: Optional[str] = None):
        """
        Initialize the CSF wrapper.

        Args:
            executable_path: Path to the CSF descriptor executable.
                            If None, will search in common locations.
        """
        self.executable_path = self._find_executable(executable_path)
        if not self.executable_path:
            raise CSFDescriptorError("CSF descriptor executable not found")
        logger.info(f"Using C++ descriptor generator: {self.executable_path}")

    def _find_executable(self, custom_path: Optional[str] = None) -> Optional[str]:
        """Find the CSF descriptor executable."""
        if custom_path is not None:
            if os.path.isfile(custom_path) and os.access(custom_path, os.X_OK):
                return custom_path
            return None

        # Common locations to search
        search_paths = [
            "csf_descriptor",
            "/usr/bin/csf_descriptor",
            "/opt/bin/csf_descriptor",
            shutil.which("csf_descriptor"),
        ]

        for path in search_paths:
            if path is not None and os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        return None

    def generate_descriptors(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        cpu_threads: int = 0,
        quiet: bool = True,
        stream: bool = True,
        auto_config: bool = True,
        normalized: bool = True,
        debug: bool = False,
    ) -> Dict[str, Union[int, float, str]]:
        """
        Run the CSF descriptor program.

        Args:
            input_file: Path to the input CSF file
            output_file: Path for the output HDF5 file. If None, generates automatically
            cpu_threads: Number of threads to use (0 for auto-detection)
            quiet: Whether to suppress console output
            stream: Use streaming processing mode (for very large files)
            auto_config: Automatically optimize settings for file size
            normalized: Use new normalized descriptor scheme
            chunk_lines: Set chunk size in lines for streaming mode
            stream_memory: Set memory limit for streaming mode in MB
            estimate_time: Estimate processing time before starting
            debug: Enable debug mode with detailed error messages

        Returns:
            Dictionary with processing statistics

        Raises:
            CSFDescriptorError: If processing fails
        """
        if not os.path.isfile(input_file):
            raise CSFDescriptorError(f"Input file not found: {input_file}")

        # Generate output filename if not provided
        if output_file is None:
            input_path = Path(input_file)
            output_file = str(input_path.with_suffix(".h5"))

        # Build command
        cmd = [str(self.executable_path)]

        if cpu_threads > 0:
            cmd.extend(["-t", str(cpu_threads)])

        if quiet:
            cmd.append("-q")

        if output_file:
            cmd.extend(["-o", str(output_file)])

        # New streaming and processing options
        if stream:
            cmd.append("--stream")

        if auto_config:
            cmd.append("--auto-config")

        if normalized:
            cmd.append("--normalized")

        if debug:
            cmd.append("--debug")

        cmd.append(str(input_file))

        try:
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                raise CSFDescriptorError(f"CSF descriptor failed: {error_msg}")

            return {
                "input_file": input_file,
                "output_file": output_file,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.TimeoutExpired:
            raise CSFDescriptorError("CSF descriptor timed out")
        except FileNotFoundError:
            raise CSFDescriptorError(f"Executable not found: {self.executable_path}")

    def batch_process_with_multi_block(
        self,
        input_files: List[str],
        output_dir: str,
        cpu_threads: int = 0,
        verbose: bool = True,
        stream: bool = True,
        auto_config: bool = True,
        normalized: bool = True,
        debug: bool = False,
    ) -> List[Dict]:
        """
        Wrapper function compatible with existing Python API

        Args:
            input_files: List of input CSF files
            output_dir: Directory to save output files
            cpu_threads: Number of threads to use
            verbose: Whether to print progress
            stream: Use streaming processing mode (for very large files)
            auto_config: Automatically optimize settings for file size
            normalized: Use new normalized descriptor scheme
            chunk_lines: Set chunk size in lines for streaming mode
            stream_memory: Set memory limit for streaming mode in MB
            estimate_time: Estimate processing time before starting
            debug: Enable debug mode with detailed error messages
        Returns:
            List of processing results for each file
        """

        os.makedirs(output_dir, exist_ok=True)
        results = []
        for i, input_file in enumerate(input_files, 1):
            if verbose:
                print(f"Processing {i}/{len(input_files)}: {input_file}")

            try:
                input_path = Path(input_file)
                output_file = str(
                    Path(output_dir) / f"{input_path.stem}_descriptors.h5"
                )

                result = self.generate_descriptors(
                    input_file,
                    output_file,
                    cpu_threads=cpu_threads,
                    quiet=verbose,
                    stream=stream,
                    auto_config=auto_config,
                    normalized=normalized,
                    debug=debug,
                )
                results.append(result)

            except Exception as e:
                if verbose:
                    print(f"Error processing {input_file}: {e}")
                results.append({"input_file": input_file, "error": str(e)})

        return results

    def validate_input(self, input_file: str) -> bool:
        """
        Validate if a file is a valid CSF input file.

        Args:
            input_file: Path to the file to validate

        Returns:
            True if valid, False otherwise
        """
        if not os.path.isfile(input_file):
            return False

        try:
            with open(input_file, "r") as f:
                content = f.read()
                return "Core subshells:" in content and "Peel subshells:" in content
        except Exception:
            return False


# Convenience functions for backward compatibility
def batch_process_csfs_with_multi_block_cpp(
    input_files: List[str],
    output_dir: str,
    cpu_threads: int = 0,
    verbose: bool = True,
    stream: bool = True,
    auto_config: bool = True,
    normalized: bool = True,
    debug: bool = False,
) -> List[Dict]:
    """
    Drop-in replacement for batch_process_csfs_with_multi_block using C++ backend
    """
    generator = CppDescriptorGenerator()
    return generator.batch_process_with_multi_block(
        input_files,
        output_dir,
        cpu_threads=cpu_threads,
        verbose=verbose,
        stream=stream,
        auto_config=auto_config,
        normalized=normalized,
        debug=debug,
    )
