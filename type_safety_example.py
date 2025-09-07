#!/usr/bin/env python3
"""
Example demonstrating the improved type safety in GraspFileLoad class.

This example shows how to use the new type-safe helper methods to avoid
type warnings and ensure proper data handling.
"""

from pathlib import Path
from typing import List, cast
import pandas as pd

# Import the GraspFileLoad class
from src.graspkit.data_IO.grasp_data_loader import GraspFileLoad
from src.graspkit.utils.data_modules import CSFs, MixCoefficientData

def example_type_safe_usage():
    """Demonstrate type-safe usage of GraspFileLoad"""
    
    # Example 1: Loading level data with type safety
    print("Example 1: Loading level data")
    level_loader = GraspFileLoad({
        'atom': 'Fe',
        'file_dir': './data',
        'file_type': 'ENERGY', 
        'level_parameter': '1',
        'this_as': 0
    })
    
    # Method 1: Use the type-safe helper method (RECOMMENDED)
    try:
        level_data = level_loader.get_level_data()  # Type: List[str]
        print(f"✅ Level data loaded: {len(level_data)} lines")
    except TypeError as e:
        print(f"❌ Type error: {e}")
    
    # Method 2: Use the general method with explicit type checking
    result = level_loader.data_file_process()
    if isinstance(result, list) and all(isinstance(item, str) for item in result):
        level_data = result  # Type is now narrowed to List[str]
        print(f"✅ Level data verified: {len(level_data)} lines")
    else:
        print(f"❌ Unexpected return type: {type(result)}")
    
    # Example 2: Loading plot data with type safety
    print("\nExample 2: Loading plot data")
    plot_loader = GraspFileLoad({
        'atom': 'Fe',
        'file_dir': './data',
        'file_type': 'PLOT',
        'level_parameter': '1', 
        'this_as': 0
    })
    
    # Method 1: Use the type-safe helper method (RECOMMENDED)
    try:
        plot_data = plot_loader.get_plot_data()  # Type: pd.DataFrame
        print(f"✅ Plot data loaded: {plot_data.shape}")
    except TypeError as e:
        print(f"❌ Type error: {e}")
    
    # Method 2: Use the general method with explicit type checking
    result = plot_loader.data_file_process()
    if isinstance(result, pd.DataFrame):
        plot_data = result  # Type is now narrowed to pd.DataFrame
        print(f"✅ Plot data verified: {plot_data.shape}")
    else:
        print(f"❌ Unexpected return type: {type(result)}")
    
    # Example 3: Loading CSF data with type safety
    print("\nExample 3: Loading CSF data")
    csf_loader = GraspFileLoad({
        'atom': 'Fe',
        'file_dir': './data',
        'file_type': 'CSF',
        'level_parameter': '1',
        'this_as': 0
    })
    
    # Method 1: Use the type-safe helper method (RECOMMENDED)
    try:
        csf_data = csf_loader.get_csfs_data()  # Type: CSFs
        print(f"✅ CSF data loaded: {csf_data.block_num} blocks")
    except TypeError as e:
        print(f"❌ Type error: {e}")
    
    # Method 2: Use the general method with explicit type checking
    result = csf_loader.data_file_process()
    if isinstance(result, CSFs):
        csf_data = result  # Type is now narrowed to CSFs
        print(f"✅ CSF data verified: {csf_data.block_num} blocks")
    else:
        print(f"❌ Unexpected return type: {type(result)}")

def example_error_handling():
    """Demonstrate error handling for type mismatches"""
    
    print("\nExample 4: Error handling")
    
    # This would cause a type error if the file doesn't contain the expected data
    loader = GraspFileLoad({
        'atom': 'Fe',
        'file_dir': './data', 
        'file_type': 'ENERGY',
        'level_parameter': '1',
        'this_as': 0
    })
    
    # Simulate a type mismatch by calling the wrong helper method
    try:
        # This will fail if the data is not actually CSF data
        csf_data = loader.get_csfs_data()
    except TypeError as e:
        print(f"✅ Caught expected type error: {e}")
    
    # Better approach: check the file_type first
    if loader.file_type.upper() in ["ENERGY", "LEVEL"]:
        level_data = loader.get_level_data()
        print(f"✅ Correctly identified as level data: {len(level_data)} lines")

def example_best_practices():
    """Demonstrate best practices for avoiding type warnings"""
    
    print("\nExample 5: Best practices")
    
    # Best practice 1: Use the type-safe helper methods
    loader = GraspFileLoad({
        'atom': 'Fe',
        'file_dir': './data',
        'file_type': 'MIX_COEFFICIENT',
        'level_parameter': '1',
        'this_as': 0
    })
    
    # This eliminates type warnings completely
    mix_data = loader.get_mix_coefficient_data()
    print(f"✅ Mix coefficient data: {mix_data.block_num} blocks")
    
    # Best practice 2: Use isinstance checks before assignment
    result = loader.data_file_process()
    if isinstance(result, MixCoefficientData):
        mix_data = result  # No type warning here
        print(f"✅ Mix coefficient data verified: {mix_data.block_num} blocks")
    
    # Best practice 3: Use cast() when you're certain about the type
    result = loader.data_file_process()
    if isinstance(result, MixCoefficientData):
        mix_data = cast(MixCoefficientData, result)  # Explicit cast
        print(f"✅ Mix coefficient data cast: {mix_data.block_num} blocks")

if __name__ == "__main__":
    print("=== GraspFileLoad Type Safety Examples ===\n")
    
    example_type_safe_usage()
    example_error_handling() 
    example_best_practices()
    
    print("\n=== Summary ===")
    print("✅ Use type-safe helper methods (get_*_data) to avoid type warnings")
    print("✅ Use isinstance() checks to narrow types at runtime")
    print("✅ Use cast() when you're certain about the type")
    print("✅ Handle TypeError exceptions for robust error handling")
    print("✅ Check file_type before calling specific methods")