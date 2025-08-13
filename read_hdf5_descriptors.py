#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
读取C++ CSF描述符生成器输出的HDF5文件

使用示例:
    python read_hdf5_descriptors.py output.h5
    python read_hdf5_descriptors.py output.h5 --labels

需要安装:
    pip install h5py numpy
"""

import h5py
import numpy as np
import argparse
from pathlib import Path

def read_hdf5_descriptors(hdf5_file: str, read_labels: bool = False):
    """
    读取C++生成的HDF5描述符文件
    
    Args:
        hdf5_file (str): HDF5文件路径
        read_labels (bool): 是否读取标签数据
    
    Returns:
        tuple: (descriptors, labels) 如果read_labels=True
               descriptors 如果read_labels=False
    """
    file_path = Path(hdf5_file)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # 读取描述符数据
        if 'descriptors' not in f:
            raise KeyError("HDF5文件中未找到 'descriptors' 数据集")
        
        descriptors = f['descriptors'][:]
        
        # 获取属性信息
        attrs = f['descriptors'].attrs
        expected_rows = attrs.get('rows', descriptors.shape[0])
        expected_cols = attrs.get('cols', descriptors.shape[1])
        
        print(f"文件: {file_path}")
        print(f"描述符形状: {descriptors.shape}")
        print(f"数据类型: {descriptors.dtype}")
        print(f"预期维度: ({expected_rows}, {expected_cols})")
        
        # 验证维度
        if descriptors.shape != (expected_rows, expected_cols):
            print(f"警告: 实际维度 {descriptors.shape} 与预期维度 ({expected_rows}, {expected_cols}) 不匹配")
        
        if read_labels:
            if 'labels' not in f:
                print("警告: HDF5文件中未找到 'labels' 数据集")
                labels = None
            else:
                labels = f['labels'][:]
                print(f"标签形状: {labels.shape}")
                print(f"标签数据类型: {labels.dtype}")
                
                # 验证标签数量与描述符匹配
                if len(labels) != descriptors.shape[0]:
                    print(f"警告: 标签数量 {len(labels)} 与描述符数量 {descriptors.shape[0]} 不匹配")
            
            return descriptors, labels
        
        return descriptors

def print_file_info(hdf5_file: str):
    """打印HDF5文件的详细信息"""
    file_path = Path(hdf5_file)
    if not file_path.exists():
        print(f"文件不存在: {file_path}")
        return
    
    print(f"\n=== HDF5文件信息: {file_path.name} ===")
    
    with h5py.File(file_path, 'r') as f:
        def print_dataset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"数据集: {name}")
                print(f"  形状: {obj.shape}")
                print(f"  数据类型: {obj.dtype}")
                if obj.attrs:
                    print(f"  属性: {dict(obj.attrs)}")
            elif isinstance(obj, h5py.Group):
                print(f"组: {name}")
        
        f.visititems(print_dataset_info)
        
        # 打印文件根属性
        if f.attrs:
            print(f"文件属性: {dict(f.attrs)}")

def main():
    parser = argparse.ArgumentParser(description='读取C++ CSF描述符生成器输出的HDF5文件')
    parser.add_argument('hdf5_file', help='HDF5文件路径')
    parser.add_argument('--labels', action='store_true', help='同时读取标签数据')
    parser.add_argument('--info', action='store_true', help='仅显示文件信息')
    
    args = parser.parse_args()
    
    try:
        if args.info:
            print_file_info(args.hdf5_file)
        else:
            result = read_hdf5_descriptors(args.hdf5_file, args.labels)
            
            if isinstance(result, tuple):
                descriptors, labels = result
                print(f"\n描述符数据:")
                print(f"  形状: {descriptors.shape}")
                print(f"  数据类型: {descriptors.dtype}")
                print(f"  前5行:\n{descriptors[:5]}")
                
                if labels is not None:
                    print(f"\n标签数据:")
                    print(f"  形状: {labels.shape}")
                    print(f"  数据类型: {labels.dtype}")
                    print(f"  前10个: {labels[:10]}")
            else:
                descriptors = result
                print(f"\n描述符数据:")
                print(f"  形状: {descriptors.shape}")
                print(f"  数据类型: {descriptors.dtype}")
                print(f"  前5行:\n{descriptors[:5]}")
                
    except Exception as e:
        print(f"错误: {e}")

if __name__ == '__main__':
    main()