# -*- encoding: utf-8 -*-
"""
@Id :produced_data_write.py
@date :2025/06/16 16:15:12
@author :YenochQin (秦毅)
"""

from pathlib import Path
from typing import Any, cast

import pickle
from numpy.typing import NDArray
import rtoml

import numpy as np
import polars as pl


# TODO not good enough
def write_sorted_CSFs_to_cfile(
    CSFs_file_info: list[str], sorted_CSFs_data_list: list[list[list[str]]], output_file: str | Path
):
    """
    将排序后的CSFs数据写入到指定的输出文件中。

    Args:
        CSFs_file_info (list): CSFs文件的头部信息(CSF(s):行上面的信息)
        sorted_CSFs_data (list): 排序后的CSFs数据列表
            sorted_CSFs_data[block
                                [CSFs
                                    [CSFS_1]
                                    [CSFS_2]
                                    ...
                                    [CSFS_n]
                                ]
            ]
        output_file (str): 输出文件的路径。
    """
    if len(CSFs_file_info) != 4:
        raise ValueError("CSFs file header info error!")
    blocks_num = len(sorted_CSFs_data_list)
    with open(output_file, "w") as file:
        for line in CSFs_file_info:
            file.write(line)

        file.write("CSF(s):\n")
        for idx, block in enumerate(sorted_CSFs_data_list):
            if idx != blocks_num - 1:
                for csf in block:
                    for line in csf:
                        file.write(line)
                file.write(" *\n")
            else:
                for csf in block:
                    for line in csf:
                        file.write(line)
                        

def write_CSFs_pl_to_cfile(
    CSFs_file_info: list[str],
    CSFs_data_df: pl.DataFrame,
    output_file: str | Path
) -> None:
    if len(CSFs_file_info) != 5:
        raise ValueError("CSFs file header info error!")

    with open(output_file, "w") as file:
        for line in CSFs_file_info:
            file.write(line+"\n")
        
        for row in CSFs_data_df.select(["line1", "line2", "line3"]).iter_rows():
        # row 类似于 ("text1", "text2", "text3")
            file.write("\n".join(row)+ "\n")


#######################################################################


def update_config(config_path: str | Path, updates: dict[str, Any]):
    """更新TOML配置文件

    Args:
        config_path: 配置文件路径
        updates: 要更新的键值对字典，支持嵌套字典结构
                 例如：{'cal_settings': {'sampling_ratio': 0.1}}
    """
    # 确保路径是Path对象
    if isinstance(config_path, str):
        config_path = Path(config_path)

    # 使用 rtoml 读取TOML文件
    config = rtoml.load(config_path)

    # 深度更新配置值，保留缺失的参数
    for key, value in updates.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            # 递归更新嵌套字典
            for subkey, subvalue in cast(dict[str, Any], value).items():
                config[key][subkey] = subvalue
        else:
            # 更新或添加键值对
            config[key] = value

    # 写入配置文件
    rtoml.dump(config, config_path)

#######################################################################


def csfs_idxs_ci_storage(
    save_file_path: str | Path,
    csfs_idx: NDArray[np.int64],
    csfs_ci_squared: NDArray[np.float64],
    ) -> None:
    """
    将CSFs索引与CI系数平方值存储为 npz 文件。

    Args:
        save_file_path: 存储文件路径（无扩展名时自动补 .npz）
        csfs_idx: NDArray[np.int64]   CSFs整数索引（一维，长度 n_csfs）
        csfs_ci:  NDArray[np.float64] CI系数平方值，二维 (n_levels, n_csfs)
    """
    # 转换为Path对象并检查是否有扩展名
    file_path = Path(save_file_path)
    if not file_path.suffix:
        file_path = file_path.with_suffix(".npz")

    np.savez(
        file=file_path,
        idxs=csfs_idx,
        ci_squared=csfs_ci_squared
        )


#######################################################################


def save_descriptors(
    descriptors: np.ndarray, save_path: str | Path, file_format: str = "npy"
):
    """
    保存描述符数组

    Args:
        descriptors (np.ndarray): 描述符数组
        save_path (str | Path): 保存路径（不含扩展名）
        file_format (str): 保存格式 ('npy', 'csv', 'pkl')

    Example:
        >>> descriptors = batch_process_csfs_to_descriptors(csfs_data)
        >>> save_descriptors(descriptors, 'output/csf_descriptors', 'csv')
        >>> save_descriptors(descriptors, Path('output/csf_descriptors'), 'npy')
    """

    # 转换为Path对象
    save_path = Path(save_path)

    if file_format.lower() == "npy":
        file_path = save_path.parent / f"{save_path.name}_descriptors.npy"
        np.save(file_path, descriptors)
        print(f"Descriptors saved to: {file_path}")

    elif file_format.lower() == "csv":
        file_path = save_path.parent / f"{save_path.name}_descriptors.csv"
        pl.from_numpy(descriptors).write_csv(file_path)
        print(f"Descriptors saved to: {file_path}")

    elif file_format.lower() == "pkl":
        file_path = save_path.parent / f"{save_path.name}_descriptors.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(descriptors, f)
        print(f"Descriptors saved to: {file_path}")

    else:
        raise ValueError("file_format must be 'npy', 'csv', or 'pkl'")


def save_descriptors_with_multi_block(
    descriptors: np.ndarray,
    labels: np.ndarray,
    save_path: str | Path,
    file_format: str = "npy",
):
    """
    保存带标签的描述符数组

    Args:
        descriptors (np.ndarray): 描述符数组
        labels (np.ndarray): 标签数组
        save_path (str | Path): 保存路径（不含扩展名）
        file_format (str): 保存格式 ('csv', 'npy', 'pkl')

    Example:
        >>> X, y = batch_process_csfs_with_block_idxs(csfs_data)
        >>> save_descriptors_with_block_idxs(X, y, 'ml_data/features', 'csv')
        >>> save_descriptors_with_block_idxs(X, y, Path('ml_data/features'), 'npy')
    """

    # 转换为Path对象
    save_path = Path(save_path)

    if file_format.lower() == "csv":
        # CSV格式：将标签作为最后一列
        file_path = save_path.parent / f"{save_path.name}_descriptors_block_idxs.csv"
        pl.from_numpy(descriptors).with_columns(pl.Series("label", labels)).write_csv(file_path)
        print(f"Descriptors with labels saved to: {file_path}")

    elif file_format.lower() == "npy":
        # NPY格式：分别保存数据和标签
        data_path = save_path.parent / f"{save_path.name}_descriptors.npy"
        labels_path = (
            save_path.parent / f"{save_path.name}_descriptors_block_idxs.npy"
        )
        np.save(data_path, descriptors)
        np.save(labels_path, labels)
        print(f"Descriptors saved to: {data_path}")
        print(f"Labels saved to: {labels_path}")

    elif file_format.lower() == "pkl":
        # PKL格式：保存为字典
        file_path = save_path.parent / f"{save_path.name}_descriptors_block_idxs.pkl"
        data_dict = {"descriptors": descriptors, "labels": labels}
        with open(file_path, "wb") as f:
            pickle.dump(data_dict, f)
        print(f"Descriptors and labels saved to: {file_path}")

    else:
        raise ValueError("file_format must be 'csv', 'npy', or 'pkl'")
