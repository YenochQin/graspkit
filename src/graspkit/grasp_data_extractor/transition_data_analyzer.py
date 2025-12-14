#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Id :transition_data_analyzer.py
@date :2025/12/14 16:17:38
@author :YenochQin (秦毅)
'''
import pandas as pd
import numpy as np
from ..utils.progress_manager import wrap_iterator
######################################################################

def lsj_transition_data_level_location(
        transition_data_df : pd.DataFrame, 
        level_df : pd.DataFrame, 
        level_file_parameters : dict
    ) -> pd.DataFrame:
    
    transition_data_df['Upper_level_location'] = 0
    transition_data_df['Upper_level_location'] = transition_data_df['Upper_level_location'].astype('int')
    transition_data_df['Lower_level_location'] = 0
    transition_data_df['Lower_level_location'] = transition_data_df['Lower_level_location'].astype('int')
    level_paramenter = level_file_parameters.get('level_parameter')
    level_as = level_file_parameters.get('this_as')
    
    level_conf_column = f"Configuration_{level_paramenter}{level_as}raw"
    
    def get_level_idx(level_J : str, level_conf : str, level_df : pd.DataFrame) -> int:
        
        level_idx = level_df[(level_df['J'] == level_J) & (level_df[f'{level_conf_column}'] == level_conf)].idx.values[0]
        
        return level_idx

    for i, (_, row) in enumerate(transition_data_df.iterrows()):
        print(row["Upper_J"], row["Upper_configuration"])
        temp_upper_idx = get_level_idx(row['Upper_J'], row['Upper_configuration'], level_df)

        print(row['Lower_J'], row["Lower_configuration"])
        temp_lower_idx = get_level_idx(row['Lower_J'], row['Lower_configuration'], level_df)

        transition_data_df.iloc[i, list(transition_data_df.columns).index('Upper_level_location')] = temp_upper_idx + 1
        transition_data_df.iloc[i, list(transition_data_df.columns).index('Lower_level_location')] = temp_lower_idx + 1
    return transition_data_df


######################################################################

def transition_data_level_location(
        transition_data_df : pd.DataFrame, 
        level_df : pd.DataFrame
    ) -> pd.DataFrame:
    
    transition_data_df['Upper_level_location'] = 0
    transition_data_df['Upper_level_location'] = transition_data_df['Upper_level_location'].astype('int')
    transition_data_df['Lower_level_location'] = 0
    transition_data_df['Lower_level_location'] = transition_data_df['Lower_level_location'].astype('int')

    for i, (_, row) in enumerate(wrap_iterator(transition_data_df.iterrows(), desc="处理跃迁数据")):

        # print(row["Upper_loc"], row["Upper_J"], row["Upper_parity"])
        temp_upper_idx = level_df[(level_df["Pos"] == row["Upper_loc"]) & (level_df["J"] == row["Upper_J"]) & (level_df["Parity"] == row["Upper_parity"])].idx.values[0]
        # print(row["Lower_loc"], row["Lower_J"], row["Lower_parity"])
        temp_lower_idx = level_df[(level_df["Pos"] == row["Lower_loc"]) & (level_df["J"] == row["Lower_J"]) & (level_df["Parity"] == row["Lower_parity"])].idx.values[0]
        transition_data_df.iloc[i, list(transition_data_df.columns).index('Upper_level_location')] = temp_upper_idx + 1
        transition_data_df.iloc[i, list(transition_data_df.columns).index('Lower_level_location')] = temp_lower_idx + 1

    return transition_data_df

######################################################################

def transition_dT_cal(
        transition_rate_B: float | np.float64, 
        transition_rate_C: float | np.float64
    ) -> float | np.float64:
    """
    计算跃迁几率的相对差异

    Args:
        transition_rate_B: B方法计算的跃迁几率
        transition_rate_C: C方法计算的跃迁几率

    Returns:
        跃迁几率的相对差异值
    """
    if max(transition_rate_B, transition_rate_C) == 0:
        return 0.0

    transition_dT = abs(transition_rate_B - transition_rate_C) / max(transition_rate_B, transition_rate_C)

    return transition_dT