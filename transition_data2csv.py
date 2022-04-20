#! python3
# UTF-8
# Qin yi
# GRASP transition data transfer to csv file

"""
    Version 1.0.0(11/23/2021)
    first version for read transition data from *.ct and energylable.latex(produced by rtabtransition1)
    
    Version 1.1.0(12/24/2021)
    Function is full finished.
    The function only read data from *ct files and collect the data into one DataFrame. 
    The function strore the Electric transition in Coulomb gauge. And it can calculate wavelength and uncertainty automatically.
    The function also can search the index from the level DataFrame, and sort the Lower_index from smallest to biggest, so do the Upper_index.
    
    Version 1.2.0(04/09/2022)
    Add four columns in transition_data dataframe
    respectively they are Upper_J, Lower_J, f_l, f_v
    
    Version 1.2.1(04/20/2022)
    Add a parameter 'Braching_Fraction' in *data_process* function. The default value of 'Braching_Fraction' is 0.0001.
"""
import os
import numpy as np
import pandas as pd
import re


def data_load(file):
    # load*.ct file
    transition_file = open(f'{file}','r')
    transition_data = transition_file.readlines()
    transition_file.close()
    
    transition_file_df = pd.DataFrame(columns=['Lower_index','Lower_J', 'Lower_conf', 'Lowerlev', 'Upper_index', 'Upper_J', 'Upper_conf','Upperlev', 'Type', 'deltaE', 'Wavelength', 'A_l', 'gf_l', 'f_l', 'S_l', 'A_v', 'gf_v', 'f_v', 'S_v', 'BR', 'A_l_to_A_V', 'deltaS'])
    
    loc_tran_data = []
    """[summary]: This for loop find the location of the start transition data and store them in a list > loc_tran_data

    Returns:
        [type: list]: [description]: location and transition type in a list > loc_tran_data
    """
    for line in range(len(transition_data)):
        if 'Electric' in transition_data[line]:
            if line+5 >= len(transition_data):
                break 
            elif 'f' in transition_data[line+5]:
                print(line)
                temp_pole= re.findall(r'\(+ +[0-9]\)', transition_data[line]) 
                print('E',temp_pole)
                loc_tran_data.append(line)
                loc_tran_data.append('E'+f"{temp_pole[0][2]}")
        elif 'Magnetic' in transition_data[line]:
            if line+5 >= len(transition_data):
                break 
            elif 'f' in transition_data[line+5]:
                print(line)
                temp_pole= re.findall(r'\(+ +[0-9]\)', transition_data[line]) 
                print('M',temp_pole)
                loc_tran_data.append(line)
                loc_tran_data.append('M'+f"{temp_pole[0][2]}")
                
    """[summary] load and process the transition data into transtition_df DataFrame in sequence of list 'loc_tran_data'.

    Returns:
        [type: DataFrame]: [description]
    """
    for i in range(0,len(loc_tran_data),2):
        print(loc_tran_data[i])
        tran_line = loc_tran_data[i] + 5
        if 'E' in loc_tran_data[i+1]:
            tran_data_temp = pd.DataFrame(columns=['Upperlev', 'Lowerlev', 'Type', 'deltaE', 'A_l',  'gf_l', 'S_l', 'A_v', 'gf_v', 'S_v'])   # Electric *pole transition have two form: Babushkin gauge(length gauge) and Coulomb gauge(velocity gauge).
            no_tran = 0
            for tran_data_line in range(tran_line, len(transition_data)):
                # print(transition_data[tran_data_line])
                temp_data = transition_data[tran_data_line]
                if 'f' in temp_data and 'C' in temp_data:
                    # print(temp_data)
                    temp_data = temp_data.split()
                    tran_data_temp.loc[no_tran,['Upperlev', 'Lowerlev', 'Type', 'deltaE', 'A_v', 'gf_v', 'S_v']] = [" ".join(temp_data[0:4]), " ".join(temp_data[4:8]), f"{loc_tran_data[i+1]}", temp_data[8], temp_data[10], temp_data[11], temp_data[12] ] 

                elif 'B' in temp_data:
                    temp_data = temp_data.split()
                    tran_data_temp.loc[no_tran,['A_l', 'gf_l', 'S_l']] = [temp_data[1], temp_data[2], temp_data[3] ]
                    no_tran += 1
                elif '\n' == temp_data:
                    break

                tran_data_temp.replace(r'(.*)D(.*)', r'\1E\2',regex=True, inplace=True)
                tran_data_temp[['deltaE','A_v', 'A_l', 'gf_v', 'gf_l', 'S_v', 'S_l']] = tran_data_temp[['deltaE', 'A_v', 'A_l', 'gf_v', 'gf_l', 'S_v', 'S_l']].astype(np.float64, errors = 'raise')
                tran_data_temp = tran_data_temp[(tran_data_temp.S_v != 0) & (tran_data_temp.S_l != 0)]
                tran_data_temp['deltaS'] = tran_data_temp.apply( lambda x :  abs(x['S_v'] - x['S_l']) / max(x['S_v'], x['S_l']), axis=1)
                # exec( f"{loc_tran_data[i+1]}_temp = tran_data_temp")
            # tran_data_temp.rename(columns={'A_l' : 'A', 'gf_l' : 'gf', 'S_l' : 'S' }, inplace=True)
            transition_file_df = transition_file_df.append(pd.DataFrame(tran_data_temp, columns=['Upperlev', 'Lowerlev', 'Type' ,'deltaE','A_v', 'A_l', 'gf_v', 'gf_l', 'S_v', 'S_l', 'deltaS']), ignore_index=True)
            
        elif 'M' in loc_tran_data[i+1]:
            tran_data_temp = pd.DataFrame(columns=['Upperlev', 'Lowerlev', 'Type', 'deltaE', 'A_l',  'gf_l', 'S_l', 'A_v', 'gf_v', 'S_v'])   
            tran_data_temp.deltaS = np.nan
            no_tran = 0
            for tran_data_line in range(tran_line, len(transition_data)):
                temp_data = transition_data[tran_data_line]
                # print(temp_data)
                if 'M' in temp_data:
                    temp_data = temp_data.split()
                    tran_data_temp.loc[no_tran,['Upperlev', 'Lowerlev','Type', 'deltaE', 'A_l', 'gf_l', 'S_l']] = [" ".join(temp_data[0:4]), " ".join(temp_data[4:8]), f"{loc_tran_data[i+1]}", temp_data[8], temp_data[10], temp_data[11], temp_data[12]] 
                    tran_data_temp.loc[no_tran,['Upperlev', 'Lowerlev','Type', 'deltaE', 'A_v', 'gf_v', 'S_v']] = [" ".join(temp_data[0:4]), " ".join(temp_data[4:8]), f"{loc_tran_data[i+1]}", temp_data[8], temp_data[10], temp_data[11], temp_data[12]] 
                    no_tran += 1
                elif '\n' == temp_data:
                    break
                tran_data_temp.replace(r'(.*)D(.*)', r'\1E\2',regex=True, inplace=True)
                tran_data_temp[['deltaE', 'A_l',  'gf_l', 'S_l', 'A_v', 'gf_v', 'S_v']] = tran_data_temp[['deltaE', 'A_l',  'gf_l', 'S_l', 'A_v', 'gf_v', 'S_v']].astype(np.float64, errors = 'raise')
                # exec( f"{loc_tran_data[i+1]}_temp = tran_data_temp")
                tran_data_temp = tran_data_temp[(tran_data_temp.A_l != 0)&(tran_data_temp.gf_l != 0)]
            transition_file_df = transition_file_df.append(pd.DataFrame(tran_data_temp, columns=['Upperlev', 'Lowerlev', 'Type' ,'deltaE',  'A_l',  'gf_l', 'S_l', 'A_v', 'gf_v', 'S_v']), ignore_index=True)
            
    return transition_file_df

def data_process(transition_df, level, parameter, a_s, Branching_Fraction=0.0001):
    """[summary]: Find the index of lower and upper levels in level_DataFrame. Add the results into transition_df and sort the Lower_index from smallest to biggest, so do the Upper_index.

    Args:
        transition_df ([type: DataFrame]): [description] : columns=['Lower_index','Lower_J', 'Lower_conf', 'Lowerlev', 'Upper_index', 'Upper_J', 'Upper_conf','Upperlev', 'Type', 'deltaE', 'Wavelength', 'A', 'gf', 'S', 'deltaS']

    Returns:
        [type : DataFrame]: [description: return transition_df]
    """
    for line in range(len(transition_df)):
    #     print(transition_df.loc[line])
        Up_temp = transition_df.loc[line,'Upperlev'].split()
        Low_temp = transition_df.loc[line,'Lowerlev'].split()
        Up_index_temp = level[(level.Pos == Up_temp[1]) & (level.J == Up_temp[2]) & (level.Parity == Up_temp[3])].index.tolist()[0]
        Up_J_temp = Up_temp[2]
        Up_J_temp_value = eval(Up_J_temp)
        Low_index_temp = level[(level.Pos == Low_temp[1]) & (level.J == Low_temp[2]) & (level.Parity == Low_temp[3])].index.tolist()[0]
        Low_J_temp = Low_temp[2]
        Low_J_temp_value = eval(Low_J_temp)
        
        transition_df.loc[line, ['Upper_index', 'Upper_conf', 'Upper_J', 'Upper_J_value', 'Lower_index', 'Lower_conf', 'Lower_J', 'Lower_J_value']] =[level.loc[Up_index_temp, 'No'], level.loc[Up_index_temp, f'Configuration_{parameter}{a_s}raw'], Up_J_temp, Up_J_temp_value, level.loc[Low_index_temp, 'No'], level.loc[Low_index_temp, f'Configuration_{parameter}{a_s}raw'], Low_J_temp, Low_J_temp_value]
        transition_df.loc[line, 'Wavelength'] = 10 ** 8 / transition_df.loc[line, 'deltaE'] 

    transition_df['Lower_index'] = transition_df.Lower_index.astype(np.int8)
    transition_df['Upper_index'] = transition_df.Upper_index.astype(np.int8)
    transition_df.sort_values(by=['Lower_index', 'Upper_index'], ascending=True, inplace=True)
    transition_df.f_l = transition_df.gf_l / (2*transition_df.Lower_J_value + 1)
    transition_df.f_v = transition_df.gf_v / (2*transition_df.Lower_J_value + 1)
    transition_df.A_l_to_A_V = transition_df.A_l / transition_df.A_v
    level['lifetime_l'] = 0
    level['lifetime_l'] = level.lifetime_l.astype(np.float64)
    level['lifetime_v'] = 0
    level['lifetime_v'] = level.lifetime_v.astype(np.float64)
    # level.loc[0,'lifetime'] = 0
    for lno in range(2, len(level)+1, 1):
        temp_sum_A_l = transition_df.loc[(transition_df['Upper_index'] == lno), 'A_l'].sum()
        temp_sum_A_v = transition_df.loc[(transition_df['Upper_index'] == lno), 'A_v'].sum()
        print(temp_sum_A_l,temp_sum_A_v)
        if temp_sum_A_l != 0:
            transition_df.loc[(transition_df['Upper_index'] == lno), 'sum_A_l'] = temp_sum_A_l
            temp_lifetime_l = 1 / temp_sum_A_l
            level.loc[lno-1, 'lifetime_l'] = temp_lifetime_l
        else:
            continue
        if temp_sum_A_v != 0:
            transition_df.loc[(transition_df['Upper_index'] == lno), 'sum_A_v'] = temp_sum_A_v
            temp_lifetime_v = 1 / temp_sum_A_v
            level.loc[lno-1, 'lifetime_v'] = temp_lifetime_v
        else:
            continue
    transition_df['BR'] = np.float64(0)
    for trannum in range(len(transition_df)):
        transition_df.loc[trannum, 'BR'] = transition_df.loc[trannum, 'A_l']/transition_df.loc[trannum, 'sum_A_l']
        if transition_df.loc[trannum, 'BR'] <= Branching_Fraction:
            transition_df.drop(trannum, axis=0, inplace=True)
    
    return transition_df, level

def data_collect(dir):
    """
    [summary]: collect all transition data at once.
    """
    dir_temp= os.listdir(dir)
    trans_seque = []
    for i in dir_temp:
        if 'even' in i or 'odd' in i and '-' in i:
            trans_seque.append(i)
    
    for s in trans_seque:
        trans_sq_file = os.listdir(f'{dir}/{s}')
        for f in trans_sq_file:
            if f.split('.')[-1] == 'ct':
                print(f)
                file = f'{dir}/{s}/{f}'
                trans_data = data_load(file)
                if 'transition_df' not in locals().keys():
                    transition_df = trans_data.copy()
                else:
                    transition_df = transition_df.append(pd.DataFrame(trans_data), ignore_index=True)
    return transition_df