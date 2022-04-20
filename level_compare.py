#! python3
# UTF-8
# Qin yi
# GRASP levels comparision
#################################################
"""
    Version 1.0.0(11/1/2021)
    first version for read levels from 
    rlevels even.m > {atom}{as}
    this version didn't consider different parities
    
    Version 1.0.1(11/2/2021)
    this version could read levels from
    rlevels {atom}{even}{as}.m {atom}{odd}{as}.m> {atom}{as}
    and compare the difference between two active space(consider parities)
    
    Version 1.0.2(12/1/2021)
    Add "Configuration" and "LSj" columns in DataFrame. 
    Save DataFrame as csv files
    
    Version 1.0.3(12/2/2021)
    Change level_storage function to level_mcdhf_storage.
    Add level_add function to add CI calculation results.
    level_read function could detect the Configurations in energy files, and add "Configuration
    
    Version 1.0.4(12/12/2021)
    Add file.close()
    
    Version 1.0.5(12/17/2021)
    Change level configuration storage strategy.
    Due to the DHF results and the active space optimization results may have difference, the as0's configuration added.
    Add configuration skip VALUE to omit some electron orbital which is not needed.
    
    Version 1.0.6(12/18/2021)
    Add level's composition of ASF function.
    
    Version 1.0.7(1/10/2022)
    Add "Configuration"   subshells LS-coupling state, for better identifying the atomic state.
    
    Version 1.1.1(04/08/2022)
    TODO Object-oriented programming of the functions involved
    A function is needed to read configuration state raw and change it in LaTeX form. 
    
"""

import numpy as np
import pandas as pd
import re

def configuration_format(this_conf):
    '''
    Format the configuration state of energy levels data file to a LaTeX form string and a row list.
    
    this_conf: a string of configuration state
    '''
    conf_temp = this_conf.split('.')
    format_conf = ''
    for subshell in conf_temp:
        print(subshell)
        if '(' and ')' in subshell:
            ele_num = re.findall(r"[(](.*?)[)]", subshell)
            temp_subshell_LS_transform = ''
            if re.findall(r"[)]([0-9][A-Z][0-9]?)[_]", subshell):
                temp_subshell_LS = re.findall(
                    r"[)]([0-9][A-Z][0-9]?)[_]", subshell)
                temp_subshell_LS_transform = f"\\,(^{temp_subshell_LS[0][0]}_{temp_subshell_LS[0][-1]}{temp_subshell_LS[0][1]})"
            format_conf = format_conf + f'{subshell[0:2]}^{{{ele_num[0]}}}{temp_subshell_LS_transform}\\,' 
            
        else:
            if subshell != conf_temp[-1]:
                temp_subshell_LS_transform = ''
                if '_' in subshell:
                    temp_subshell_LS = re.findall(r"[_]([0-9][A-Z]?)", subshell)
                    temp_subshell_LS_transform = f"\\,^{temp_subshell_LS[0][0]}{temp_subshell_LS[0][1]}"
                    format_conf = format_conf + f'{subshell[0:2]}{temp_subshell_LS_transform}\\,'
            elif subshell == conf_temp[-1]:
                format_conf = format_conf + f'{subshell[0:2]}\\,'
                
    return format_conf, conf_temp

# load level data from {atom}{as}

def level_read(atom, dir, parameters, a_s, skipconf=0):
    level_as = f'{dir}/{atom}{parameters}{a_s}'
    print(level_as)
    this_as = parameters + str(a_s)
    level_file = open(level_as, 'r')
    level_data = level_file.readlines()
    level_file.close()
    
    # This for skip line to read levels 
    # ' No Pos  J ' is the target in the begining of the level data
    for i in range(len(level_data)):
        if ' No Pos  J ' in level_data[i]:
            skip_line = i + 3
            break
    
    # define 'level' to store the level data
    level_read_df = pd.DataFrame(columns=['No', 'Pos', 'J', 'Parity', f'{this_as}',f'Configuration_{this_as}raw'])
    
    # store level data to 'level' array
    for line in range(0,len(level_data)-skip_line-1):
        # print(level_data[line+skip_line])
        level_source = level_data[line+skip_line].split()
        # print(level_source)
        if len(level_source) != 8:
            if line == 0 :
                level_read_df.loc[line, ['No', 'Pos', 'J', 'Parity', f'{this_as}']] = [level_source[0], level_source[1], level_source[2], level_source[3], float(0)]
            else:
                level_read_df.loc[line, ['No', 'Pos', 'J', 'Parity', f'{this_as}']] =[level_source[0], level_source[1], level_source[2], level_source[3], level_source[5]]
        else:
            level_read_df.loc[line, ['No', 'Pos', 'J', 'Parity', f'{this_as}', f'Configuration_{this_as}raw']] =[level_source[0], level_source[1], level_source[2], level_source[3], level_source[5], level_source[7]]

    # change Levels columns type to float
    level_read_df[f'{this_as}'] = level_read_df[f'{this_as}'].astype(np.float64)

    # change Configuration into a LaTeX equation form
    if not level_read_df[f'Configuration_{this_as}raw'].isnull().any():
        # print(level_read_df['Configuration_raw'])
        for n in range(len(level_read_df[f'Configuration_{this_as}raw'])):
            format_conf, conf_temp = configuration_format(level_read_df.loc[n, f'Configuration_{this_as}raw'][skipconf:])
            ls = conf_temp[-1][-2:]

            level_read_df.loc[n, f'Configuration_{this_as}']= f'${format_conf }$'
            total_angular_j = level_read_df.loc[n, 'J']
            if level_read_df.loc[n, 'Parity'] == '-':
                level_read_df.loc[n, f'LSj_{this_as}'] = f'$\\,^{ls[0]}{ls[1]}_{{{total_angular_j}}}^{{o}}$'
            elif level_read_df.loc[n, 'Parity'] == '+':
                level_read_df.loc[n, f'LSj_{this_as}'] = f'$\\,^{ls[0]}{ls[1]}_{{{total_angular_j}}}$'
    return level_read_df

# add extra energy levels data to level DataFrame
def level_add(level, atom, dir, parameters, a_s=np.nan, skipconf=0):
    this_as = parameters + str(a_s)
    level_temp = level_read(atom, dir, parameters, a_s, skipconf)
    level[f'{parameters}{a_s}'] = np.nan
    for i in range(0, len(level)):
        for j in range(0, len(level_temp)):
            if level.loc[i,'Pos'] == level_temp.loc[j,'Pos'] and level.loc[i,'J'] == level_temp.loc[j,'J'] and level.loc[i,'Parity'] == level_temp.loc[j,'Parity']:
                level.loc[i, f'{this_as}'] = level_temp.loc[j, f'{this_as}']
                if not level_temp[f'Configuration_{this_as}raw'].isnull().any():
                    level.loc[i, f'Configuration_{this_as}'] = level_temp.loc[j, f'Configuration_{this_as}']
                    level.loc[i, f'LSj_{this_as}'] = level_temp.loc[j, f'LSj_{this_as}']
                    level.loc[i, f'Configuration_{this_as}raw'] = level_temp.loc[j, f'Configuration_{this_as}raw']
    return level

def level_mcdhf_storage(level, atom, dir, parameters, max_as=np.nan, min_as=0, skipconf=0):
    for a in range(min_as, max_as+1):
        level = level_add(level, atom, dir, parameters, a, skipconf)
    level['No'] = level.No.astype(np.int32)
    level['Comp_of_asf'] = ''
    return level



# difference between two as levels
def level_diff(level, parameters, max_as):
    for a in range(max_as):
        if a == 0:
            level[f'dE{a+1}'] = level[f'{parameters}{a+1}'] - level[f'{parameters}{a}']
            level[f'dE{a+1}per'] = (level[f'{parameters}{a+1}'] - level[f'{parameters}{a}']) / level[f'{parameters}{a}']
        else:
            level[f'dE{a+1}'] = level[f'{parameters}{a+1}'] - level[f'{parameters}{a}']
            level[f'dE{a+1}per'] = (level[f'{parameters}{a+1}'] - level[f'{parameters}{a}']) / level[f'{parameters}{a}']
    return level


# Add level's composition of ASF
def level_comp_asf(level, dir, parity, parameters, max_as, min_comp=0.03, skipconf=0):

    lsjlblfile = open(f'{dir}/{parity}{parameters}{max_as}.lsj.lbl', 'r')
    lsjlbl = lsjlblfile.readlines()
    lsjlblfile.close()
    # lvloclbl =level located in lbl file
    lvloclbl = []

    for line in range(0,len(lsjlbl)):
        # print(lsjlbl[line])
        if re.search('\d+.\d+%', lsjlbl[line]) :
            lvloclbl.append(line)
            # print(line)

    compasf = []
    lvloc = []
    for i in range(len(lvloclbl)):
        if lvloclbl[i] != lvloclbl[-1]:
            lvloc.append(lsjlbl[lvloclbl[i]].split())
            comp_temp=[]
            for line in lsjlbl[lvloclbl[i]+1: lvloclbl[i+1]]:
                if line != ' \n':
                    comp_temp.append(line.split())
        else:
            lvloc.append(lsjlbl[lvloclbl[i]].split())
            comp_temp=[]
            for line in lsjlbl[lvloclbl[i]+1: ]:
                # print(line)
                if line != ' \n':
                    comp_temp.append(line.split())
        compasf.append(comp_temp)

    for l in range(len(lvloc)):
        lvloc_temp = lvloc[l]
        lv_index = level.loc[(level['Pos'] == lvloc_temp[0]) & (level['J'] == lvloc_temp[1]) & (level['Parity']== lvloc_temp[2])].index[0]
        
        # Store the Comp of ASF in a temp DataFrame
        comp_temp = pd.DataFrame(columns=['c', 'w', 'comp_conf'])
        for line in range(len(compasf[l])):
            comp_temp.loc[line, ['c', 'w', 'comp_conf']] = compasf[l][line]

        comp_temp['w'] = comp_temp.w.astype(np.float64).round(2)
        comp_temp = comp_temp[comp_temp['w'] >= min_comp]
        comp_temp['comp_conf'] = comp_temp['comp_conf'].str[skipconf:]
        print(comp_temp)
        # temp_tlevel = level.loc[lv_index, 'Configuration_as0raw']
        
        for n in range(len(comp_temp['comp_conf'])):
            temp_comp_of_asf  = ''
            format_comp_conf_temp, comp_conf_temp = configuration_format(comp_temp.loc[n, 'comp_conf'])
            
            temp_comp_ls = comp_conf_temp[-1][-2:]
            comp_LS = f"^{temp_comp_ls[0]}{temp_comp_ls[1]}"
            temp_comp_of_asf = temp_comp_of_asf+ '$' + str(comp_temp.loc[n,'w']) + '\\;' + format_comp_conf_temp + '\\,' + comp_LS + '$ + '

            level.loc[lv_index, 'Comp_of_asf'] = level.loc[lv_index, 'Comp_of_asf'] + temp_comp_of_asf
        level.loc[lv_index, 'Comp_of_asf'] = level.loc[lv_index, 'Comp_of_asf'][:-2]

    return level   
