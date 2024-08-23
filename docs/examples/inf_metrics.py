#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:44:28 2023
@author: maboum
"""
import pandas as pd 
import os 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image
matplotlib.use('TkAgg')
path = ['inf_s2.xlsx']

all_df = []
mean_all_df = []
std_all_df = []
df_all_mean = []
df_all_std = []
for p in path : 
    df_init = pd.ExcelFile(p)
    for sheet in df_init.sheet_names : 
        if 'bilan' in sheet : 
            continue 
        else : 
            df = pd.read_excel(p, sheet_name=sheet)
            df['model']  = df['model'].str.split(pat = '_seed_', n= 1, expand  = True )[0]
            df['num_domains'] = sheet.split('_')[1]
            print(sheet.split('_')[1])
            df['expe_type'] = sheet.split('_')[0]
            # all_df.append(df)
        
            # all_df = pd.concat(all_df)
            mean_df = df.groupby(by = ['model', 'inference_domain', 'num_domains', 'expe', 'expe_type'] ).mean()
            std_df = df.groupby(by = ['model', 'inference_domain', 'num_domains', 'expe', 'expe_type'] ).std(ddof =1)
            std_df.reset_index(inplace = True)
            mean_df.reset_index(inplace = True)
            mean_all_df.append(mean_df)
            std_all_df.append(std_df)
            
            
            df_all_mean.append(mean_df)
            df_all_std.append(std_df)
    
    
mean_df = pd.concat(df_all_mean)
std_df = pd.concat(df_all_std)    

group_mean = mean_df.groupby(by = ['inference_domain', 'expe_type', 'num_domains'], group_keys=False).apply(lambda x: x)
group_mean.reset_index(inplace = True)

group_mean_nd = group_mean.groupby( by = ['expe_type','num_domains', 'expe'], group_keys = False).mean()
group_mean_nd.reset_index(inplace = True)
group_mean_nd['legend'] = group_mean_nd.expe_type + group_mean_nd.expe
barplot  = sns.barplot(data = group_mean_nd, x= 'num_domains', y= 'IoU', hue = 'legend')
plt.legend(loc='lower left')
fig = barplot.get_figure()
fig.savefig("out2.png")