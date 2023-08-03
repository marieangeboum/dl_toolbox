#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:22:36 2023

@author: maboum
"""
import os 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import statistics

sequence_list =  ['D072_2019','D031_2019', 'D023_2020', 'D009_2019', 'D014_2020', 'D007_2020', 'D004_2021', 'D080_2021', 'D055_2018', 'D035_2020']
list_of_tuples = [(item, sequence_list.index(item)) for item in sequence_list]
all_metrics = []
paths = ['test_id_1571.xlsx','test_1571.xlsx']
# Create a new subplot for each strategy
fig, axes = plt.subplots(8, 1, figsize=(15, 8 * 8))
fig2, axes2 = plt.subplots(8, 1, figsize=(15, 8 * 8))
fig3, axes3 = plt.subplots(8, 1, figsize=(15, 8 * 8))
k = 0
for path in paths : 

    df = pd.ExcelFile(path)
    # metrics = 'flair1_replay'
    for metrics in df.sheet_names : 
        print(metrics)
        if metrics == 'Sheet1' : 
            continue
        metrics_df = pd.read_excel(io = path,sheet_name = metrics)
        # metrics_df['id'] = 'yes' if path == 'test_id_1571.xlsx' else 'no'
        metrics_df['strategy'] = metrics_df.model.str.split('_', expand = True)[1] if path == 'test_id_1571.xlsx' else metrics_df.model.str.split('_', expand = True)[0]
        # metrics_df['step'] = metrics_df['step'] +1
        # colums = ['Task ' + str(sequence_list.index(item) + 1) for item in sequence_list]
        # mean_metrics_df = metrics_df.groupby(['step', 'method','strategy' ]).mean()
        # mean_metrics_df.reset_index(inplace = True)
        # replay_model=mean_metrics_df[(mean_metrics_df['strategy'] == 'replay')& (mean_metrics_df['method']== 'model')]
        # baseline_model=mean_metrics_df[(mean_metrics_df['strategy'] == 'baseline')& (mean_metrics_df['method']== 'model')]
        
        metrics_df['BWT'] = pd.Series([])
        # metrics_df['AF'] = pd.Series([])
        metrics_df['BWT'] = metrics_df['BWT'].apply(pd.to_numeric)
        # metrics_df['AF'] = metrics_df['AF'].apply(pd.to_numeric)
        metrics_df_concat = pd.concat([metrics_df.iloc[:1]] * 4, ignore_index=True)
        metrics_df = pd.concat([metrics_df_concat,metrics_df], ignore_index=True)
        metrics_df['strategy'].iloc[[0,1]] =  'baseline'
        metrics_df['strategy'].iloc[[2,3]] =  'replay'
        metrics_df['method'].iloc[[0,2]] =  'decoder'
        for strategy in ('replay', 'baseline'): 
            for method in ('model','decoder'):    
                for j in range(1,10):
                    bwt = []
                    for i in range(j):                        
                        backward = metrics_df[(metrics_df['step'] == j) & (metrics_df['inference_domain'] == list_of_tuples[i][0]) & (metrics_df['method'] ==method) & (metrics_df['strategy'] == strategy)]['IoU'].iloc[0] - metrics_df[(metrics_df['step'] == i) & (metrics_df['inference_domain'] == list_of_tuples[i][0]) & (metrics_df['method'] == method) & (metrics_df['strategy'] ==strategy)]['IoU'].iloc[0] 
                        bwt.append(backward)                
                    bwt_mean = statistics.mean(bwt)
                    metrics_df.loc[(metrics_df['step'] == j) & (metrics_df['method'] == method)& (metrics_df['strategy'] == strategy), 'BWT'] = backward
        
        # for strategy in ('replay', 'baseline'): 
        #     for method in ('model','decoder'):    
        #         for j in range(1,10):                   
        #             max_iou = []
        #             for i in range(j):
        #                 max_iou.append(metrics_df[(metrics_df['step'].between(0, j))&(metrics_df['inference_domain'] == list_of_tuples[i][0]) & (metrics_df['method'] ==method) & (metrics_df['strategy'] == strategy)]['IoU'].max())
                        
        #                 forgetting = max_iou - 
                          
        results = metrics_df.groupby(['strategy', 'method', 'step']).mean(numeric_only=True)
        weights = metrics.split('_')[0]
        # Plot using Seaborn
        ax = axes[k]
        sns.set(style='ticks')
        sns.lineplot(x='step', y='IoU', hue='strategy', style='method', data=results, markers=True, ax = ax)
        iD = 'yes' if path == 'test_id_1571.xlsx' else 'no'
        ax.set_title(f'Initial model pretrained on {weights} with, {iD} is provided')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        results.reset_index(inplace = True)
        results2 = metrics_df.groupby(['strategy', 'method', 'step', 'inference_domain']).mean(numeric_only=True)
        results2.reset_index(inplace = True)
        results_model_baseline = results2[(results2['method']== 'model')& (results2['strategy']== 'baseline')]
        ax2 = axes2[k]
        sns.barplot(x='step', y='IoU', hue='inference_domain', data=results_model_baseline,ax = ax2)
        ax2.set_title(f'Initial model pretrained on {weights} with, {iD} is provided strategy : baseline')
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax3 = axes3[k]
        results_model_replay = results2[(results2['method']== 'model')& (results2['strategy']== 'replay')]
        sns.barplot(x='step', y='IoU', hue='inference_domain', data=results_model_replay, ax = ax3)
        
        ax3.set_title(f'Initial model pretrained on {weights} with, {iD} is provided, strategy : replay')
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        k+=1
        
        # results_domain = metrics_df.groupby(['strategy', 'method', 'step', 'inference_domain']).mean(numeric_only=True)
        # results_domain.reset_index(inplace = True)
        
        # results_domain2 = results_domain[results_domain['inference_domain']=='D072_2019']
        # Pivot the DataFrame to have step as columns and IoU as values
        metrics_pivot = results.pivot(index=['strategy', 'method'], columns='step', values=['IoU','BWT']).reset_index()
        
        # Rename the step columns to step_{step}
        metrics_pivot.columns = ['strategy', 'method'] + [f'step_{col}' for col in metrics_pivot.columns[2:]]
        metrics_pivot['pretraining'] = metrics
        metrics_pivot['id'] = iD
        # Save the pivoted DataFrame to a CSV file
        # metrics_pivot.to_csv('metrics_table.csv', index=False)
        all_metrics.append(metrics_pivot)


all_metrics_df = pd.concat(all_metrics)
# all_metrics_df.to_csv('metrics_table.csv', index=False)

k = 0
for j in range(2):
    print("étape" ,j)
    for i in range (4):        
        k+=1
        print("domaine ", k)
    print("fin de la boucle")


















