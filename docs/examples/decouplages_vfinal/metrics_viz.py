#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:24:26 2023

@author: maboum
"""
import os 
import pandas as pd 
import seaborn as sns
sns.set_theme(style="whitegrid")

df_vienna = pd.DataFrame({'Models': ['M0','M1','M2','M3','M4'],
    'Accuracy' :[0.952937, 0.939926, 0.939926, 0.865892, 0.856454],
             'IoU': [0.795017,0.743876673956172,0.743876673956172,0.544090416828365,0.495189352866045],
             'Recall': [0.895315485436258,0.85755290086897,0.85755290086897,0.726022666587264,0.760711207834555],
             'Precision': [0.874839505552787,0.844616148978985,0.844616148978985,0.670167970797779,0.583806203187594]})

df_austin = pd.DataFrame({'Models': ['M0','M1','M2','M3','M4'],
    'Accuracy' :[0.9704949259758,0.958875894546509, 0.95889276266098, 0.907210648059845, 0.904219627380371],
             'IoU': [0.804947020621836,0.737672535439643,0.737796491915137,0.407317055492385,0.426284380812644],
             'Recall': [0.89294515739784,0.862539922852578,0.862607050876682,0.791227148289257,0.771345840948107],
             'Precision': [0.889753054544368,0.835734928119307,0.835815885719231,0.463712014850942,0.507756279994717]})

# sns.barplot(data = df_vienna, x = df_vienna.Models,y = df_vienna.Accuracy)
# sns.barplot(data = df_vienna, x = [df_vienna.Accuracy, df_vienna.IoU, df_vienna.Recall, df_vienna.Precision],y = df_vienna.Models)
sns.catplot(data = df_vienna, kind = "bar", y = "Accuracy", x ="Models" , hue = "Models")
sns.catplot(data = df_vienna, kind = "bar", y = "IoU" , x = [i for i in range(5)],hue = "Models")