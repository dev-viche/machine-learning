#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:18:51 2017

@author: viche
"""

# Import the libararies

import numpy as up #math tools
import matplotlib.pyplot as plt #plat charts
import pandas as pd #import/manage datasets
import math

# Adjustable parameters

P_data_source_csv = 'Ads_CTR_Optimisation.csv'
P_test_size_ratio = 0.2

# Import the dataset
dataset = pd.read_csv(P_data_source_csv)

nb_rounds = dataset.shape[0]
d = 10

nb_selections = [0] * d
sum_rewards = [0] * d

ads_selected = []
total_reward = 0

for n in range (0, nb_rounds):
    
    ad_selected = -1
    max_upper_bound = 0
    
    for i in range (0, d):
        if(nb_selections[i]>0):
            ave_rewards = sum_rewards[i] / nb_selections[i]
            delta_i = math.sqrt (3/2 * math.log(n+1)/nb_selections[i])
            upper_bound = ave_rewards + delta_i
        else:
            upper_bound = 1e400
        
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad_selected = i
    
    ads_selected.append(ad_selected)
    
    if (ad_selected>-1):     
        nb_selections[ad_selected] = nb_selections[ad_selected] + 1
        reward = dataset.values[n, ad_selected]
        sum_rewards[ad_selected] = sum_rewards[ad_selected] + reward
        
        total_reward = total_reward + reward
    

plt.hist(ads_selected)
        
    