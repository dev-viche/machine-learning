#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:25:57 2017

@author: viche
"""


# Import the libararies

import numpy as up #math tools
import matplotlib.pyplot as plt #plat charts
import pandas as pd #import/manage datasets
import random

# Adjustable parameters

P_data_source_csv = 'Ads_CTR_Optimisation.csv'
P_test_size_ratio = 0.2

# Import the dataset
dataset = pd.read_csv(P_data_source_csv)

rounds = dataset.shape[0]
ads = dataset.shape[1]

ads_selected = []
total_reward = 0

nb_reward_y = [0] * ads
nb_reward_n = [0] * ads


for n in range (0, rounds):
    
    ad_selected = -1
    max_random = 0
    
    for i in range (0, ads):
        random_beta = random.betavariate(nb_reward_y[i]+1,nb_reward_n[i]+1)
        
        if random_beta > max_random:
            max_random = random_beta
            ad_selected = i
    
    ads_selected.append(ad_selected)
    
    if (ad_selected>-1):     
        reward = dataset.values[n, ad_selected]
        if reward == 1:
            nb_reward_y[ad_selected] = nb_reward_y[ad_selected]+1
        else:
            nb_reward_n[ad_selected] = nb_reward_n[ad_selected]+1
        
        total_reward = total_reward + reward

plt.hist(ads_selected)