#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:06:24 2017

@author: viche
"""

# Import the libararies
import numpy as np #math tools
import matplotlib.pyplot as plt #plat charts
import pandas as pd #import/manage datasets


# Adjustable parameters
P_data_source_csv = 'Market_Basket_Optimisation.csv'

"""---------------------------------
# Data Preprocessing
---------------------------------"""
# Import the dataset
dataset = pd.read_csv(P_data_source_csv, header=None)

nb_records = dataset.shape[0]
nb_goods = dataset.shape[1]

transactions=[]

for n in range(0, nb_records):
    transactions.append([str(dataset.values[n, i]) for i in range(0,nb_goods)])
    

"""---------------------------------
# Train Apriori model
---------------------------------"""

from apyori import apriori

# min_support: min appreances in total transactions in %
# min_confidence: rules met/correct in the transactions containing G1 in %
# min_lift: confidence / support
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

"""---------------------------------
# Visualize the results
---------------------------------"""

results = list(rules)