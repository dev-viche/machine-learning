#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 08:12:02 2017

@author: viche
"""

# Data Preprocessing

# Import the libararies

import numpy as up #math tools
import matplotlib.pyplot as plt #plat charts
import pandas as pd #import/manage datasets

# Adjustable parameters

P_data_source_csv = 'Data.csv'
P_test_size_ratio = 0.2
P_input_dimension = 0
P_output_dimension = 0

# Import the dataset
dataset = pd.read_csv(P_data_source_csv)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# Missing data - mean replacement
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y=labelencoder_y.fit_transform(y)

# Split dataset to Training set and Validation set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=P_test_size_ratio)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train =  sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Data Preprocessing Template


