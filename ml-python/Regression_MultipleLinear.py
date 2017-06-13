#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:30:33 2017

@author: viche
"""


# Import the libararies
import numpy as np #math tools
import matplotlib.pyplot as plt #plat charts
import pandas as pd #import/manage datasets


# Adjustable parameters
P_data_source_csv = '50_Startups.csv'
P_test_size_ratio = 0.2
P_output_dimension = 1


# Data Preprocessing --- starts ---

# Import the dataset
dataset = pd.read_csv(P_data_source_csv)
features_input = dataset.shape[1] - P_output_dimension

X = dataset.iloc[:,:features_input].values
Y = dataset.iloc[:,dataset.shape[1]-1].values

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoid the Dummy Variable Trap
X = X[:,1:]


# Split dataset to Training set and Validation set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=P_test_size_ratio)

# Fit Multiple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor_ml = LinearRegression()
regressor_ml.fit(X_train,Y_train)

# Validate with Test set data
Y_predict = regressor_ml.predict(X_test)

# Optimize using Backward Elimination
import statsmodels.formula.api as sm

X = np.append(arr=np.ones((dataset.shape[0],1)).astype(int), values=X,axis=1)
X_opt = X[:,:]

while True:
    regressor_OLS = sm.OLS(endog=Y, exog=X_opt).fit()
    pvalues = regressor_OLS.pvalues
    
    max_pvalue = -1.0
    i=0
    
    for p in pvalues:
        if max_pvalue < p:
            index=i
            max_pvalue = p
        i=i+1
    
    if max_pvalue < 0.05:
        break
    
    X_opt=np.delete(X_opt, index ,axis=1)









