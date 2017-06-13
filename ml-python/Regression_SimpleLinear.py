#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:30:30 2017

@author: viche
"""


# Import the libararies
import numpy as up #math tools
import matplotlib.pyplot as plt #plat charts
import pandas as pd #import/manage datasets


# Adjustable parameters
P_data_source_csv = 'Salary_Data.csv'
P_test_size_ratio = 0.2
P_output_dimension = 1


# Data Preprocessing --- starts ---

# Import the dataset
dataset = pd.read_csv(P_data_source_csv)
features_input = dataset.shape[1] - P_output_dimension

X = dataset.iloc[:,:features_input].values
Y = dataset.iloc[:,dataset.shape[1]-1].values

# Split dataset to Training set and Validation set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=P_test_size_ratio)


# Fit Simple Linear Regression to Training set
from sklearn.linear_model import LinearRegression

regressor_sl = LinearRegression()
regressor_sl.fit(X_train,Y_train)


# Validation with Test set data
Y_predict = regressor_sl.predict(X_test)

# Visualising the training set data
plt.scatter(X_train,Y_train,color='black')
plt.plot(X_train,regressor_sl.predict(X_train),color='blue')

plt.scatter(X_test,Y_test,color='green')
plt.scatter(X_test,Y_predict,color='red')
plt.show



