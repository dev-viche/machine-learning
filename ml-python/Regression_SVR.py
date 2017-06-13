#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:34:45 2017

@author: viche
"""


# Import the libararies
import numpy as np #math tools
import matplotlib.pyplot as plt #plat charts
import pandas as pd #import/manage datasets


# Adjustable parameters
P_data_source_csv = 'Position_Salaries.csv'
P_test_size_ratio = 0.2
P_output_dimension = 1


# Data Preprocessing --- starts ---

# Import the dataset
dataset = pd.read_csv(P_data_source_csv)
features_input = dataset.shape[1] - P_output_dimension

X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,dataset.shape[1]-1].values

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_Y = StandardScaler()
Y = sc_Y.fit_transform(Y)

"""
# Split dataset to Training set and Validation set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=P_test_size_ratio)
"""

# Fit Simple Linear Regression to the data set
from sklearn.svm import SVR
regressor_svr = SVR(kernel='rbf')
regressor_svr.fit(X,Y)

# Visualize the Linear Regression results
plt.scatter(sc_X.inverse_transform(X),sc_Y.inverse_transform(Y),color='black')
plt.plot(sc_X.inverse_transform(X),sc_Y.inverse_transform(regressor_svr.predict(X)),color='blue')

plt.show()

# Predict a new result