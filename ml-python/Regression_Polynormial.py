#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 13:39:53 2017

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

"""
# Split dataset to Training set and Validation set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=P_test_size_ratio)
"""

# Fit Simple Linear Regression to the data set
from sklearn.linear_model import LinearRegression
reg_lin = LinearRegression()
reg_lin.fit(X,Y)

# Fit Polynomial Linear Regression to the data set
from sklearn.preprocessing import PolynomialFeatures
reg_poly = PolynomialFeatures(degree=3)
X_poly = reg_poly.fit_transform(X)

reg_linpoly = LinearRegression()
reg_linpoly.fit(X_poly,Y)

# Visualize the Linear Regression results
plt.scatter(X,Y,color='black')
plt.plot(X,reg_lin.predict(X),color='blue')
plt.plot(X,reg_linpoly.predict(reg_poly.fit_transform(X)),color='red')

# Predict a new result

