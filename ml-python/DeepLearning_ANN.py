#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 11:24:56 2017

@author: viche
"""

# Artificial Neural Network

"""
# Install the libararies: Theano, Tensorflow, Keras

# Installing Theano
$ pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow with Anaconda
$ conda create -n tensorflow
$ source activate tensorflow
$ export TF_PYTHON_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.1.0-py3-none-any.whl
$ pip install --ignore-installed --upgrade $TF_PYTHON_URL

# Installing Keras
$ pip install --upgrade keras
"""

# Import the libararies
import numpy as np #math tools
import matplotlib.pyplot as plt #plat charts
import pandas as pd #import/manage datasets


# Adjustable parameters
P_data_source_csv = 'Churn_Modelling.csv'
P_test_size_ratio = 0.2
P_output_dimension = 1

# ---------------------------------
# Data Preprocessing --- starts ---

# Import the dataset
dataset = pd.read_csv(P_data_source_csv)

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,dataset.shape[1]-1].values

# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encoder 1 - Encode Country
labelencoder_Ctry = LabelEncoder()
X[:,1]=labelencoder_Ctry.fit_transform(X[:,1])

# Encoder 2 - Encode Gender
labelencoder_Gender = LabelEncoder()
X[:,2]=labelencoder_Gender.fit_transform(X[:,2])

# Create Dummy Variables and Remove Trap
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

# Split dataset to Training set and Validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=P_test_size_ratio)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Data Preprocessing --- ended ---
# --------------------------------

# ---------------------------------
# Make the ANN model --- starts ---

# Import the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Construct the model
# Initialize the ANN
classifier = Sequential()

# Add the input layer and the first hidden layer
classifier.add(Dense(6,input_dim=11,kernel_initializer='uniform',activation='relu'))

# Add the second hidden layer
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))

# Add the output layer
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

# Compile the model
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Fit the model
classifier.fit(X_train,y_train, epochs=100, batch_size=10)

# Make the ANN model --- ended ---
# --------------------------------

# ---------------------------------
# Evaluate the ANN model --- starts ---

# Predict the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

# Compare the predicted v/s actual Test set results
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Evaluate the ANN model --- ended ---
# --------------------------------





