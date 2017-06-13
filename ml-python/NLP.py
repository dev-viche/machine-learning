#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:38:44 2017

@author: viche
"""

# Import the libararies

import numpy as up #math tools
import matplotlib.pyplot as plt #plat charts
import pandas as pd #import/manage datasets
import nltk #nlp libraries

# Adjustable parameters
P_data_source_csv = 'Restaurant_Reviews.tsv'
P_test_size_ratio = 0.2

# Import the dataset
dataset = pd.read_csv(P_data_source_csv, delimiter='\t', quoting=3)


"""-------------------------------------------------
Section 1: Clean up the text - text preprocessing
-------------------------------------------------"""
# Import necessary libraries and packages
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []

for i in range (0,dataset.shape[0]):
    # 1. remove non-alphabetic characters
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    
    # 2. conver to lower case
    review = review.lower()
    
    # 3. remove non-meaningful words
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    
    corpus.append(review)
    
"""-------------------------------------------------
Section 2: Create the Bag of Words model
-------------------------------------------------"""
# Import necessary libraries and packages
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

"""-------------------------------------------------
Section 3: Create and Train the Classification model
-------------------------------------------------"""

# Split dataset to Training set and Validation set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=P_test_size_ratio)

# Fit Logistic Linear Regression to the data set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predict the Test set result
y_predict = classifier.predict(X_test)

# Make the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)





