#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 17:52:45 2017

@author: viche
"""

# Import the libararies
import numpy as np #math tools
import matplotlib.pyplot as plt #plat charts
import pandas as pd #import/manage datasets


# Adjustable parameters
P_data_source_csv = 'Mall_customers.csv'
P_test_size_ratio = 0.2

"""--------------------------------
Section 1: Data Prepocessing
--------------------------------"""

# Import the dataset
dataset = pd.read_csv(P_data_source_csv)
X = dataset.iloc[:,[3,4]].values


"""--------------------------------
Section 2: Fit the K-Mean Clustering
--------------------------------"""

from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.show()


"""--------------------------------
Section 3: Fit the K-Mean Clustering
--------------------------------"""

kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10)
y_km = kmeans.fit_predict(X)

"""--------------------------------
Section 4: Visualize the results
--------------------------------"""
plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_km == 3, 0], X[y_km == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_km == 4, 0], X[y_km == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='yellow',label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
