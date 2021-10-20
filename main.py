# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import sklearn

def loadDataSet():
    a = []
    f = open("B.txt",'r')
    lines = f.readlines()   #make lines a list, has index inside already
    f.close()
    for line in lines:
        b = ['','']
        b[0],b[1] = line.split()
        a.append(b)
    dataSet = np.array(a,dtype=float)
    return dataSet

X = loadDataSet()

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='single')
cluster.fit_predict(X)
print(cluster.labels_)


labels = range(1, 3)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-4, 4),
        textcoords='offset points', ha='right', va='bottom')
plt.show()
