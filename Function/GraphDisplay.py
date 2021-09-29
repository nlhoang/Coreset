import matplotlib.pyplot as plt
import numpy as np
import Function.FileOperations as fo

def Display1data(X, title=''):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    color = (0,0,0)
    plt.scatter(X[:, 0], X[:, 1], alpha=.3, color=color)
    plt.axis('equal')
    plt.plot()
    plt.show()

def Display2data(X, Y, title=''):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.scatter(X[:, 0], X[:, 1], alpha=0.3, color='green')
    plt.scatter(Y[:, 0], Y[:, 1], alpha=0.3, color='red')
    plt.axis('equal')
    plt.plot()
    plt.show()

def Display1data_index(index, title=''):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    (X, dataname, datasize, K, datacode, csmaxsize) = fo.GetData(index)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.3, color='green')
    plt.axis('equal')
    plt.plot()
    plt.show()

#Display X=data1, Y=data2
def display(X, Y, title):
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.plot(X[:, 0], X[:, 1], '.', color='green')
    plt.plot(Y[:, 0], Y[:, 1], '.', color='red')
    plt.axis('equal')
    plt.plot()
    plt.show()

#Display for kmeans with K=cluster_amount and label array
def kmeans_display(X, K, label):
    for k in range(K):
        data = X[label == k]
        plt.plot(data[:, 0], data[:, 1], '.')
    plt.axis('equal')
    plt.plot()
    plt.show()

