import numpy as np
from time import time
import csv
import Function.FileOperations as fo

"""
----------------------------------------------------------------------------------------------------
FFT Coreset
By NLHOANG 
# Input: X = dataset; m = number of coreset elements
# Output: C = coreset index (in X); data_info = index and distance to nearest center
----------------------------------------------------------------------------------------------------
"""
def CoresetConstruction(X, m):
    n = X.__len__()     # number of data's elements
    C = []              # coresets
    C_count = -1        # number of element in C
    # X_info: contain all info of each element in X
    # [] = index in X; [0] = closest center; [1] = distance to closest center
    # if [0] = -1 >> this point is a center
    X_info = []
    for i in range(n):
        X_info.insert(i, [-1, 1000000.0])
    X_info = np.asarray(X_info)

    #Initialization
    pos_min = np.argmin(X)
    new_center = int(pos_min / X.shape[1])
    C += [new_center]
    X_info[new_center][1] = -1
    C_count += 1

    while C_count < m:
        # Find nearest center of each point in data
        new_center = C[C_count]
        for i in range(n):
            if i not in C:
                new_dist = np.sqrt(np.sum(np.power(X[i] - X[new_center], 2), axis=0))
                if new_dist < X_info[i][1]:
                    X_info[i][0] = C_count
                    X_info[i][1] = new_dist

        # Find farthest point in each cluster
        MaxWD = -1          # Measurement to omit noise
        farthest = -1       # index of farthest point
        for i in range(C_count + 1):
            max_dist = -1
            max_id = -1
            count = 0
            for j in range(n):
                if X_info[j][0]==i:
                    count += 1
                    if max_dist < X_info[j][1]:
                        max_dist = X_info[j][1]
                        max_id = j
            temp = max_dist * count
            if temp > MaxWD:
                MaxWD = temp
                farthest = max_id

        C += [farthest]
        C_count += 1
        X_info[farthest][0] = -1
        X_info[farthest][1] = -1
    return np.asarray(C), np.asarray(X_info)

'''
----------------------------------------------------------------------------------------------------
FFT Coreset with Time Record
By NLHOANG 
# Input: X = dataset; m = number of coreset elements
# Output: C = coreset index (in X); data_info = index and distance to nearest center; time_record
----------------------------------------------------------------------------------------------------
'''
def CoresetConstruction_time(X, m):
    n = X.__len__()     # number of data's elements
    C = []              # coresets
    C_count = 0        # number of element in C
    # X_info: contain all info of each element in X
    # [] = index in X; [0] = closest center; [1] = distance to closest center
    # if [0] = -1 >> this point is a center
    X_info = []
    for i in range(n):
        X_info.insert(i, [-1, 1000000.0])
    X_info = np.asarray(X_info)
    timerec = []

    #Initialization
    pos_min = np.argmin(X)
    new_center = int(pos_min / X.shape[1])
    C += [new_center]
    X_info[new_center][1] = -1
    #C_count += 1
    timerec += [time()]
    while C_count < m-1:
        # Find nearest center of each point in data
        new_center = C[C_count]
        for i in range(n):
            if i not in C:
                new_dist = np.sqrt(np.sum(np.power(X[i] - X[new_center], 2), axis=0))
                if new_dist < X_info[i][1]:
                    X_info[i][0] = C_count
                    X_info[i][1] = new_dist

        # Find farthest point in each cluster
        MaxWD = -1          # Measurement to omit noise
        farthest = -1       # index of farthest point
        for i in range(C_count + 1):
            max_dist = -1
            max_id = -1
            count = 0
            for j in range(n):
                if X_info[j][0]==i:
                    count += 1
                    if max_dist < X_info[j][1]:
                        max_dist = X_info[j][1]
                        max_id = j
            temp = max_dist * count
            if temp > MaxWD:
                MaxWD = temp
                farthest = max_id

        C += [farthest]
        timerec += [time()]
        C_count += 1
        X_info[farthest][0] = -1
        X_info[farthest][1] = -1
    return np.asarray(C), np.asarray(X_info), np.asarray(timerec)


'''
----------------------------------------------------------------------------------------------------
Find X_info from Coreset Data
# Input: index of dataset, amount of selected-coreset
# Output: index of nearest point of data, amount of point in each selected-point
----------------------------------------------------------------------------------------------------
'''
def GetDataInfo(index, amount):
    cls = []
    count = []
    near_dist = []
    (X, dataname, datasize, K, datacode, csmaxsize) = fo.GetData(index)
    (coreset, timerec, cluster_id, cluster_dis) = fo.ReadProTraS(index)
    subsample = coreset[0:amount]
    C = X[subsample]
    for x in X:
        dist = np.sum(np.power(x - C, 2), axis=1)
        closest = np.argmin(dist)
        near_dist += [np.sqrt(np.min(dist))]
        cls += [closest]
    cls = np.asarray(cls)
    near_dist = np.asarray(near_dist)

    for i in range(amount):
        temp = (cls == i).sum()
        count += [temp]
    count = np.asarray(count)

    return cls, count, near_dist


