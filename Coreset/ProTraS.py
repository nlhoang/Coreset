import numpy as np
from time import time
import csv
import Function.FileOperations as fo


'''
----------------------------------------------------------------------------------------------------
ProTraS Coreset
(from Ros and Guillaume in 'ProTraS: A probabilistic traversaling sampling algorithm' (2018))
# Input: X = dataset; epsilon 
# Output: C = coreset index (in X); data_info = index and distance to nearest center 
----------------------------------------------------------------------------------------------------
'''
def ProTraS_coreset(X, epsilon):
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
    cost = 100

    while cost > epsilon:
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
            cost += temp / n

        C += [farthest]
        C_count += 1
        X_info[farthest][0] = -1
        X_info[farthest][1] = -1

    return np.asarray(C), np.asarray(X_info)

'''
----------------------------------------------------------------------------------------------------
Improved ProTraS Coreset
By NLHOANG 
# Input: X = dataset; epsilon
# Output: C = coreset index (in X); data_info = index and distance to nearest center
----------------------------------------------------------------------------------------------------
'''
def ProTraS_improved(index, amount):
    (X, dataname, datasize, K, datacode, csmaxsize) = fo.GetData(index)
    (coreset, timerec, cluster_id, cluster_dis) = fo.ReadProTraS(index)
    (cls, count, near_dist) = GetDataInfo(index, amount)
    subsample = coreset[0:amount]
    C = X[subsample]
    newcs = []
    for i in range(amount):
        sample = []
        for j in range(datasize):
            if cls[j] == i:
                sample += [j]
        sample = np.asarray(sample)
        center = ProTraS_newcenter(X, sample)
        newcs += [center]
    return newcs

def ProTraS_newcenter(X, sample):
    min = -1
    totalmin = 1000000
    data = X[sample]
    for i in sample.__len__():
        item = data[i]
        total = np.sqrt(np.sum(np.power(item - data, 2), axis=1))
        if total < totalmin:
            totalmin = total
            min = i
    min = sample[min]
    return min

'''
----------------------------------------------------------------------------------------------------
Find Cost value from Coreset Data
# Input: index of dataset, amount of selected-coreset
# Output: cost value
----------------------------------------------------------------------------------------------------
'''
def ProTraS_costfunction(index, amount):
    (cls, count, near_dist, datasize) = GetDataInfo(index, amount)
    cost = 0
    for i in range(amount):
        cost += count[i]*near_dist[i]
    return cost / datasize

'''
----------------------------------------------------------------------------------------------------
Record all coreset results running from new ProTraS
# Input: X = dataset; m = number of coreset elements
# Output: C = coreset index (in X); data_info = index and distance to nearest center; time_record
----------------------------------------------------------------------------------------------------
'''
def ProTraS_Runtime(index):
    data, dataname, datasize, K, datacode, coresetsize = fo.GetData(index)
    t0 = time()
    coreset, data_info, time_record = CoresetConstruction_time(data, coresetsize)
    time_record = time_record - t0
    file = open(datacode + '.csv', 'w', newline='')
    datawriter = csv.writer(file, delimiter=',')
    datawriter.writerow([dataname, datasize, coresetsize])
    datawriter.writerow(coreset)
    datawriter.writerow(time_record)
    datawriter.writerow(data_info[:, 0])
    datawriter.writerow(data_info[:, 1])
    file.close()

'''
----------------------------------------------------------------------------------------------------
Initial Step for New ProTraS Coreset
# Input: X = dataset
# Output: first element for coreset
----------------------------------------------------------------------------------------------------
'''
def InitialStep(data):
    n = data.__len__()
    left = right = top = bot = data[0]
    m_left = m_right = m_top = m_bot = 1000000

    for i in range(n):
        d_left = data[i][0] - left[0]
        d_right = data[i][0] - right[0]
        d_top = data[i][1] - top[1]
        d_bot = data[i][1] - bot[1]

        if d_left < 0: left = data[i]
        if d_right > 0: right = data[i]
        if d_top > 0: top = data[i]
        if d_bot < 0: bot = data[i]

        if (d_left != 0) & (np.abs(d_left) < m_left): m_left = np.abs(d_left)
        if (d_right != 0) & (np.abs(d_right) < m_right): m_right = np.abs(d_right)
        if (d_top != 0) & (np.abs(d_top) < m_top): m_top = np.abs(d_top)
        if (d_bot != 0) & (np.abs(d_bot) < m_bot): m_bot = np.abs(d_bot)

    index = np.argmin([m_left, m_right, m_top, m_bot])
    if index == 0: result = left
    if index == 1: result = right
    if index == 2: result = top
    if index == 3: result = bot
    return result
