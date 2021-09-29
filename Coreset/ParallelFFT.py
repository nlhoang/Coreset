import numpy as np
import multiprocessing as mp

from time import time
import csv

import Coreset.ProTraS as protras
import Function.FileOperations as fo

count1 = []
count2 = []
count3 = []
count4 = []

#--------------------------------------------------------------------------------------

#--------------------------------------------------------------------------------------
def divideSpace2D(data, mean):
    global count1, count2, count3, count4
    x0 = mean[0]
    y0 = mean[1]
    for i in range(data.__len__()):
        point = data[i]
        if (point[0] < x0) & (point[1] >= y0):
            count1.append(i)
        if (point[0] >= x0) & (point[1] >= y0):
            count2.append(i)
        if (point[0] >= x0) & (point[1] < y0):
            count3.append(i)
        if (point[0] < x0) & (point[1] < y0):
            count4.append(i)

    count1 = np.asarray(count1)
    count2 = np.asarray(count2)
    count3 = np.asarray(count3)
    count4 = np.asarray(count4)
    return (count1, count2, count3, count4)

#--------------------------------------------------------------------------------------
def divideSpace3D(data, mean):
    global count1, count2, count3, count4
    x0 = mean[0]
    y0 = mean[1]
    z0 = mean[2]
    for i in range(data.__len__()):
        point = data[i]
        if (point[0] < x0) & (point[1] >= y0):
            count1.append(i)
        if (point[0] >= x0) & (point[1] >= y0):
            count2.append(i)
        if (point[0] >= x0) & (point[1] < y0):
            count3.append(i)
        if (point[0] < x0) & (point[1] < y0):
            count4.append(i)

    count1 = np.asarray(count1)
    count2 = np.asarray(count2)
    count3 = np.asarray(count3)
    count4 = np.asarray(count4)
    return (count1, count2, count3, count4)

#--------------------------------------------------------------------------------------
def divideQuantity2D(n, m, n1, n2, n3):
    m1 = int(m * n1 / n)
    m2 = int(m * n2 / n)
    m3 = int(m * n3 / n)
    m4 = m - m1 - m2 - m3
    return (m1, m2, m3, m4)

#--------------------------------------------------------------------------------------
def CoresetConstruction(arguments):
    data, index, m = arguments
    n = index.__len__()     # number of data's elements
    X = data[index]
    C = []              # coresets
    C_count = 0        # number of element in C
    # X_info: contain all info of each element in X
    # [] = index in X; [0] = closest center; [1] = distance to closest center
    # if [0] = -1 >> this point is a center

    X_info = []
    for i in range(n):
        X_info.insert(i, [index[i], -1, 1000000.0])
    X_info = np.asarray(X_info)

    #Initialization
    pos_min = int(np.argmin(X) / X.shape[1])
    new_center = index[pos_min]
    C += [new_center]
    X_info[pos_min][2] = -1

    while C_count < m-1:
        # Find nearest center of each point in data
        new_center = C[C_count]
        for i in range(n):
            if X_info[i][0] not in C:
                new_dist = np.sqrt(np.sum(np.power(X[i] - data[new_center], 2), axis=0))
                if new_dist < X_info[i][2]:
                    X_info[i][1] = new_center
                    X_info[i][2] = new_dist

        # Find farthest point in each cluster
        MaxWD = -1          # Measurement to omit noise
        farthest = -1       # index of farthest point
        for i in range(C_count + 1):
            max_dist = -1
            max_id = -1
            count = 0
            center = C[i]
            for j in range(n):
                if X_info[j][1]==center:
                    count += 1
                    if max_dist < X_info[j][2]:
                        max_dist = X_info[j][2]
                        max_id = j
            temp = max_dist * count
            if temp > MaxWD:
                MaxWD = temp
                farthest = max_id

        C += [int(X_info[farthest][0])]
        C_count += 1
        X_info[farthest][1] = -1
        X_info[farthest][2] = -1
    return C

#--------------------------------------------------------------------------------------
'''
if __name__ == '__main__':
    data, dataname, datasize, K, datacode, csmaxsize = fo.GetData(7)

    temp = time()
    mean = np.mean(data, axis=0)
    count1, count2, count3, count4 = divideSpace2D(data, mean)
    m1, m2, m3, m4 = divideQuantity(datasize, 100, count1.__len__(), count2.__len__(), count3.__len__(),
                                    count4.__len__())
    count_full = [count1, count2, count3, count4]
    m_full = [m1, m2, m3, m4]
    full = [(data, count1, m1), (data, count2, m2), (data, count3, m3), (data, count4, m4)]
    pool = mp.Pool(4)
    results = pool.map(CoresetConstruction, [X for X in full])
    coreset_index = []
    for res in results:
        coreset_index += res
    coreset = data[coreset_index]
    t1 = time() - temp

    temp = time()
    core2 = protras.CoresetConstruction(data, 100)
    t2 = time() - temp

    print(t1)
    print(t2)


    plt.scatter(data[:, 0], data[:, 1], alpha=.3, color='green')
    plt.scatter(coreset[:, 0], coreset[:, 1], alpha=.3, color='red')
    plt.axis('equal')
    plt.plot()
    plt.show()
'''
