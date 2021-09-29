import numpy as np
import csv
from timeit import time
import matplotlib.pyplot as plt
import Function.FileOperations as fo
import Function.ClusterSpark as cls
import Coreset.Lightweight as lw
import Coreset.Uniform as un
import Coreset.ProTraS as pt

def QuantizationError(X, Q):
    sum = 0
    for x in X:
        dist = np.sum(np.power(x - Q, 2), axis=1)
        d = np.min(dist)
        sum += d
    return sum

def RelativeError(x, x0):
    delta = np.abs(x - x0)
    return delta / x

def CalculateRelativeError(index):
    index = 14
    (X, dataname, datasize, K, datacode, csmaxsize) = fo.GetData(index)
    (coreset, timerec, cluster_id, cluster_dis) = fo.ReadProTraS(index)
    C = X[coreset]

    dt_cluster = cls.KMeansSpark(X, K)
    dt_quanErr = QuantizationError(X, dt_cluster)

    file = open('abc.csv', 'w', newline='')
    cssize = 424
    pt_coreset = C[0:cssize]
    pt_cluster = cls.KMeansSpark(pt_coreset, K)
    pt_quanErr = QuantizationError(X, pt_cluster)
    pt_relaErr = RelativeError(dt_quanErr, pt_quanErr)

    pt_time_ar = timerec[0:cssize]
    pt_time = np.sum(pt_time_ar)

    temp = time()
    lw_coreset_id = lw.CoresetConstruction(X, cssize)
    lw_time = time() - temp
    lw_coreset = X[lw_coreset_id]
    lw_cluster = cls.KMeansSpark(lw_coreset, K)
    lw_quanErr = QuantizationError(X, lw_cluster)
    lw_relaErr = RelativeError(dt_quanErr, lw_quanErr)

    temp = time()
    un_coreset_id = un.CoresetConstruction(X, cssize)
    un_time = time() - temp
    un_coreset = X[un_coreset_id]
    un_cluster = cls.KMeansSpark(un_coreset, K)
    un_quanErr = QuantizationError(X, un_cluster)
    un_relaErr = RelativeError(dt_quanErr, un_quanErr)

    print(index, cssize, un_relaErr, pt_relaErr, lw_relaErr, un_time, pt_time, lw_time)
    datawriter = csv.writer(file, delimiter=',')
    datawriter.writerow([index, cssize, un_relaErr, pt_relaErr, lw_relaErr, un_time, pt_time, lw_time])
    file.close()

#index,cssize,un_relaErr,pt_relaErr,lw_relaErr,un_time,pt_time,lw_time
def ExportRelativeError():
    link = 'results/RelativeErrors.csv'
    data = []
    file = open(link)
    text = file.read().splitlines()
    for temp in text:
        val = list(filter(None, temp.split(',')))
        data += [[int(val[0]), int(val[1]), float(val[2]), float(val[3]),
                  float(val[4]), float(val[5]), float(val[6]), float(val[7])]]
    file.close()
    return data

def Table_RelativeError():
    data = ExportRelativeError()
    for i in range(30):
        if i % 2 == 0:
            item = data[i]
            item1 = data[i+1]
            print(' & ', item[1], ' & ', '{0:.4f}'.format(item[2]), ' & ', '{0:.4f}'.format(item[3]), ' & ', '{0:.4f}'.format(item[4]),
                  ' & ', item1[1], ' & ', '{0:.4f}'.format(item1[2]), ' & ', '{0:.4f}'.format(item1[3]), ' & ', '{0:.4f}'.format(item1[4]))

def Table_Time():
    data = ExportRelativeError()
    for i in range(30):
        if i % 2 == 0:
            item = data[i]
            item1 = data[i+1]
            print(' & ', item[1], ' & ', '{0:.4f}'.format(item[5]), ' & ', '{0:.0f}'.format(item[6]), ' & ', '{0:.4f}'.format(item[7]),
                  ' & ', item1[1], ' & ', '{0:.4f}'.format(item1[5]), ' & ', '{0:.0f}'.format(item1[6]), ' & ', '{0:.4f}'.format(item1[7]))

def DrawGraph(index, title=''):
    data = ExportRelativeError()
    row1 = data[index*2]
    row2 = data[index*2 + 1]

    un = np.asarray([[row2[1], row2[2]], [row1[1], row1[2]]])
    pt = np.asarray([[row2[1], row2[3]], [row1[1], row1[3]]])
    lw = np.asarray([[row2[1], row2[4]], [row1[1], row1[4]]])
    fig = plt.figure()
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.plot(un[:, 0], un[:, 1], linestyle='-')
    plt.plot(pt[:, 0], pt[:, 1], linestyle=':')
    plt.plot(lw[:, 0], lw[:, 1], linestyle='--')
    #plt.plot(uniform[:, 0], uniform[:, 1], linestyle=':', color='black')
    #plt.plot(lightweight[:, 0], lightweight[:, 1], linestyle='--')
    #plt.plot(adaptive[:, 0], adaptive[:, 1], linestyle='-.')
    plt.legend(('Uniform Sampling', 'Improved ProTraS', 'Lightweight Coreset'),
               shadow=True, loc=(0.5, 0.1), handlelength=1.5, fontsize=12)

    plt.xlabel('Coreset size')
    plt.ylabel('Adjusted Rand Index')
    plt.plot()
    plt.xlim(140, 450)
    plt.ylim(0.00, 0.90)
    plt.show()