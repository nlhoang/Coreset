import numpy as np

'''
----------------------------------------------------------------------------------------------------
Data Generalization for Sampling Coreset
By NLHOANG
# Input: data; Coreset index+means+label
# Output: label of full data
----------------------------------------------------------------------------------------------------
'''
def DataGenForSampling(data, cs_index, cs_means, cs_label):
    n = data.__len__()
    m = cs_index.__len__()
    label = np.zeros(n) - 1
    for i in range(n):
        if i in cs_index:
            i_index_cs = int(np.where(cs_index == i)[0])
            label[i] = cs_label[i_index_cs]
        else:
            i_index = dist_near(data[i], cs_means)
            label[i] = i_index
    return label

#Find nearest center to point
def dist_near(point, Centers):
    dist = np.sum(np.power(point - Centers, 2), axis=1)
    return np.argmin(dist)

'''
----------------------------------------------------------------------------------------------------
Data Generalization for ProTraS Coreset
By NLHOANG 
# Input: data_info from ProTraS; Coreset index+label
# Output: label of full data
----------------------------------------------------------------------------------------------------
'''
def DataGenForProTraS(data_info, cs_index, cs_label):
    n = data_info.__len__()
    m = cs_index.__len__()
    label = np.zeros(n) - 1
    for i in range(n):
        if i in cs_index:
            if label[i]==-1:
                i_index_cs = int(np.where(cs_index == i)[0])
                label[i] = cs_label[i_index_cs]
        else:
            i_nearest = cs_index[int(data_info[i][0])]
            if label[i_nearest]==-1:
                i_nearest_index_cs = int(np.where(cs_index == i_nearest)[0])
                clusterid = cs_label[i_nearest_index_cs]
                label[i_nearest] = clusterid
                label[i] = clusterid
            else:
                label[i] = label[i_nearest]
    return label