import numpy as np

'''
----------------------------------------------------------------------------------------------------
Lightweight Coreset
By Bachem et al. in 'Scalable k-Means Clustering via Lightweight Coresets' (2018)
# Input: X = dataset; m = number of coreset elements
# Output: C = coreset index (in X); C_w = weight of element in C
----------------------------------------------------------------------------------------------------
'''
def CoresetConstruction(X, m):
    meanX = np.mean(X, axis=0)
    dist = np.sum(np.power(X - meanX, 2), axis=1)
    q = 1/2 * 1/X.shape[0] + 1/2 * (dist/dist.sum())
    C = np.random.choice(X.shape[0], size=m, replace=False, p=q)
    #C_w = 1.0/(m*q[C])
    return C

'''
----------------------------------------------------------------------------------------------------
Improved Lightweight Coreset
By NLHOANG
# Input: X = dataset; m = number of coreset elements
# Output: C = coreset index (in X)
----------------------------------------------------------------------------------------------------
'''
def CoresetConstruction2(X, m):
    meanX = np.mean(X, axis=0)
    dist = np.sum(np.power(X - meanX, 2), axis=1)
    q = 2/5 * 1/X.shape[0] + 3/5 * (dist/dist.sum())
    C = np.random.choice(X.shape[0], size=m, replace=False, p=q)
    return C

