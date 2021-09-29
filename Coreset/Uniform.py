import numpy as np

'''
----------------------------------------------------------------------------------------------------
Uniform Sampling Coreset
By 
# Input: X = dataset; m = number of coreset elements
# Output: C = coreset index (in X)
----------------------------------------------------------------------------------------------------
'''
def CoresetConstruction(X, m):
    n = X.__len__()
    q = np.zeros(n) + 1/n
    C = np.random.choice(X.shape[0], size=m, replace=False, p=q)
    return C
