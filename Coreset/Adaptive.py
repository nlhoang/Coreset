import numpy as np
import scipy.linalg
import scipy.cluster
import scipy.spatial

'''
----------------------------------------------------------------------------------------------------
Adaptive Sampling Coreset
By Feldman et al. in 'Scalable Training of Mixture Models via Coresets' (2011)
Coded by https://github.com/jiohyoo/gamelanpy
# Input: X = dataset; K = number of clusters; m = number of coreset elements; delta
# Output: C = coreset index (of X); C_w = weight of element in C
----------------------------------------------------------------------------------------------------
'''

def CoresetConstruction(X, K, coreset_size, delta=0.1):
    num_frames, num_vars = X.shape
    data_remain = X.copy()
    samples = [[0]*num_vars]

    # first, do the subsampling : pick core samples, and remove closest point to it
    num_iters = 0
    num_single_samples = int(1.0 * num_vars * K * np.log(1.0 / delta))

    while data_remain.shape[0] > num_single_samples:
        num_frames_remain = data_remain.shape[0]
        idx = np.random.permutation(num_frames_remain)[:num_single_samples]
        single_samples = data_remain[idx, :]

        # Here we define similarity matrix, based on some measure of similarity or kernel. Feel free to change
        dists = scipy.spatial.distance.cdist(data_remain, single_samples)

        # minimum distance from random samples
        min_dists = np.min(dists, axis=1)
        # median distance
        v = np.median(min_dists)

        # remove rows with distance <= median distance
        remove_idx = np.where(min_dists <= v)[0]

        # remove rows of remove_idx
        data_remain = np.delete(data_remain, remove_idx, 0)
        samples = np.vstack((samples, single_samples))
        num_iters += 1

    samples = np.vstack((samples, data_remain))

    # now compute the weights of all the points, according to how close they are to the closest core-sample.
    db_size = np.zeros(samples.shape[0])
    min_dists = np.zeros(num_frames)
    closest_sample_idx = np.zeros(num_frames)
    for i in range(num_frames):
        dists = scipy.spatial.distance.cdist(X[i:i+1, :], samples)
        min_dist = np.min(dists)
        min_idx = np.argmin(dists)
        min_dists[i] = min_dist
        closest_sample_idx[i] = min_idx

    for i in range(num_frames):
        # for each datapoint, idx[i] is the index of the coreset point it is assigned to.
        db_size[int(closest_sample_idx[i])] += 1

    sq_sum_min_dists = (min_dists ** 2).sum()
    m = np.zeros(num_frames)
    for i in range(num_frames):
        m[i] = np.ceil(5.0 / db_size[int(closest_sample_idx[i])] + (min_dists[i] ** 2) / sq_sum_min_dists)

    m_sum = m.sum()
    cdf = (1.0 * m / m_sum).cumsum()
    C = []
    C_w = np.zeros(coreset_size)

    # Now, sample from the weighted points, to generate final corset and the corresponding weights
    for i in range(coreset_size):
        again = True
        while (again):
            r = np.random.rand()
            idx = (cdf <= r).sum()
            if idx in C:
                again = True
            else:
                again = False
                C += [idx]

        #C_w[i] = m_sum / (coreset_size * m[idx])

    return np.asarray(C)