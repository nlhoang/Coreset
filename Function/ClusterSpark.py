import numpy as np
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans

def KMeansSpark(data, K):
    sc = SparkContext(appName="KMeansExample")
    data_sc = sc.parallelize(data)
    clusters = KMeans.train(data_sc, k=K)
    label = np.asarray(clusters.predict(data_sc).collect())
    sc.stop()
    return np.asarray(clusters.centers), label