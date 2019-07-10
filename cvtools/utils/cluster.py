# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/6/24 15:00
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import numpy as np
from sklearn.cluster import KMeans


def k_means_cluster(data, n_clusters):
    if not isinstance(data, np.ndarray):
        raise RuntimeError('not supported data format!')
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(data)
    centroids = estimator.cluster_centers_
    return centroids
