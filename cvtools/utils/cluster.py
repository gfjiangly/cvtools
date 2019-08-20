# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/6/24 15:00
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import numpy as np
from sklearn.cluster import KMeans, DBSCAN


def k_means_cluster(data, n_clusters):
    if not isinstance(data, np.ndarray):
        raise RuntimeError('not supported data format!')
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(data)
    centroids = estimator.cluster_centers_
    return centroids


def DBSCAN_cluster(data, metric='euclidean'):
    # data量大后，计算机一直在进行磁盘读写，导致死机
    y_db_pre = DBSCAN(eps=25., min_samples=10, metric=metric).fit_predict(data)
    print(y_db_pre)

