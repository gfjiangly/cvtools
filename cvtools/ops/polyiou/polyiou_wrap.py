# -*- encoding:utf-8 -*-
# @Time    : 2020/2/9 17:27
# @Author  : jiang.g.f
# @File    : polyiou_wrap.py
# @Software: PyCharm
import numpy as np
"""
polyiou只能计算两个单OBB的IOU
poly_overlaps可以矢量化计算OBB的IOU，但形式需要是5参数旋转角表示
"""
# try:
#     from .poly_overlaps import poly_overlaps as poly_overlaps_gpu
# except ImportError as e:
#     poly_overlaps_gpu = None
#     print(e)

from .polyiou import VectorDouble, iou_poly


def poly_overlaps_cpu(polygons1, polygons2, mode='iou'):
    assert mode in ['iou', 'iof']
    polygons1 = polygons1.astype(np.float32)
    polygons2 = polygons2.astype(np.float32)
    rows = polygons1.shape[0]
    cols = polygons2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if polygons1.shape[0] > polygons2.shape[0]:
        polygons1, polygons2 = polygons2, polygons1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    for i in range(polygons1.shape[0]):
        for j in range(polygons2.shape[0]):
            if mode == 'iou':
                try:
                    ious[i, j] = iou_poly(
                        VectorDouble(polygons1[i][:8].tolist()),
                        VectorDouble(polygons2[j][:8].tolist())
                    )
                except IndexError:
                    ious[i, j] = 0.
            else:
                raise NotImplemented("iof模式暂未实现！")
    if exchange:
        ious = ious.T
    return ious


def poly_overlaps(a, b, force_cpu=False):
    a = a.astype(np.float32)[:, :8]
    b = b.astype(np.float32)[:, :8]
    if len(a) == 0 or len(b) == 0:
        return np.empty(shape=(len(a), len(b)))
    # if poly_overlaps_gpu is not None and not force_cpu:
    #     return poly_overlaps_gpu(a, b)
    # else:
    #     # raise NotImplemented("poly_overlaps的cpu版本暂未实现！")
    return poly_overlaps_cpu(a, b)
