# -*- encoding:utf-8 -*-
# @Time    : 2019/4/18 21:21
# @Author  : gfjiang
# @Site    :
# @File    : iou.py
# @Software: PyCharm
import numpy as np
import platform

if platform.system() == "Linux":
    import polyiou


def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)

    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious


def polygon_overlaps(polygons1, polygons2, mode='iou'):
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
            try:
                ious[i, j] = polyiou.iou_poly(
                    polyiou.VectorDouble(polygons1[i][:8].tolist()),
                    polyiou.VectorDouble(polygons2[j][:8].tolist()))
            except IndexError:
                ious[i, j] = 0.
    if exchange:
        ious = ious.T
    return ious


# 两种box形式均已测试
def box_iou(b1, b2, center=False):
    """Return iou tensor

    Parameters
    ----------
    b1: array, shape=(i1,...,iN, 4), xywh or x1y1x2y2
    b2: array, shape=(j, 4), xywh or x1y1x2y2
    center: box format

    Returns
    -------
    iou: array, shape=(i1,...,iN, j)"""
    if not isinstance(b1, np.ndarray):
        b1 = np.array([b1])
    if not isinstance(b2, np.ndarray):
        b2 = np.array([b2])

    # 这是学习数组广播一个非常好的例子
    # 当两个数组维度无法广播时，不要尝试手动扩展复制数据，
    # 而应该在需要广播的方向插入一个维度，这样数据就会顺着这个插入的1维度进行广播

    # Expand dim to apply broadcasting.
    b1 = b1[:, np.newaxis, :]
    b1_wh = b1[..., [2, 3]] - b1[..., [0, 1]]
    b1_mins = b1[..., :2]
    b1_maxes = b1[..., 2:4]
    if center:
        b1_wh = b1[..., 2:4]
        b1_mins = b1[..., :2] - b1_wh / 2.
        b1_maxes = b1[..., :2] + b1_wh / 2.

    # Expand dim to apply broadcasting.
    b2 = b2[np.newaxis, :, :]
    b2_wh = b2[..., [2, 3]] - b2[..., [0, 1]]
    b2_mins = b2[..., :2]
    b2_maxes = b2[..., 2:4]
    if center:
        b2_wh = b2[..., 2:4]
        b2_mins = b2[..., :2] - b2_wh / 2.
        b2_maxes = b2[..., :2] + b2_wh / 2.

    intersect_mins = np.maximum(b1_mins, b2_mins)
    intersect_maxes = np.minimum(b1_maxes, b2_maxes)
    intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


if __name__ == "__main__":
    BB = np.array([[100, 200, 120, 240], [140, 220, 150, 280], [180, 260, 200, 300]])
    BBGT = np.array([[108, 220, 130, 260], [130, 210, 145, 270]])
    # center = False
    # [100, 200, 120, 240] VS [108, 220, 130, 260]
    # intersect_x1y1x2y2 = 108, 220, 120, 240   intersect_wh = 12, 20
    # BB_area = 20*40, BBGT_area = 22*40
    # iou = 240 / (800 + 880 - 240) = 240 / 1440 = 0.16666
    # [140, 220, 150, 280] VS [130, 210, 145, 270]
    # intersect_x1y1x2y2 = 140, 220, 145, 270   intersect_wh = 5, 50
    # BB_area = 10*60, BBGT_area = 15*60
    # iou = 250 / (600 + 900 - 250) = 250 / 1250 = 0.2
    # center = True
    # [100, 200, 120, 240] VS [108, 220, 130, 260]
    # convert to xiyix2y2 format [40, 80, 160, 320] VS [43, 90, 173, 350]
    # intersect_x1y1x2y2 = 43, 90, 160, 320   intersect_wh = 117, 230
    # BB_area = 120*240, BBGT_area = 130*260
    # iou = 117*230 / (120*240 + 130*260 - 117*230) = 0.7539
    # [140, 220, 150, 280] VS [130, 210, 145, 270]
    # convert to xiyix2y2 format [65, 80, 215, 360] VS [57.5, 75, 202.5, 345]
    # intersect_x1y1x2y2 = 65, 80, 202.5, 345   intersect_wh = 137.5, 265
    # BB_area = 150*280, BBGT_area = 145*270
    # iou = 137.5*265 / (150*280 + 145*270 - 137.5*265) = 36437.5 / 44712.5 = 0.8149
    box_iou(BB, BBGT, center=True)
