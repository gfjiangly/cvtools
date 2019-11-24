# -*- encoding:utf-8 -*-
# @Time    : 2019/11/20 14:53
# @Author  : gfjiang
# @Site    : 
# @File    : crop_abc.py
# @Software: PyCharm
from collections import defaultdict
import numpy as np

import cvtools


class Crop(object):

    def crop_for_train(self):
        raise NotImplementedError

    def crop_for_test(self):
        raise NotImplementedError

    def save(self, to_file):
        raise NotImplementedError


class CropDataset(object):
    """An abstract class representing a Dataset for crop.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class CropMethod(object):

    def __init__(self, iof_th=0.7):
        self.iof_th = iof_th
        self.img_boxes = []

    def __call__(self, img, anns):
        return self.crop(img, anns)

    def crop(self, img, anns=None):
        raise NotImplementedError

    def match_anns(self, anns):
        assert len(self.img_boxes) > 0
        reserved = defaultdict()
        gt_boxes = [ann['bbox'] for ann in anns]
        if len(gt_boxes) == 0:
            iof = None
        else:
            iof = self.cal_iof(gt_boxes)
        for i, img_box in enumerate(self.img_boxes):
            index_reserved = list(
                set(np.where(iof[..., i] >= self.iof_th)[0])
            ) if iof is not None else []
            if len(index_reserved) > 0:     # 不允许无anns的crop子图
                reserved[tuple(img_box)] = index_reserved
        return reserved

    def cal_iof(self, gt_boxes):
        """iof: 行是gt_boxes，列是crop_imgs"""
        assert len(gt_boxes) > 0 and len(self.img_boxes) > 0
        gt_boxes = cvtools.x1y1wh_to_x1y1x2y2(np.array(gt_boxes))
        img_boxes = np.array(self.img_boxes)
        iof = cvtools.bbox_overlaps(gt_boxes, img_boxes, mode='iof')
        return iof


def cal_iof(gt_boxes, img_boxes):
    """
    img_boxes: x1,y1,x2,y2  gt_boxes: x1,y1,w,h
    iof: 行是gt_boxes，列是crop_imgs
    """
    assert len(gt_boxes) > 0 and len(img_boxes) > 0
    gt_boxes = cvtools.x1y1wh_to_x1y1x2y2(np.array(gt_boxes))
    img_boxes = np.array(img_boxes)
    iof = cvtools.bbox_overlaps(gt_boxes, img_boxes, mode='iof')
    return iof
