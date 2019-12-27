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
        self._ann_ids = []
        self._crop_ann_ids = []
        self._num_imgs = 0
        self._num_crop_imgs = 0

    def get_stats(self):
        num_anns = len(self._ann_ids)
        num_crop_anns = len(self._crop_ann_ids)
        num_missing_anns = num_anns - len(set(self._crop_ann_ids))
        stats = {
            'num_imgs': self._num_imgs,
            'num_crop_imgs': self._num_crop_imgs,
            'radio_img': round(self._num_crop_imgs / float(self._num_imgs), 2),
            'num_anns': num_anns,
            'num_crop_anns': num_crop_anns,
            'num_failing_anns': num_missing_anns,
            'radio_ann': round(num_crop_anns / float(num_anns), 2)
        }
        return stats

    def reset_stats(self):
        self._ann_ids = []
        self._crop_ann_ids = []
        self._num_imgs = 0
        self._num_crop_imgs = 0

    def __call__(self, img, anns):
        return self.crop(img, anns)

    def crop(self, img, anns=None):
        raise NotImplementedError

    def match_anns(self, anns):
        assert len(self.img_boxes) > 0
        reserved = defaultdict()
        gt_boxes = [ann['bbox'] for ann in anns]
        self._ann_ids += [ann['id'] for ann in anns]
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
                self._crop_ann_ids += [anns[ind]['id']
                                       for ind in index_reserved]
                self._num_crop_imgs += 1
        self._num_imgs += 1
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
