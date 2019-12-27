# -*- encoding:utf-8 -*-
# @Time    : 2019/12/26 16:00
# @Author  : jiang.g.f
# @File    : coco_to_dets.py
# @Software: PyCharm

import numpy as np
from collections import defaultdict
import cv2.cv2 as cv

import cvtools


class COCO2Dets(object):
    """
        将DOTA-COCO兼容格式GT转成检测结果表达形式results，保存成pkl
        results:
        {
            image_id: dets,  # image_id必须是anns中有效的id
            image_id: dets,
            ...
        }
        dets:
        {
            类别：[[位置坐标，得分], [...], ...],
            类别: [[位置坐标，得分], [...], ...],
            ...
        }，
    """
    def __init__(self, anns_file, num_coors=4):
        assert num_coors in (4, 8), "不支持的检测位置表示"
        self.coco = cvtools.COCO(anns_file)
        self.results = defaultdict()    # 动态创建嵌套字典
        self.num_coors = num_coors

    def handle_ann(self, ann):
        """如果想自定义ann处理方式，继承此类，然后重新实现此方法"""
        if self.num_coors == 4:
            bboxes = cvtools.x1y1wh_to_x1y1x2y2(ann['bbox'])
        elif self.num_coors == 8:
            segm = ann['segmentation'][0]
            if len(segm) != 8:
                segm_hull = cv.convexHull(
                    np.array(segm).reshape(-1, 2).astype(np.float32),
                    clockwise=False)
                xywha = cv.minAreaRect(segm_hull)
                segm = cv.boxPoints(xywha).reshape(-1).tolist()
            bboxes = segm
        else:
            raise RuntimeError('不支持的坐标数！')
        return bboxes + [1.]

    def convert(self, to_file=None):
        for img_id, img_info in self.coco.imgs.items():
            dets = defaultdict(list)
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                dets[ann['category_id']].append(self.handle_ann(ann))
            for cls, det in dets.items():
                dets[cls] = np.array(det, dtype=np.float)
            self.results[img_id] = dets
        if to_file is not None:
            self.save_pkl(to_file)
        return self.results

    def save_pkl(self, to_file):
        cvtools.dump_pkl(self.results, to_file)

    def save_json(self, to_file):
        cvtools.dump_json(self.results, to_file)
