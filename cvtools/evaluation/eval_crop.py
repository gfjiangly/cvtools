# -*- encoding:utf-8 -*-
# @Time    : 2019/12/25 22:54
# @Author  : jiang.g.f
# @File    : eval_crop.py
# @Software: PyCharm

import numpy as np
import platform

import cvtools
from cvtools.evaluation.merge_dets import MergeCropDetResults
from cvtools.evaluation.mean_ap import eval_map
from cvtools.utils.iou import polygon_overlaps, bbox_overlaps

if platform.system() == "Linux":
    import poly_nms


class EvalCropQuality(object):
    """此类设计目前不够完善，convert_crop_gt应隐藏在内部"""
    def __init__(self,
                 ann_file,
                 crop_ann_file,
                 convert_crop_gt=None,
                 num_coors=4):
        assert num_coors in (4, 8), "不支持的检测位置表示"
        self.num_coors = num_coors
        self.coco = cvtools.COCO(ann_file)
        self.anns = self.coco.anns
        self.calc_ious = (
            bbox_overlaps if self.num_coors == 4 else polygon_overlaps)
        if self.num_coors == 4:
            self.nms = cvtools.py_cpu_nms
        else:
            self.nms = poly_nms.poly_gpu_nms
        if convert_crop_gt is None:
            gt = cvtools.COCO2Dets(crop_ann_file)
        else:
            gt = convert_crop_gt(crop_ann_file)
        self.results = gt.convert()
        dets = MergeCropDetResults(crop_ann_file, self.results, self.num_coors)
        self.merge_dets = dets.merge(self.nms)

    def handel_ann(self, ann):
        if self.num_coors == 4:
            bboxes = cvtools.x1y1wh_to_x1y1x2y2(ann['bbox'])
        elif self.num_coors == 8:
            bboxes = ann['segmentation'][0]
        else:
            raise RuntimeError('不支持的坐标数！')
        return bboxes

    def eval(self):
        gt_bboxes = []
        gt_labels = []
        det_results = []
        for img_id, img_info in self.coco.imgs.items():
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            anns = self.coco.loadAnns(ann_ids)
            gt_bboxes.append(np.array([self.handel_ann(ann) for ann in anns]))
            gt_labels.append(np.array([ann['category_id'] for ann in anns]))
            try:
                bboxes, labels, scores = self.merge_dets[img_info['file_name']]
                bboxes = np.hstack([bboxes, scores[:, np.newaxis]])
            except ValueError:
                bboxes, labels = [], []
            dets = [[] for _ in range(len(self.coco.cats))]
            for cls, bbox in zip(labels, bboxes):
                dets[cls - 1].append(bbox)
            for cls, det in enumerate(dets):
                dets[cls] = np.array(det) if len(det) > 0 else \
                    np.empty((0, self.num_coors+1))
            det_results.append(dets)

        eval_map(det_results, gt_bboxes, gt_labels, dataset='dota',
                 calc_ious=self.calc_ious)
