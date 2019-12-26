# -*- encoding:utf-8 -*-
# @Time    : 2019/12/25 22:54
# @Author  : jiang.g.f
# @File    : eval_crop.py
# @Software: PyCharm

import numpy as np

import cvtools
from cvtools.evaluation.merge_dets import ConvertGT2Dets, MergeCropDetResults
from cvtools.evaluation.mean_ap import eval_map


def eval_crop(ann_file, crop_ann_file):
    gt = ConvertGT2Dets(crop_ann_file)
    results = gt.convert()
    dets = MergeCropDetResults(crop_ann_file, results)
    merge_dets = dets.merge()

    coco = cvtools.COCO(ann_file)

    gt_bboxes = []
    gt_labels = []
    det_results = []
    for img_id, img_info in coco.imgs.items():
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        gt_bboxes.append(np.array([cvtools.x1y1wh_to_x1y1x2y2(ann['bbox'])
                                   for ann in anns]))
        gt_labels.append(np.array([ann['category_id'] for ann in anns]))
        try:
            bboxes, labels, scores = merge_dets[img_info['file_name']]
            bboxes = np.hstack([bboxes, scores[:, np.newaxis]])
        except ValueError:
            bboxes, labels = [], []
        dets = [[] for _ in range(len(coco.cats))]
        for cls, bbox in zip(labels, bboxes):
            dets[cls-1].append(bbox)
        for cls, det in enumerate(dets):
            dets[cls] = np.array(det) if len(det) > 0 else np.empty((0, 5))
        det_results.append(dets)

    eval_map(det_results, gt_bboxes, gt_labels, dataset='dota')
