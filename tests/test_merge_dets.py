# -*- encoding:utf-8 -*-
# @Time    : 2019/12/25 17:02
# @Author  : jiang.g.f
# @File    : test_merge_dets.py
# @Software: PyCharm

import os.path as osp
import numpy as np
import cvtools
from cvtools.evaluation.merge_dets import ConvertGT2Dets, MergeCropDetResults

current_path = osp.dirname(__file__)


def test_convert_gt_to_dets():
    ann_file = current_path + '/data/DOTA/dota_crop_1024.json'
    gt = ConvertGT2Dets(ann_file)
    gt.convert(to_file=current_path + '/out/DOTA/dets.pkl')


def test_MergeCropDetResults():
    ann_file = current_path + '/data/DOTA/dota_crop_1024.json'
    results = current_path + '/out/DOTA/dets.pkl'

    gt = ConvertGT2Dets(ann_file)
    gt.convert(to_file=results)

    dets = MergeCropDetResults(ann_file, results)
    merge_dets = dets.merge()

    original_ann_file = current_path + '/data/DOTA/dota_x1y1wh_polygon.json'
    coco = cvtools.COCO(original_ann_file)

    gt_bboxes = []
    gt_labels = []
    det_results = []
    for img_id, img_info in coco.imgs.items():
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        gt_bboxes.append(np.array([ann['bbox'] for ann in anns]))
        gt_labels.append(np.array([ann['category_id'] for ann in anns]))
        bboxes, labels, _ = merge_dets[img_info['file_name']]
        dets = [[] for _ in range(len(coco.cats))]
        for cls, bbox in zip(labels, bboxes):
            dets[cls].append(bbox)
        for cls, det in enumerate(dets):
            dets[cls] = np.array(det)
        det_results.append(dets)

    from cvtools.evaluation.mean_ap import eval_map
    mean_ap, eval_results = eval_map(det_results, gt_bboxes, gt_labels)
    print(mean_ap, eval_results)
