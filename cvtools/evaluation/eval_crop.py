# -*- encoding:utf-8 -*-
# @Time    : 2019/12/25 22:54
# @Author  : jiang.g.f
# @File    : eval_crop.py
# @Software: PyCharm

import numpy as np

import cvtools
from cvtools.evaluation.merge_crop_dets import MergeCropDetResults
from cvtools.evaluation.mean_ap import eval_map
from cvtools.utils.iou import bbox_overlaps
from cvtools.ops.polynms import poly_nms
from cvtools.ops.polyiou import poly_overlaps


class EvalCropQuality(object):
    """此类设计目前不够完善，convert_crop_gt应隐藏在内部"""
    def __init__(self,
                 ann_file,
                 crop_ann_file,
                 results=None,
                 num_coors=4):
        assert num_coors in (4, 8), "不支持的检测位置表示"
        self.num_coors = num_coors
        self.coco = cvtools.COCO(ann_file)
        self.anns = self.coco.anns
        self.calc_ious = (
            bbox_overlaps if self.num_coors == 4 else poly_overlaps)
        if self.num_coors == 4:
            self.nms = cvtools.py_cpu_nms
        else:
            self.nms = poly_nms
        self.results = results
        if cvtools.is_str(crop_ann_file):
            if results is None:
                gt = cvtools.COCO2Dets(crop_ann_file)
                self.results = gt.convert()
            else:
                self.results = crop_ann_file
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

        return eval_map(det_results, gt_bboxes, gt_labels, dataset='dota',
                        calc_ious=self.calc_ious)


if __name__ == '__main__':
    anns = '../../tests/data/DOTA/eval/val_dota_original.json'
    crop_anns = '../../tests/data/DOTA/eval/val_dota_crop800.json'
    crop_anns = cvtools.COCO(crop_anns)
    results = '../../tests/data/DOTA/eval/dets.pkl'
    results = cvtools.load_pkl(results)
    eval_crop_quality = EvalCropQuality(anns, crop_anns, results,
                                        num_coors=8)
    mean_ap, eval_results = eval_crop_quality.eval()
    """
    outputs:
        No module named 'cvtools.ops.polyiou.poly_overlaps'
        No module named 'cvtools.ops.polynms.poly_nms'
        loading annotations into memory...
        Done (t=0.40s)
        creating index...
        index created!
        loading annotations into memory...
        Done (t=0.21s)
        creating index...
        index created!
        +--------------------+------+-------+--------+-----------+-------+
        | class              | gts  | dets  | recall | precision | ap    |
        +--------------------+------+-------+--------+-----------+-------+
        | large-vehicle      | 4387 | 19356 | 0.893  | 0.202     | 0.824 |
        | swimming-pool      | 440  | 3043  | 0.786  | 0.114     | 0.603 |
        | helicopter         | 73   | 1214  | 0.411  | 0.025     | 0.157 |
        | bridge             | 464  | 16303 | 0.653  | 0.019     | 0.398 |
        | plane              | 2531 | 9607  | 0.944  | 0.249     | 0.924 |
        | ship               | 8960 | 22159 | 0.814  | 0.329     | 0.776 |
        | soccer-ball-field  | 153  | 3680  | 0.458  | 0.019     | 0.302 |
        | basketball-court   | 132  | 1707  | 0.773  | 0.060     | 0.662 |
        | ground-track-field | 144  | 2671  | 0.590  | 0.032     | 0.495 |
        | small-vehicle      | 5438 | 28363 | 0.758  | 0.145     | 0.494 |
        | harbor             | 2090 | 10070 | 0.714  | 0.148     | 0.633 |
        | baseball-diamond   | 214  | 1887  | 0.883  | 0.100     | 0.782 |
        | tennis-court       | 760  | 2993  | 0.930  | 0.236     | 0.923 |
        | roundabout         | 179  | 3829  | 0.799  | 0.037     | 0.665 |
        | storage-tank       | 2888 | 13410 | 0.827  | 0.178     | 0.776 |
        +--------------------+------+-------+--------+-----------+-------+
        | mAP                |      |       |        |           | 0.628 |
        +--------------------+------+-------+--------+-----------+-------+
    """
