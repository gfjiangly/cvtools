# -*- encoding:utf-8 -*-
# @Time    : 2020/12/17 20:48
# @Author  : jiang.g.f
import numpy as np
import cvtools

from cvtools.evaluation.mean_ap import eval_map
from cvtools.ops.polyiou import poly_overlaps


class EvalDOTADets:

    def __init__(self, dets, anns):
        """
        对OBB检测结果评估，支持全图或子图形式
        Args:
            dets: 检测结果，支持序列化为字典或list形式的pkl文件
            anns: COCO格式标注文件
        """
        self.dets = cvtools.load_pkl(dets)
        self.anns_like_coco = cvtools.COCO(anns)
        print('dets\'s type is {}'.format(type(self.dets)))
        self.gt_bboxes, self.gt_labels = self.get_gt_bboxes_labels()

    def dets_from_dict_to_list(self):
        """按anns内图片ID顺序将dets由dict形式转成可直接用于评估的list形式"""
        if isinstance(self.dets, list):
            return
        print('dets: from dict to list')

    def get_gt_bboxes_labels(self):
        gt_bboxes = []
        gt_labels = []
        # img_id顺序佷重要
        i = 0
        for img_id, img_info in self.anns_like_coco.imgs.items():
            ann_ids = self.anns_like_coco.getAnnIds(imgIds=[img_id])
            anns = self.anns_like_coco.loadAnns(ann_ids)
            gt_bboxes.append(np.array([ann['segmentation'][0] for ann in anns]))
            gt_labels.append(np.array([ann['category_id'] for ann in anns]))
            i += 1
            if i == 100:
                break
        return gt_bboxes, gt_labels

    def eval(self):
        return eval_map(self.dets, self.gt_bboxes, self.gt_labels,
                        dataset='dota', calc_ious=poly_overlaps)


if __name__ == '__main__':
    dets_dict = '../../tests/data/DOTA/eval/croped_dets.pkl'
    anns = '../../tests/data/DOTA/eval/DOTA_val1024.json'
    eval_data_dets = EvalDOTADets(dets_dict, anns)
    eval_data_dets.eval()
