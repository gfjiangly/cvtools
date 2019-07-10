# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/7/3 10:15
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import pickle
import os
import cv2

from cvtools.utils import *


class DetsAnalysis(object):

    def __init__(self, dataset_file, dets_file):
        self.coco_dataset = load_json(dataset_file)
        self.dets = pickle.load(open(dets_file, 'rb'), encoding='utf-8')

    def vis_dets(self, save_root, box_format='x1y1x2y2', vis_score=0.5):
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        for image_info, det in zip(self.coco_dataset['images'], self.dets):
            print('Visualize {}'.format(image_info['file_name']))
            image_name = os.path.basename(image_info['file_name'])
            img = cv2.imread(image_info['file_name'].replace('/media/data1/jgf/', 'F:/data/'))
            # det is organized by category
            for cat_id, cat_boxes in enumerate(det):
                if len(cat_boxes) == 0:
                    continue
                class_name = self.coco_dataset['categories'][cat_id]['name']
                cat_boxes = cat_boxes[cat_boxes[:, 4] > vis_score]
                img = draw_box_text(img, cat_boxes[:, :4], [class_name]*len(cat_boxes), box_format=box_format)
            cv2.imwrite(os.path.join(save_root, image_name), img)


if __name__ == '__main__':
    dets_analysis = DetsAnalysis('Arcsoft/market_head_val.json',
                                 'tests/market_head_retinanet_r50_fpn_1x_results.pkl')
    dets_analysis.vis_dets('Arcsoft/vis_dets')
