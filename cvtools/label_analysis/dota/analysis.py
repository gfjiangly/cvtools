# -*- encoding:utf-8 -*-
# @Time    : 2019/8/20 13:22
# @Author  : gfjiang
# @Site    : 
# @File    : analysis.py
# @Software: PyCharm
import cvtools


def analyze_dota(mode='train'):
    img_prefix = 'D:/data/DOTA/{}/images'.format(mode)
    ann_file = '../../label_convert/dota/train_dota_x1y1wh_polygen.json'
    coco_analysis = cvtools.COCOAnalysis(img_prefix, ann_file)

    coco_analysis.stats_num('num_per_img.json')
    coco_analysis.crop_in_order_with_label('crop800x800/{}'.format(mode),
                                           w=800., h=800., overlap=0.1)


