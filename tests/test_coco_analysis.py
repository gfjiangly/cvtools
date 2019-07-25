# -*- encoding:utf-8 -*-
# @Time    : 2019/7/24 22:52
# @Author  : gfjiang
# @Site    : 
# @File    : test_coco_analysis.py
# @Software: PyCharm
import os.path as osp


import cvtools


class TestCocoAnalysis(object):

    def test_crop_in_order_for_test(self):
        img_prefix = 'data/rscup/images/'
        coco_analysis = cvtools.COCOAnalysis(img_prefix)
        save = 'rscup/test_rscup2019_crop800x800.json'
        coco_analysis.crop_in_order_for_test(save, w=800., h=800., overlap=0.1)
        assert osp.isfile(save)
