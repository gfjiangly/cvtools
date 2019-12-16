# -*- encoding:utf-8 -*-
# @Time    : 2019/7/24 22:52
# @Author  : gfjiang
# @Site    : 
# @File    : test_coco_analysis.py
# @Software: PyCharm
import os.path as osp
import cvtools

current_path = osp.dirname(__file__)


class TestCocoAnalysis(object):

    def test_vis(self):
        img_prefix = current_path + '/data/DOTA/images'
        ann_file = current_path + '/data/DOTA/dota_x1y1wh_polygon.json'
        coco_analysis = cvtools.COCOAnalysis(img_prefix, ann_file)

        # 测试segmentation可视化
        coco_analysis.vis_instances(
            current_path + '/out/dota/vis/all',
            vis='segmentation',
            box_format='polygon'
        )

        # 测试bbox可视化，按类别输出
        coco_analysis.vis_instances(
            current_path + '/out/dota/vis/cats',
            vis='bbox',
            vis_cats=['large-vehicle', 'ship'],
            box_format='x1y1wh'
        )
