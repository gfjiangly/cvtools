# -*- encoding:utf-8 -*-
# @Time    : 2019/7/14 11:24
# @Author  : gfjiang
# @Site    : 
# @File    : test_dota_to_coco.py
# @Software: PyCharm
import os.path as osp
import cvtools.label_convert

current_path = osp.dirname(__file__)


class TestDOTA2COCO(object):

    def test_convert(self):
        # must use absolute path to get files
        self.label_root = current_path + '/data/rscup/labelTxt/'
        self.image_root = current_path + '/data/rscup/images/'
        self.path_replace = {'\\': '/'}

        dota_to_coco = cvtools.label_convert.DOTA2COCO(
            self.label_root, self.image_root,
            classes=current_path + '/data/rscup/cls_id_map.txt',
            path_replace=self.path_replace, box_form='x1y1wh')
        dota_to_coco.convert()
        dota_to_coco.save_json(
            current_path + '/out/rscup/rscup_x1y1wh_polygon.json')
