# -*- encoding:utf-8 -*-
# @Time    : 2019/7/14 11:24
# @Author  : gfjiang
# @Site    : 
# @File    : test_dota_to_coco.py
# @Software: PyCharm
import os


class TestDOTA2COCO(object):

    def test_convert(self):
        # must use absolute path to get files
        self.label_root = os.path.dirname(__file__) + '/data/rscup/labelTxt/'
        self.image_root = os.path.dirname(__file__) + '/data/rscup/images/'
        self.path_replace = {'\\': '/'}

        import cvtools.label_convert
        dota_to_coco = cvtools.label_convert.DOTA2COCO(
            self.label_root, self.image_root,
            cls_map='data/rscup/cls_id_map.txt',
            path_replace=self.path_replace, box_form='x1y1wh')
        dota_to_coco.convert()
        dota_to_coco.save_json(os.path.dirname(__file__) + '/rscup/rscup_x1y1wh_polygen.json')
