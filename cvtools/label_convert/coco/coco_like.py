# -*- encoding:utf-8 -*-
# @Time    : 2019/7/14 11:50
# @Author  : gfjiang
# @Site    : 
# @File    : coco_like.py
# @Software: PyCharm
from cvtools.pycocotools.coco import COCO


class COCOLike(object):
    """coco-like datasets analysis"""
    def __init__(self, file):
        # coco api has written a COCO dataset representation class for us,
        # what data is needed to take it out
        self.COCO = COCO(file)


if __name__ == '__main__':
    # Mainly check if the file_name in self.COCO.imgs and cats in self.COCO are correct
    cocolike_datasets = COCOLike('../rscup/train_crop1920x1080_rscup_x1y1wh_polygen.json')
    pass
