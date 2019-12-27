# -*- encoding:utf-8 -*-
# @Time    : 2019/12/26 16:13
# @Author  : jiang.g.f
# @File    : dets.py
# @Software: PyCharm

from collections import defaultdict
import copy
import itertools

import cvtools


class Dets(object):
    """
        Unified representation
        {
              "cats": [{"id": int, name": str, "supercategory": "str}, ...],
              "imgs": [{"id": int, "name": str}, ...],
              "dets": [det, det, ...]  # det为K*(n+1), n为box坐标数
        }
        det:
        {
            "id": int, "cls_id": int, "img_id": int, "det": list
        }
        Memory-efficient representation
        [
            [cls1_det, cls2_det, ...],  # image1
            [cls1_det, cls2_det, ...],  # image2
            ...
        ]   # 此种表示方式，信息不是完备的，需要有image_list和cat_list
        Image-first representation
        {
            image_id: dets,  # image_id一般与anns中image_id对应
            image2: dets,
            ...
        }
        dets:
        {
            class_id：[[位置坐标，得分], [...], ...],  # class_id从1开始
            class2: [[位置坐标，得分], [...], ...],
            ...
        }
        Category-first representation
        {
            class_id: dets,  # class_id从1开始，0表示一般背景
            class2: dets,
            ...
        }
        dets:
        {
            image_id：[[位置坐标，得分], [...], ...],
            image2: [[位置坐标，得分], [...], ...],
            ...
        }
    """
    def __init__(self, det_file=None, image_first=True):
        self.dets, self.cats, self.imgs = [], [], []
        self.imgToDets, self.catToDets = defaultdict(), defaultdict()
        if det_file is not None:
            self.imgToCatToDets = defaultdict()
            self.catToImgToDets = defaultdict()
            if image_first:
                self.imgToCatToDets = cvtools.load_pkl(det_file)
            else:
                self.catToImgToDets = cvtools.load_pkl(det_file)
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        if len(self.imgToDets) > 0:
            det_id = 0
            det_dict = {}
            for img_id, dets in self.imgToDets.items():
                for cls_id, det in dets.items():
                    det_dict["id"] = det_id
                    det_dict["cls_id"] = cls_id
                    det_dict["img_id"] = img_id
                    for box in det:
                        det_dict["det"] = box
                        self.dets.append(copy.deepcopy(det_dict))
                        self.cats.append(cls_id)
                        self.imgs.append(img_id)
                    det_id += 1

        elif len(self.catToDets) > 0:
            pass

        print('index created!')

    def imgDets2catDets(self):
        pass

    def getDetIds(self, imgIds=[], catIds=[]):
        """Get ann ids that satisfy given filter conditions.
        default skips that filter

        Args:
            imgIds (int array): get dets for given imgs
            catIds int array): get dets for given imgs
        """
        imgIds = imgIds if cvtools.is_array_like(imgIds) else [imgIds]
        catIds = catIds if cvtools.is_array_like(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            dets = self.dets
        else:
            if len(imgIds) > 0:
                lists = [self.imgToDets[imgId] for imgId in imgIds
                         if imgId in self.imgToDets]
                dets = list(itertools.chain.from_iterable(lists))
            else:
                dets = self.dets
            dets = dets if len(catIds) == 0 else \
                [dets for dets in dets if dets['cls_id'] in catIds]

        ids = [det['id'] for det in dets]
        return ids
