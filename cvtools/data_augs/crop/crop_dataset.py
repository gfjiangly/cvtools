# -*- encoding:utf-8 -*-
# @Time    : 2019/11/20 14:53
# @Author  : gfjiang
# @Site    : 
# @File    : crop_dataset.py
# @Software: PyCharm
"""所有需要裁剪的数据集需继承CropDataset，必须实现__getitem__和__len__接口"""
import os.path as osp
import numpy as np
import copy
from shapely.geometry import Polygon

import cvtools
from cvtools.data_augs.crop.crop_abc import CropDataset
from cvtools.data_augs.crop.crop_abc import cal_iof
from cvtools.utils.boxes import x1y1wh_to_x1y1x2y2x3y3x4y4


"""
迭代器是一种最简单也最常见的设计模式。
它可以让用户透过特定的接口巡访容器中的每一个元素而不用了解底层的实现。
"""


class CocoDatasetForCrop(CropDataset):

    def __init__(self,
                 img_prefix,
                 ann_file):
        super(CocoDatasetForCrop, self).__init__()
        self.img_prefix = img_prefix
        self.ann_file = ann_file
        self.crop_dataset = cvtools.load_json(ann_file)
        self.COCO = cvtools.COCO(ann_file)
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        self.roidb = self.COCO.loadImgs(image_ids)
        if cvtools._DEBUG:
            self.roidb = self.roidb[:50]

    def __getitem__(self, item):
        entry = self.roidb[item]
        image_name = entry['file_name']
        image_file = osp.join(self.img_prefix, image_name)
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        anns = self.COCO.loadAnns(ann_ids)
        anns = [
            {'bbox': ann['bbox'],
             'segmentation': ann['segmentation'],
             'category_id': ann['category_id']
             }  # 此处可以订制
            for ann in anns]
        return {'image': image_file, 'anns': anns}

    def __len__(self):
        return len(self.roidb)

    def save(self, crops, to_file):
        new_images = []
        new_annotations = []
        image_id = 1
        ann_id = 1
        for image_i in range(len(self.roidb)):
            image_info = self.roidb[image_i]
            ann_ids = self.COCO.getAnnIds(imgIds=image_info['id'], iscrowd=None)
            anns = self.COCO.loadAnns(ann_ids)
            if not crops[image_i]:
                continue
            for img_box, ann_indexes in crops[image_i].items():
                new_image_info = copy.deepcopy(image_info)
                new_image_info['crop'] = img_box
                new_image_info['id'] = image_id
                new_images.append(new_image_info)
                crop_anns = [anns[index] for index in ann_indexes]
                # 不能修改原始数据，因为同一个ann可能分布在多个图片中
                crop_anns = copy.deepcopy(crop_anns)
                self.recalc_anns(img_box, crop_anns)
                for ann in crop_anns:
                    self.trans_ann(ann, img_box)
                    ann['id'] = ann_id
                    ann['image_id'] = image_id
                    new_annotations.append(ann)
                    ann_id += 1
                image_id += 1
        self.crop_dataset['images'] = new_images
        self.crop_dataset['annotations'] = new_annotations
        cvtools.save_json(self.crop_dataset, to_file)

    def trans_ann(self, ann, img_box):
        segm = np.array(ann['segmentation'][0]).reshape(-1, 2)
        new_segm = segm - np.array(img_box[:2]).reshape(-1, 2)
        ann['segmentation'] = [new_segm.reshape(-1).tolist()]
        ann['bbox'][0] -= img_box[0]
        ann['bbox'][1] -= img_box[1]

    def recalc_anns(self, img_box, anns):
        segms = np.array([ann['segmentation'][0] for ann in anns])
        boxes = [ann['bbox'] for ann in anns]
        iof = cal_iof(boxes, [img_box])[..., -1]
        cutted_segms_ids = np.where(iof < 1)[0]
        for id in cutted_segms_ids:
            segm = segms[id]
            img_box_polygon = np.array(x1y1wh_to_x1y1x2y2x3y3x4y4(
                cvtools.x1y1x2y2_to_x1y1wh(img_box))).reshape(-1, 2)
            inters, bounds = cut_polygon(np.array(segm).reshape(-1, 2),
                                         img_box_polygon)
            anns[id]['segmentation'] = [inters.reshape(-1).tolist()]
            anns[id]['bbox'] = cvtools.x1y1x2y2_to_x1y1wh(list(bounds))


def cut_polygon(polygon, box):
    # polygon可以不按点的顺序构建，但是必须使用convex_hull求交
    polygon = Polygon(polygon).convex_hull
    box = Polygon(box).convex_hull
    # boundary属性对应的LineString可以直接转为array对象
    # 最后一个点与第一个点相同，予以去除
    inters = polygon.intersection(box)
    try:
        intersections = np.array(inters.boundary)[:-1]
        bounds = np.array(inters.bounds)    # 最小外接矩形
    except ValueError:
        return None, None
    return intersections, bounds


if __name__ == '__main__':
    # mode = 'train'
    # img_prefix = 'D:/data/DOTA/{}/images'.format(mode)
    # ann_file = 'D:/data/DOTA/annotations/{}_dota+original.json'.format(mode)
    # dataset = CocoDatasetForCrop(img_prefix, ann_file)
    # for i in range(len(dataset)):
    #     data = dataset[i]
    #     print(data)

    a = [(15, 15), (10, 10), (10, 20), (5, 15)]
    b = [(0, 0), (12.5, 0), (12.5, 30), (0, 30)]
    cut_polygon(a, b)
