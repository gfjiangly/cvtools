# -*- encoding:utf-8 -*-
# @Time    : 2019/11/20 16:12
# @Author  : gfjiang
# @Site    : 
# @File    : crop.py
# @Software: PyCharm
"""顶层通用Crop类，此类无须修改。只需将CropDataset和CropMethod传递给此类。
支持对少样本类别过采样，支持自定义裁剪数据集，支持自定义裁剪方法"""
import os.path as osp
import numpy as np

import cvtools
from cvtools.data_augs.crop.crop_abc import Crop
from cvtools.data_augs.crop.crop_method import CropImageProtected


class CropLargeImages(Crop):

    def __init__(self, dataset, crop_method):
        self.dataset = dataset
        self.crop_method = crop_method
        self.img_boxes = []
        self.crops = []
        self.cat_id_to_name = {
            cat['id']: cat['name']
            for cat in self.dataset.crop_dataset['categories']
        }
        self.crop_for_protected = CropImageProtected()

    def crop_for_train(self, over_samples=None):
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            # 索引或迭代dataset必须提供包含image字段和anns字段信息
            img = cvtools.imread(data['image'])
            anns = data['anns']
            self.crop_method.crop(img, anns)
            # croped可为空，即没有任何裁剪，同时原始图亦不保留
            cropped = self.crop_method.match_anns(anns)

            # 过采样扩展，对少样本类别过采样
            if over_samples is None:
                continue
            # self.crop_for_protected.size_th = max(img.shape[:2])
            for over_cat in over_samples:
                # 选出少样本类别实例
                protected_anns, protected_ann_ids = [], []
                for ann_index, ann in enumerate(anns):
                    if self.cat_id_to_name[ann['category_id']] == over_cat:
                        protected_anns.append(ann)
                        protected_ann_ids.append(ann_index)
                if len(protected_anns) == 0: continue
                for _ in range(over_samples[over_cat]):
                    if len(self.crop_for_protected(img, protected_anns)):
                        add_croped = self.crop_for_protected.match_anns(
                            anns)  # fix bug! must using all anns
                        cropped.update(add_croped)
                    else:
                        print('Protection cropping failure!')

            self.crops.append(cropped)
            print('crop image %d of %d: %s' %
                  (i, len(self.dataset), osp.basename(data['image'])))

    def crop_for_test(self):
        pass

    def save(self, to_file):
        self.dataset.save(self.crops, to_file=to_file)


if __name__ == '__main__':
    cvtools._DEBUG = False
    # 滑动窗口裁剪
    # from cvtools.data_augs.crop.crop_dataset import CocoDatasetForCrop
    # # from cvtools.data_augs.crop.crop_method import CropImageInOrder
    # from cvtools.data_augs.crop.crop_method import CropImageAdaptive
    # mode = 'val'
    # img_prefix = 'D:/data/DOTA/{}/images'.format(mode)
    # ann_file = 'D:/data/DOTA/annotations/{}_dota+original.json'.format(mode)
    # dataset = CocoDatasetForCrop(img_prefix, ann_file)
    # # crop_method = CropImageInOrder(1024, 1024, 0.2)
    # crop_method = CropImageAdaptive(overlap=0.1, iof_th=1., max_objs=500, size_th=1024)
    # crop = CropLargeImages(dataset, crop_method)
    # crop.crop_for_train()
    # # to_file = 'D:/data/DOTA/annotations/{}_dota+crop1024x1024.json'.format(mode)
    # to_file = 'D:/data/DOTA/annotations/{}_dota+newcrop.json'.format(mode)
    # crop.save(to_file=to_file)

    # 根据一定规则裁剪
    from cvtools.data_augs.crop.crop_dataset import CocoDatasetForCrop
    from cvtools.data_augs.crop.crop_method import CropImageAdaptive
    mode = 'train'
    img_prefix = 'D:/data/DOTA/{}/images'.format(mode)
    ann_file = 'D:/data/DOTA/annotations/{}_dota+original.json'.format(mode)
    dataset = CocoDatasetForCrop(img_prefix, ann_file)
    crop_method = CropImageAdaptive(overlap=0.1, iof_th=0.5, max_objs=500, size_th=1024)
    crop = CropLargeImages(dataset, crop_method)
    crop.crop_for_train(
        over_samples={
            'helicopter': 10,
            'soccer-ball-field': 10,
            'bridge': 2,
            'ground-track-field': 15
        }
    )
    to_file = 'D:/data/DOTA/annotations/{}_dota+newcrop+over.json'.format(mode)
    crop.save(to_file=to_file)
