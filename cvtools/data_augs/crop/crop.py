# -*- encoding:utf-8 -*-
# @Time    : 2019/11/20 16:12
# @Author  : gfjiang
# @Site    : 
# @File    : crop.py
# @Software: PyCharm
"""顶层通用Crop类，此类无须修改。只需将CropDataset和CropMethod传递给此类。
支持对少样本类别过采样，支持自定义裁剪数据集，支持自定义裁剪方法"""
import os.path as osp
from collections import defaultdict

import cvtools
from cvtools.data_augs.crop.crop_abc import Crop
from cvtools.data_augs.crop.crop_method import CropImageProtected


class CropLargeImages(Crop):

    def __init__(self, dataset, crop_method, over_strict=True):
        self.dataset = dataset
        self.crop_method = crop_method
        self.ovover_strict = over_strict
        self.crops = []
        self.cat_id_to_name = {
            cat['id']: cat['name']
            for cat in self.dataset.crop_dataset['categories']
        }
        self.crop_for_protected = CropImageProtected(strict=over_strict)

    def crop_for_train(self, over_samples=None):
        """训练集裁剪

        Args:
            over_samples (dict): {类别: 重采样次数， ...}
        """
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            # 索引或迭代dataset必须提供包含image字段和anns字段信息
            img = cvtools.imread(data['image'])
            anns = data['anns']
            self.crop_method.crop(img, anns)
            # croped可为空，即没有任何裁剪，同时原始图亦不保留
            cropped = self.crop_method.match_anns(anns)

            # 过采样扩展，对少样本类别过采样
            if over_samples is not None:
                add_croped = self.over_sample(img, anns, over_samples)
                cropped.update(add_croped)

            self.crops.append(cropped)
            print('crop image %d of %d: %s' %
                  (i, len(self.dataset), osp.basename(data['image'])))

        if hasattr(self.crop_method, 'stats_crop'):
            print(self.crop_method.stats_crop)

    def over_sample(self, img, anns, over_samples):
        add_crops = defaultdict()
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
                    add_crop = self.crop_for_protected.match_anns(
                        anns)  # fix bug! must using all anns
                    add_crops.update(add_crop)
                else:
                    print('Protection cropping failure!')
        return add_crops

    def crop_for_test(self):
        pass

    def save(self, to_file):
        self.dataset.save(self.crops, to_file=to_file)
        self.crops = []
