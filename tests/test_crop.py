# -*- encoding:utf-8 -*-
# @Time    : 2019/12/16 21:10
# @Author  : jiang.g.f
# @File    : test_crop.py
# @Software: PyCharm
import os.path as osp
import cvtools.data_augs as augs

current_path = osp.dirname(__file__)


def test_sliding_crop():
    img_prefix = current_path + '/data/DOTA/images'
    ann_file = current_path + '/data/DOTA/dota_x1y1wh_polygon.json'
    dataset = augs.CocoDatasetForCrop(img_prefix, ann_file)
    crop_method = augs.CropImageInOrder(crop_w=1024, crop_h=1024, overlap=0.2)
    crop = augs.CropLargeImages(dataset, crop_method)
    crop.crop_for_train()
    crop.save(to_file=current_path+'/out/crop/train_dota_crop1024.json')

    # 对实例数较少的类别重采样
    crop.crop_for_train(over_samples={'roundabout': 100, })
    crop.save(to_file=current_path+'/out/crop/train_dota_crop1024+over.json')


def test_adaptive_crop():
    img_prefix = current_path + '/data/DOTA/images'
    ann_file = current_path + '/data/DOTA/dota_x1y1wh_polygon.json'
    dataset = augs.CocoDatasetForCrop(img_prefix, ann_file)

    crop_method = augs.CropImageAdaptive(
        overlap=0.1,      # 滑窗重合率
        iof_th=0.7,       # 超出裁剪范围iof阈值
        small_prop=0.5,   # 小目标比例阈值
        max_objs=100,     # 目标总数阈值
        size_th=1024,     # 滑窗最大尺寸阈值
        strict_size=True  # 是否严格遵循size_th约束
    )

    crop = augs.CropLargeImages(dataset, crop_method)
    crop.crop_for_train()
    crop.save(to_file=current_path+'/out/crop/train_dota_ada.json')
