# -*- encoding:utf-8 -*-
# @Time    : 2019/8/20 11:09
# @Author  : gfjiang
# @Site    : 
# @File    : convert.py
# @Software: PyCharm
import cvtools
from cvtools.data_augs.dota.crop import crop_in_order


def convert_dota(label_root, save, mode='train'):
    image_root = 'D:/data/DOTA/{}/images/'.format(mode)
    dota_to_coco = cvtools.DOTA2COCO(label_root, image_root)
    dota_to_coco.convert()
    dota_to_coco.save_json(save)


def convert_crop_dota(crop_label_root, save, mode='train'):
    image_root = 'D:/data/DOTA/{}/images/'.format(mode)
    dota_to_coco = cvtools.DOTA2COCO(crop_label_root, image_root)
    dota_to_coco.convert(use_crop=True)
    dota_to_coco.save_json(save)


if __name__ == '__main__':
    cvtools._DEBUG = False
    mode = 'val'

    label_root = 'D:/data/DOTA/{}/labelTxt-v1.0/labelTxt/'.format(mode)
    save = 'D:/data/DOTA/annotations/{}_dota+original.json'.format(mode)
    convert_dota(label_root, save, mode=mode)

    ann_file = 'D:/data/DOTA/annotations/{}_dota+original.json'.format(mode)
    save = 'D:/data/DOTA/crop/{}'.format(mode)
    crop_in_order(ann_file, save, mode, vis=False)

    crop_label_root = 'D:/data/DOTA/crop/{}/labelTxt+crop'.format(mode)
    save = 'D:/data/DOTA/annotations/{}_dota+crop.json'.format(mode)
    convert_crop_dota(crop_label_root, save, mode=mode)


