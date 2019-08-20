# -*- encoding:utf-8 -*-
# @Time    : 2019/8/20 11:09
# @Author  : gfjiang
# @Site    : 
# @File    : convert.py
# @Software: PyCharm
import cvtools


def convert_dota(mode='train', cls_map='dota_v1.0_cat_id_map.txt'):
    label_root = 'D:/data/DOTA/{}/labelTxt-v1.0/labelTxt/'.format(mode)
    image_root = 'D:/data/DOTA/{}/images/'.format(mode)
    path_replace = {'\\': '/'}
    dota_to_coco = cvtools.DOTA2COCO(label_root, image_root, cls_map=cls_map,
                                     path_replace=path_replace, box_form='x1y1wh')
    dota_to_coco.convert()
    dota_to_coco.save_json('{}_dota_x1y1wh_polygen.json'.format(mode))


def convert_crop_dota(mode='train', cls_map='dota_v1.0_cat_id_map.txt'):
    crop_label_root = '../../data_augs/dota/{}/crop800x800/labelTxt+crop'.format(mode)
    image_root = 'D:/data/DOTA/{}/images/'.format(mode)
    path_replace = {'\\': '/'}
    dota_to_coco = cvtools.DOTA2COCO(crop_label_root, image_root, cls_map=cls_map,
                                     path_replace=path_replace, box_form='x1y1wh')
    dota_to_coco.convert(use_crop=True)
    dota_to_coco.save_json('{}_crop800x800_dota_x1y1wh_polygen.json'.format(mode))


if __name__ == '__main__':
    convert_crop_dota(mode='train')

