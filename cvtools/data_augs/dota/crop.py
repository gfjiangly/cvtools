# -*- encoding:utf-8 -*-
# @Time    : 2019/8/20 13:55
# @Author  : gfjiang
# @Site    : 
# @File    : crop.py
# @Software: PyCharm
import cv2

import cvtools


def crop_in_order(mode='train'):
    img_prefix = 'D:/data/DOTA/{}/images'.format(mode)
    ann_file = '../../label_convert/dota/train_dota_x1y1wh_polygen.json'
    crop_in_order = cvtools.CropInOder(img_prefix=img_prefix, ann_file=ann_file,
                                       width_size=800, height_size=800, overlap=0.1)
    crop_in_order.crop_with_label(save_root='./{}'.format(mode), iof_th=0.5)


if __name__ == '__main__':
    crop_in_order(mode='train')
