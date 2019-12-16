# -*- encoding:utf-8 -*-
# @Time    : 2019/11/17 17:53
# @Author  : gfjiang
# @Site    : 
# @File    : test_mirror.py
# @Software: PyCharm
import os.path as osp
import cv2.cv2 as cv
import numpy as np

import cvtools
from cvtools.data_augs.augmentations import horizontal_mirror, vertical_mirror

current_path = osp.dirname(__file__)


def test_mirror():
    image_file = 'P0641'
    image = cv.imread(
        osp.join(current_path, 'data/DOTA/images/{}.png'.format(image_file))
    )
    labels = cvtools.readlines(
        osp.join(current_path, 'data/DOTA/labelTxt/{}.txt'.format(image_file))
    )
    labels = labels[2:]
    bboxes = [list(map(float, line.split()[:8]))
              for line in labels if len(line) > 1]
    draw_img = cvtools.draw_boxes_texts(
        image.copy(), bboxes, box_format='polygon')
    cvtools.imwrite(draw_img, osp.join(
        current_path, 'out', '{}.png'.format(image_file)))
    mirror = cvtools.RandomMirror()

    # 测试水平镜像
    h_image, h_bboxes = horizontal_mirror(image.copy(), np.array(bboxes))
    # h_image = np.ascontiguousarray(h_image)
    draw_img = cvtools.draw_boxes_texts(h_image, h_bboxes, box_format='polygon')
    # print((h_image == draw_img).all())
    cvtools.imwrite(draw_img, osp.join(
        current_path, 'out', 'h_{}.png'.format(image_file)))

    # 测试垂直镜像
    v_image, v_bboxes = vertical_mirror(image.copy(), np.array(bboxes))
    draw_img = cvtools.draw_boxes_texts(v_image, v_bboxes, box_format='polygon')
    cvtools.imwrite(draw_img, osp.join(
        current_path, 'out', 'v_{}.png'.format(image_file)))

    # 测试水平、垂直镜像
    hv_image, hv_bboxes = mirror(image.copy(), bboxes)
    draw_img = cvtools.draw_boxes_texts(
        hv_image, hv_bboxes, box_format='polygon')
    cvtools.imwrite(draw_img, osp.join(
        current_path, 'out', 'hv_{}.png'.format(image_file)))
