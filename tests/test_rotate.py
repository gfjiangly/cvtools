# -*- encoding:utf-8 -*-
# @Time    : 2019/11/17 14:29
# @Author  : gfjiang
# @Site    : 
# @File    : test_rotate.py
# @Software: PyCharm
import os.path as osp
import cv2.cv2 as cv

import cvtools


current_path = osp.dirname(__file__)


def test_RandomRotate():
    image_file = 'P0126'
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
    rotate = cvtools.RandomRotate()

    rotate_image, rotate_bboxes = rotate(image, bboxes, rotate='rotate_90')
    draw_img = cvtools.draw_boxes_texts(
        rotate_image, rotate_bboxes, box_format='polygon')
    cvtools.imwrite(draw_img, osp.join(
        current_path, 'out', '90_{}.png'.format(image_file)))

    rotate_image, rotate_bboxes = rotate(image, bboxes, rotate='rotate_270')
    draw_img = cvtools.draw_boxes_texts(
        rotate_image, rotate_bboxes, box_format='polygon')
    cvtools.imwrite(draw_img, osp.join(
        current_path, 'out', '270_{}.png'.format(image_file)))

    rotate_image, rotate_bboxes = rotate(image, bboxes, rotate='rotate_180')
    draw_img = cvtools.draw_boxes_texts(
        rotate_image, rotate_bboxes, box_format='polygon')
    cvtools.imwrite(draw_img, osp.join(
        current_path, 'out', '180_{}.png'.format(image_file)))



