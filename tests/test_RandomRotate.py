# -*- encoding:utf-8 -*-
# @Time    : 2019/11/17 14:29
# @Author  : gfjiang
# @Site    : 
# @File    : test_RandomRotate.py
# @Software: PyCharm
import cv2.cv2 as cv

import cvtools


def test_RandomRotate():
    image_file = 'P2305'
    image = cv.imread('data/rscup/images/{}.png'.format(image_file))
    labels = cvtools.readlines('data/rscup/labelTxt/{}.txt'.format(image_file))[4:]
    bboxes = [list(map(float, line.split()[:8])) for line in labels if len(line) > 1]
    draw_img = cvtools.draw_boxes_texts(image.copy(), bboxes, box_format='polygon')
    cvtools.imwrite(draw_img, '{}.png'.format(image_file))
    rotate = cvtools.RandomRotate()

    rotate_image, rotate_bboxes = rotate(image, bboxes, rotate='rotate_90')
    draw_img = cvtools.draw_boxes_texts(rotate_image, rotate_bboxes, box_format='polygon')
    cvtools.imwrite(draw_img, '90_{}.png'.format(image_file))

    rotate_image, rotate_bboxes = rotate(image, bboxes, rotate='rotate_270')
    draw_img = cvtools.draw_boxes_texts(rotate_image, rotate_bboxes, box_format='polygon')
    cvtools.imwrite(draw_img, '270_{}.png'.format(image_file))

    rotate_image, rotate_bboxes = rotate(image, bboxes, rotate='rotate_180')
    draw_img = cvtools.draw_boxes_texts(rotate_image, rotate_bboxes, box_format='polygon')
    cvtools.imwrite(draw_img, '180_{}.png'.format(image_file))



