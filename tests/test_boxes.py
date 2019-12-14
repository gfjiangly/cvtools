# -*- encoding:utf-8 -*-
# @Time    : 2019/10/16 21:14
# @Author  : gfjiang
# @Site    : 
# @File    : test_boxes.py
# @Software: PyCharm
import numpy as np

import cvtools


def test_rotate_rects():
    rects = np.array([[100, 200, 300, 400], [101, 201, 301, 401]])
    center = np.array([[200, 300], [201, 301]])
    angle = np.array([45, 30])
    new_boxes = cvtools.rotate_rects(rects, center, angle)
    print(new_boxes)
