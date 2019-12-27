# -*- encoding:utf-8 -*-
# @Time    : 2019/10/16 21:14
# @Author  : gfjiang
# @Site    : 
# @File    : test_boxes.py
# @Software: PyCharm
import numpy as np

import cvtools
from cvtools.utils.boxes import cut_polygon


def test_rotate_rect():
    # rects = np.array([[100, 200, 300, 400], [101, 201, 301, 401]])
    # center = np.array([[200, 300], [201, 301]])
    # angle = np.array([45, 30])
    # new_boxes = cvtools.rotate_rects(rects, center, angle)
    # print(new_boxes)

    rect = [100, 200, 300, 400]
    center = [200, 300]
    angle = 45
    new_rect = cvtools.rotate_rect(rect, center, angle)
    print(new_rect)


def test_cut_polygon():
    a = [(15, 15), (10, 10), (10, 20), (5, 15)]
    b = [(0, 0), (12.5, 0), (12.5, 30), (0, 30)]
    r = cut_polygon(a, b)
    assert len(r[0].shape) == 2 and r[0].shape[1] == 2
    assert len(r[1]) == 4
