# -*- encoding:utf-8 -*-
# @Time    : 2019/3/1 21:34
# @Author  : gfjiang
# @Site    : 
# @File    : coordinate_convert.py
# @Software: PyCharm

import numpy as np
from src.utils import read_file_to_list
from src.label_convert import draw_rect_test_label


# 框由人头扩大至人肩，往左右下三个方向扩展
def expand_rectangle(src, dst, left_scale=0.1, right_scale=0.1, bottom_scale=0.15):
    images_list = read_file_to_list(src)
    with open(dst, 'w') as f:
        for line in images_list:
            line = line.strip().split()
            if len(line) > 1:
                box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box[:, 0] = box[:, 0] - box_w * left_scale
                box[:, 2] = box[:, 2] + box_w * right_scale
                box[:, 3] = box[:, 3] + box_h * bottom_scale
                box[:, 0:2][box[:, 0:2] < 0] = 0  # 左上角坐标限幅
                temp = [','.join(list(map(str, coor))) for coor in box]
                line = ' '.join([line[0], ' '.join(temp)]) + '\n'
            else:
                line = line[0] + '\n'
            f.write(line)


if __name__ == "__main__":
    expand_rectangle("../datasets/elevator_20190106.txt",
                     "../datasets/elevator_20190106_convert.txt")
    draw_rect_test_label('../datasets/elevator_20190106_convert.txt',
                         '../results/images/elevator_20190106_convert/')
