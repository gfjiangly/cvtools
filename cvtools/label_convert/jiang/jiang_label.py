# -*- encoding:utf-8 -*-
# @Time    : 2019/3/19 21:20
# @Author  : gfjiang
# @Site    : 
# @File    : jiang_label.py
# @Software: PyCharm
import os
from PIL import Image
from voc_label import convert


def convert_jiang_txt(file, ignore_class=False):
    with open(file, 'r', encoding='utf-8') as f:
        annotations = f.readlines()
    if not os.path.exists('VOCdevkit/VOC2007/labels'):
        os.makedirs('VOCdevkit/VOC2007/labels')
    for line_annotation in annotations:
        line_annotation = line_annotation.strip().split()
        img_path = line_annotation[0]
        out_file = open('VOCdevkit/VOC2007/labels/%s' % os.path.basename(img_path).replace('.jpg', '.txt'), 'w')
        if len(line_annotation) < 2:
            out_file.close()
            continue
        image = Image.open(img_path)
        w, h = image.size
        boxes = line_annotation[1:]
        for box in boxes:
            box = box.split(',')
            box[1], box[2] = box[2], box[1]
            b = tuple(map(float, box[0:4]))
            bb = convert((w, h), b)
            if bb is None:
                print(img_path)
                continue
            if ignore_class:
                box[4] = '0'
            out_file.write(box[4] + " " + " ".join([str(a) for a in bb]) + '\n')
        out_file.close()


if __name__ == '__main__':
    sets = ['elevator_20180601_convert_train.txt', 'elevator_20181230_convert_train.txt',
            'elevator_20181231_convert_train.txt', 'elevator_20190106_convert_train.txt']
    for file in sets:
        convert_jiang_txt(file, ignore_class=True)
