# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/6/30 15:32
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import json
import numpy as np

from .file import read_files_to_list


# 读取虹软不完全格式化数据进list，测试通过
def read_arcsoft_txt_format(file):
    with open(file, 'r') as f:
        data = f.readlines()
    data_list = []
    try:
        for line in data:
            if len(line) > 1:   # filter '\n'
                data_list.append(json.loads(line.strip()))
    except json.decoder.JSONDecodeError:
        print('{} decode error!'.format(file))
    return data_list


def read_jiang_txt(file):
    data = read_files_to_list(file)
    data_list = []
    for line in data:
        if len(line) < 2:
            continue
        info_dict = {}
        line = line.strip().split()
        info_dict['file_name'] = line[0]
        info_dict['bboxs_ids'] = [bbox.split(',') for bbox in line[1:]]
        data_list.append(info_dict)
    return data_list


def read_yuncong_detect_file(file, num_class):
    dets_yolo = [[] for _ in range(num_class)]
    image_list = []
    with open(file, 'r') as f:
        line = f.readline().strip()
        image_list.append(line)
        last_line = 'image name'
        count = 0
        while True:
            if last_line is 'image name':
                line = f.readline().strip()
                count = int(line)
                last_line = 'object number'
            if last_line is 'object number':
                boxes_for_one_image = []
                for i in range(count):
                    line = f.readline().strip()
                    boxes_for_one_image.append(list(map(float, line.split())))
                boxes_for_one_image = np.array(boxes_for_one_image)
                for class_index in range(num_class):
                    boxes_for_one_image_one_class = boxes_for_one_image[
                        boxes_for_one_image[:, 4].copy().astype(np.int) == class_index][:, [0, 1, 2, 3, 5]]
                    dets_yolo[class_index].append(boxes_for_one_image_one_class)
                last_line = 'box'
            if last_line is 'box':
                line = f.readline().strip()
                if not line:
                    break
                image_list.append(line)
                last_line = 'image name'
    return dets_yolo, image_list
