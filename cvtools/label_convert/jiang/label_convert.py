# -*- encoding:utf-8 -*-
# @Time    : 2018/10/24 22:02
# @Author  : gfjiang
# @Site    : 
# @File    : label_convert.py.py
# @Software: PyCharm Community Edition


import os
import sys
import numpy as np
from PIL import Image, ImageDraw


def convert_boxes_type(box):
    return ",".join([str(int(x)) for x in box[:4]] + [str(int(box[4]))])


# 递归文件夹下所有文件夹，得到文件路径列表
def get_images_list(root_dir="E:/DL/datasets/mini-vision/"):
    if not os.path.isdir(root_dir):
        return [root_dir]
    images_list = []
    for lists in os.listdir(root_dir):  # 相当于调用多个递归
        images_list += get_images_list(os.path.join(root_dir, lists))
    return images_list


class data:
    def __init__(self, d):
        self.path = d['img_path']
        self.n_boxs = d['number']
        self.bboxs = d['coordinates_with_class']


def brainwash_data_line_parser(line):
    if ":" not in line:
        img_path, _ = line.split(";")       # 没有框
        src_path = img_path.replace('"', '')
        num_coordinates = 0
        coordinates = []
        return {'img_path': src_path, 'number': num_coordinates, 'coordinates_with_class': coordinates}
    else:
        img_path, bbox_coordinates_raw = line.split(":")
        src_path = img_path.replace('"', '')
        bbox_coordinates_raw = bbox_coordinates_raw.replace("(", "")
        bbox_coordinates_raw = bbox_coordinates_raw.replace("),", ",")
        bbox_coordinates_raw = bbox_coordinates_raw.replace(").", "")
        bbox_coordinates_raw = bbox_coordinates_raw.replace(");", "")
        bbox_coordinates_str = bbox_coordinates_raw.split(", ")
        coordinates_list = [float(i) for i in bbox_coordinates_str]
        num_coordinates = len(coordinates_list) / 4
        coordinates = np.zeros(shape=(int(num_coordinates), 5), dtype=np.float)
        entry_idx = 0
        for i in range(0, len(coordinates_list), 4):
            coord = coordinates_list[i:i + 4]
            coord = [coord[0], coord[1], coord[2], coord[3], 0]
            coordinates[entry_idx, :] = coord
            entry_idx += 1

    return {'img_path': src_path, 'number': num_coordinates, 'coordinates_with_class': coordinates}


def arc_data_line_parser(line):
    attributes_dict = {'img_path': '/root/data/'}
    if ":" not in line:
        img_path, _ = line.strip().split(";")       # 没有框
        src_path = img_path.replace('"', '')
        num_coordinates = 0
        coordinates = []
        return {'img_path': src_path, 'number': num_coordinates, 'coordinates_with_class': coordinates}
    else:
        line = line.strip().replace("{", "")
        line = line.replace("}", "")
        attributes = line.split('","')
        for index, attribute in enumerate(attributes):
            attribute_split = attribute.strip().split('":"')
            if len(attribute_split) == 2:
                attributes_dict[attribute_split[0].replace('"', "")] = attribute_split[1].replace('"', "")
        if 'rect' in attributes_dict:
            attributes_dict['rect'] = [int(i) for i in attributes_dict['rect'].split(',')]
            attributes_dict['rect'].append(attributes_dict['gender'])   # append函数无返回值，原位修改
        if 'faceRect' in attributes_dict:
            attributes_dict['faceRect'] = [int(i) for i in attributes_dict['faceRect'].split(',')]
            attributes_dict['faceRect'].append(0)   # 此数据无性别标签，忽略类别，给与默认值0
    return attributes_dict


def mini_vision_data_line_parser(line):
    line = line.strip().split(' ')
    src_path = line[0]
    if len(line) == 1:      # 没有框
        num_coordinates = 0
        coordinates = []
        return {'img_path': src_path, 'number': num_coordinates, 'coordinates_with_class': coordinates}
    else:
        bbox_coordinates_str = line[1:]
        coordinates_list = [float(i) for i in bbox_coordinates_str]
        num_coordinates = len(coordinates_list) / 4
        coordinates = np.zeros(shape=(int(num_coordinates), 5), dtype=np.float)
        entry_idx = 0
        for i in range(0, len(coordinates_list), 4):
            coord = coordinates_list[i:i + 4]
            coord = [coord[0], coord[1], coord[2], coord[3], 0]
            coordinates[entry_idx, :] = coord
            entry_idx += 1
    return {'img_path': src_path, 'number': num_coordinates, 'coordinates_with_class': coordinates}


# 一个txt只对一张图片描述
def get_phase_data(txt, parser_fun):
    d = {'img_path': txt.replace('.txt', '.jpg'), 'number': 0, 'coordinates_with_class': []}
    with open(txt, 'r') as fp:
        for line in fp.readlines():
            temp = parser_fun(line)
            if 'rect' in temp:
                d['coordinates_with_class'].append(temp['rect'])
                d['number'] += 1
            if 'faceRect' in temp:
                d['coordinates_with_class'].append(temp['faceRect'])
                d['number'] += 1
    return data(d)


# 一个txt对多张图片描述
def get_phase_data_list(txt, parser_fun):
    """Return a list of data object.
    data object: path, n_boxs, bboxs

    Args: data_list_path: list of filenames and groundtruth information available
            in the brainwash dataset.
    Returns: A list of data objects. Where the length of the list is equal
            to the number of images contained in the split of the dataset.
    """
    data_list = []
    # path_list = get_images_list()
    with open(txt, 'r') as fp:
        for line in fp.readlines():
            d = parser_fun(line)
            # for path in path_list:
            #     if path.find(d['img_path']) >= 0:
            #         d['img_path'] = path
            if d['number'] != 0:
                d_object = data(d)
                data_list.append(d_object)
    return data_list


def convert_label_to_txt(data_root_path, label_list, txt_path_name, parser_fun):
    # test_data_list_path = os.path.join('E:/DL/datasets/brainwash', 'brainwash_test.idl')
    data_list = []
    for txt in label_list:
        data_list += get_phase_data_list(data_root_path+txt, parser_fun)
    # im = Image.open(test_data_list[100].path)
    # draw = ImageDraw.Draw(im)
    # for box in test_data_list[100].bboxs:
    #     draw.rectangle(list(box))
    # im.save("test.png")
    with open(txt_path_name, 'w') as fp:
        for line in data_list:
            boxes = ''
            for box in line.bboxs:
                boxes += (' ' + (convert_boxes_type(box)))
            fp.write(line.path+boxes+'\n')
    # print(test_data_list)


def convert_arc_label_to_txt(data_root_path, txt_path_name, parser_fun):
    from src.utils import get_files_list
    label_list = get_files_list(data_root_path, file_type='txt')
    data_list = []
    for txt in label_list:
        data_list.append(get_phase_data(txt, parser_fun))
    with open(txt_path_name, 'w') as fp:
        for line in data_list:
            boxes = ''
            for box in line.bboxs:
                boxes += (' ' + (convert_boxes_type(box)))
            fp.write(line.path+boxes+'\n')


def draw_rect_test_label(src, dst, first=sys.maxsize):
    colour = ['blue', 'red', 'green', 'white']
    if not os.path.exists(dst):
        os.mkdir(dst)
    with open(src, 'r') as fp:
        for count, line in enumerate(fp.readlines()):
            if count < first:
                line = line.strip().split()
                im = Image.open(line[0])
                draw = ImageDraw.Draw(im)
                boxes = line[1:]
                for box in boxes:
                    bbox_coordinates_str = box.split(',')
                    coordinates = [float(i) for i in bbox_coordinates_str][0:4]
                    draw.rectangle(coordinates, None, colour[int(bbox_coordinates_str[4])])
                im.save(dst+line[0].split('/')[-1])


if __name__ == '__main__':
    # data_root_path = '/home/xuhao/gfjiang/ml/keras-yolo3/VOCdevkit/MY_VOC/JPEGImages/brainwash/'
    # # data_root_path = 'E:/DL/datasets/brainwash/'
    # label_list = ['brainwash_test.idl']
    # convert_label_to_txt(data_root_path, label_list, '../datasets/brainwash_test.txt',
    #                      brainwash_data_line_parser)

    # get_images_list()
    # test_label('../datasets/train/UCSD_train.txt',
    #            '../results/temp/UCSD_train/')

    data_root_path = '/root/data/elevator/20180601/'
    convert_arc_label_to_txt(data_root_path, '../datasets/elevator_20180601.txt', arc_data_line_parser)
    draw_rect_test_label('../datasets/elevator_20180601.txt', '../results/temp/elevator/')
