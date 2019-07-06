# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/6/30 15:32
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import json

from utils.file import read_files_to_list


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
