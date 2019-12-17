# -*- encoding:utf-8 -*-
# @Time    : 2019/12/12 21:12
# @Author  : jiang.g.f
# @File    : read.py
# @Software: PyCharm

import json
import os.path as osp


# 加载json文件
def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


# 按行读取文件内容，支持中文
def readlines(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            return f.readlines()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='gbk') as f:
            return f.readlines()


# 读入单个文件输出list，支持中文
def read_file_to_list(file):
    images_list = []
    try:
        with open(file, 'r', encoding='utf-8') as f:
            images_list += [line.strip('\n') for line in f]
    except UnicodeDecodeError:
        # 有的中文用utf-8可以解码成功，有的不可以，看写入时用的什么编码
        with open(file, 'r', encoding='gbk') as f:
            images_list += [line.strip('\n') for line in f]
    return images_list


# # 读入单个或多个文件合成一个list输出
# def read_files_to_list(root, files):
#     if isinstance(files, str):
#         files = [files]
#     images_list = []
#     for file in files:
#         images_list += read_file_to_list(root+file)
#         # with open(root+file, 'r') as f:
#     return images_list


# 读入单个或多个文件合成一个list输出，支持中文
def read_files_to_list(files, root=''):
    """此函数设计是一个教训，只有必要的参数才能设计成位置参数，其它参数为关键字参数"""
    if isinstance(files, str):
        files = [files]
    images_list = []
    for file in files:
        images_list += read_file_to_list(osp.join(root, file))
    return images_list


# 读取txt到字典中，每行以字符':'分割key和value
def read_key_value(file):
    """支持注释，支持中文"""
    return_dict = {}
    lines = readlines(file)
    for line in lines:
        line = line.strip().split(':')
        if line[0][0] == '#':
            continue
        key = line[0].strip()
        value = line[1].strip()
        return_dict[key] = value
    return return_dict
