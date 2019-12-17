# -*- encoding:utf-8 -*-
# @Time    : 2019/12/12 21:12
# @Author  : jiang.g.f
# @File    : read.py
# @Software: PyCharm

import _pickle as pickle
import json
import os.path as osp


# 加载json文件
def load_json(file):
    """加载json文件

    Args:
        file: 包含路径的文件名

    Returns:

    """
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def load_pkl(file):
    """加载pickle序列化对象

    Args:
        file: 包含路径的文件名

    Returns:
        unpickle object

    Raises:
        UnpicklingError

    """
    return pickle.load(open(file, 'rb'))


# 按行读取文件内容，支持中文
def readlines(file):
    """按行读取str到list

    Args:
        file: 包含路径的文件名

    Returns:

    """
    try:
        with open(file, 'r', encoding='utf-8') as f:
            return f.readlines()
    except UnicodeDecodeError:
        with open(file, 'r', encoding='gbk') as f:
            return f.readlines()


def read_file_to_list(file):
    """读入单个文件输出list，支持中文

    Args:
        file: 包含路径的文件名

    Returns:
        所有文件内容放在list中返回

    """
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


def read_files_to_list(files, root=''):
    """读入单个或多个文件合成一个list输出，支持中文

    此函数设计是一个教训，只有必要的参数才能设计成位置参数，其它参数为关键字参数

    Args:
        files (str):  文件名
        root (root):  可选，文件名路径。如果指定files不可加路径
    """
    if isinstance(files, str):
        files = [files]
    images_list = []
    for file in files:
        images_list += read_file_to_list(osp.join(root, file))
    return images_list


# 读取txt到字典中，每行以字符':'分割key和value
def read_key_value(file):
    """支持注释，支持中文

    Args:
        file (str): 包含路径的文件名
    """
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
