# -*- encoding:utf-8 -*-
# @Time    : 2019/11/15 10:41
# @Author  : gfjiang
# @Site    : 
# @File    : write.py
# @Software: PyCharm

import _pickle as pickle
import json

import cvtools


def dump_json(data, to_file='data.json'):
    """写json文件

    Args:
        data: 待保存成json格式的对象
        to_file: 保存的文件名
    """
    # save json format results to disk
    cvtools.makedirs(to_file)
    with open(to_file, 'w') as f:
        json.dump(data, f)  # using indent=4 show more friendly
    print('!save {} finished'.format(to_file))


def dump_pkl(data, to_file='data.pkl'):
    """使用pickle序列化对象

    Args:
        data: 待序列化对象
        to_file: 保存的文件名
    """
    cvtools.makedirs(to_file)
    # 默认 using protocol 0. 负数表示最高协议
    with open(to_file, 'wb') as f:
        pickle.dump(data, f, -1)


def write_str(data, to_file):
    """写字符串到文件

    Args:
        data (str): str对象
        to_file (str): 保存的文件名
    """
    cvtools.makedirs(to_file)
    with open(to_file, 'w') as f:
        f.write(data)


def write_list_to_file(data, dst, line_break=True):
    """保存list到文件

    Args:
        data (list): list中元素只能是基本类型
        dst (str): 保存的文件名
        line_break: 是否加换行

    Returns:

    """
    images_list = []
    cvtools.makedirs(dst)
    with open(dst, 'w') as f:
        for line in data:
            if line_break:
                line += '\n'
            f.write(line)
    return images_list


def write_key_value(data, to_file):
    """写字典到文件中（非序列化）

    每行以字符':'分割key和value

    Args:
        data (dict): dict中元素只能是基本类型
        to_file: 保存的文件名

    Returns:

    """
    if not isinstance(data, dict):
        return
    cvtools.makedirs(to_file)
    with open(to_file, 'w', encoding='utf8') as f:
        for key, value in data.items():
            f.write('{}: {}\n'.format(key, value))
