# -*- encoding:utf-8 -*-
# @Time    : 2020/12/16 15:32
# @Author  : jiang.g.f
import os.path as osp

from cvtools.utils.file import splitpath


def add_prefix_filename_suffix(filename, prefix='', suffix=''):
    """给文件名添加前缀和后缀，不改变路径"""
    filepath, filename, extension = splitpath(filename)
    filename = prefix + filename + suffix
    return osp.join(filepath, filename+extension)


if __name__ == '__main__':
    file = '/home/jiang/code/data/DOTA/dota1_1024/train1024/images.jpg'
    new_file = add_prefix_filename_suffix(file, prefix='color_', suffix='_2')
    print(new_file)

    file2 = 'images.jpg'
    new_file = add_prefix_filename_suffix(file2, prefix='color_', suffix='_2')
    print(new_file)

    file3 = './images.jpg'
    new_file = add_prefix_filename_suffix(file3, prefix='color_', suffix='_2')
    print(new_file)
