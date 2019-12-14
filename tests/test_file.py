# -*- encoding:utf-8 -*-
# @Time    : 2019/7/6 22:15
# @Author  : gfjiang
# @Site    : 
# @File    : test_file.py
# @Software: PyCharm

import cvtools
import os
import os.path as osp
import shutil

current_path = osp.dirname(__file__)


def test_makedirs():
    shutil.rmtree(current_path + '/out')
    os.makedirs(current_path + '/out')  # 不需要最后一个字符是路径分隔符
    assert not cvtools.makedirs('')
    assert not cvtools.makedirs(None)
    assert not cvtools.makedirs(current_path + '/out/dir')
    assert cvtools.makedirs(current_path + '/out/dir/')
    assert cvtools.makedirs(current_path + '/out/dir/test1/test.txt')
    assert cvtools.makedirs(current_path + '/out/dir\\test2/test.txt')
    assert cvtools.makedirs(current_path + '/out/dir/test3/test.txt')
    assert cvtools.makedirs(current_path + '/out/dir/test3/../test4/test.txt')
