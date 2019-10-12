# -*- encoding:utf-8 -*-
# @Time    : 2019/7/6 22:15
# @Author  : gfjiang
# @Site    : 
# @File    : test_file.py
# @Software: PyCharm

import cvtools


def test_makedirs():
    assert not cvtools.makedirs('')
    assert not cvtools.makedirs(None)
    assert not cvtools.makedirs('test.txt')
    assert cvtools.makedirs('test1/test.txt')
    assert cvtools.makedirs('test2\\test.txt')
    assert cvtools.makedirs('test3/test3/test.txt')
    assert not cvtools.makedirs('../tests/test.txt')

