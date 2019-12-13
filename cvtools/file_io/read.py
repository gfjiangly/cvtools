# -*- encoding:utf-8 -*-
# @Time    : 2019/12/12 21:12
# @Author  : jiang.g.f
# @File    : read.py
# @Software: PyCharm

import json


# 加载json文件
def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data
