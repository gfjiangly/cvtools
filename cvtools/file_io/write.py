# -*- encoding:utf-8 -*-
# @Time    : 2019/11/15 10:41
# @Author  : gfjiang
# @Site    : 
# @File    : write.py
# @Software: PyCharm

import _pickle as pickle
import json

import cvtools


# 以json格式保存数据到disk
def save_json(data, to_file='data.json'):
    # save json format results to disk
    cvtools.makedirs(to_file)
    with open(to_file, 'w') as f:
        json.dump(data, f)  # using indent=4 show more friendly
    print('!save {} finished'.format(to_file))


def save_pkl(data, to_file: str):
    assert to_file.endswith('.pkl')
    # 默认 using protocol 0. 负数表示最高协议
    pickle.dump(data, open(to_file, 'wb'))
