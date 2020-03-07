# -*- encoding:utf-8 -*-
# @Time    : 2019/7/6 22:44
# @Author  : gfjiang
# @Site    : 
# @File    : test_timer.py
# @Software: PyCharm
import time

import cvtools


def test_timer_content(capsys):
    timer = cvtools.Timer()
    timer.tic()
    time.sleep(1)
    # 好像误差有点大
    print('{:.3f}'.format(timer.toc()))
    # out, _ = capsys.readouterr()
    assert abs(timer.toc() - 1) < 0.1


def test_get_time_str():
    print(cvtools.get_time_str())
    print(cvtools.get_time_str(form='%Y-%m-%d %H:%M:%S.%f'))
    print(cvtools.get_time_str(form='%Y%m%d_%H%M%S_%f'))

