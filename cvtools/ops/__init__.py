# -*- encoding:utf-8 -*-
# @Time    : 2019/12/25 21:53
# @Author  : jiang.g.f
# @File    : __init__.py
# @Software: PyCharm

from .nms import py_cpu_nms, soft_nms

__all__ = [
    'py_cpu_nms', 'soft_nms'
]
