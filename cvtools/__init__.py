# -*- encoding:utf-8 -*-
# @Time    : 2019/7/6 22:17
# @Author  : gfjiang
# @Site    : 
# @File    : __init__.py
# @Software: PyCharm
"""
cvtools是主要用于计算机视觉领域的Python工具包。
在实现和训练CV模型过程，一些与核心无关的常用代码被剥离出，形成此库。
"""

from .cocotools import *
from .ops import *
from .utils import *
from .file_io import *
from .evaluation import *
from .label_convert import *
from .label_analysis import *
from .data_augs import *
from .version import __version__


_DEBUG = False
_NUM_DATA = 10
