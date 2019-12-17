# -*- encoding:utf-8 -*-
# @Time    : 2019/12/17 21:49
# @Author  : jiang.g.f
# @File    : test_voc_to_darknet.py
# @Software: PyCharm
import cvtools
import os.path as osp

current_path = osp.dirname(__file__)


def test_voc_to_darknet():
    voc_to_darknet = cvtools.VOC2DarkNet(
        current_path + '/data/VOC',
        mode='trainval',
        use_xml_name=True,
        read_test=True
    )
    voc_to_darknet.convert(save_root=current_path + '/out/darknet')
