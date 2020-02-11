# -*- encoding:utf-8 -*-
# @Time    : 2020/2/11 16:06
# @Author  : jiang.g.f
# @File    : model.py
# @Software: PyCharm


class Model(object):
    """Just as an interface, you have to implement specific model code"""

    def detect(self, img):
        raise NotImplementedError("detect is not implemented!")

    def prase_results(self, results):
        return results

    def draw(self, img, results):
        return img


model = Model()
