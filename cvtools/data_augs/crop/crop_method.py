# -*- encoding:utf-8 -*-
# @Time    : 2019/11/20 14:53
# @Author  : gfjiang
# @Site    : 
# @File    : crop_method.py
# @Software: PyCharm
"""提供大图裁剪方法，所有裁剪类需继承CropMethod，必须实现crop方法，
可选实现match_anns方法"""
import numpy as np
import random
import cv2.cv2 as cv

from cvtools.data_augs.crop.crop_abc import CropMethod


def sliding_crop(img, crop_w, crop_h, overlap=0.1):
    h, w, c = img.shape
    img_boxes = []
    # fix crop bug!
    y_stop = False
    for sy in range(0, h, int(crop_h * (1. - overlap))):
        x_stop = False
        for sx in range(0, w, int(crop_w * (1. - overlap))):
            ex = sx + crop_w - 1
            if ex >= w:
                ex = w - 1
                sx = w - crop_w
                if sx < 0:
                    sx = 0
                x_stop = True
            ey = sy + crop_h - 1
            if ey >= h:
                ey = h - 1
                sy = h - crop_h
                if sy < 0:
                    sy = 0
                y_stop = True
            # sy, ey, sx, ex = int(sy), int(ey), int(sx), int(ex)
            img_boxes.append([sx, sy, ex, ey])
            if x_stop:
                break
        if y_stop:
            break
    return img_boxes


def crop_for_small_intensive(img, anns, small_prop=0.5, max_objs=100):
    # 1 是否小目标数量超过50%
    areas = []
    for obj in anns:
        # prepare for Polygon class input
        a = np.array(obj['segmentation'][0]).reshape(4, 2)
        # 注意坐标系与笛卡尔坐标系方向相反，
        # 所以clockwise=False表示为真实世界的顺时针输出凸包，
        a_hull = cv.convexHull(a.astype(np.float32), clockwise=False)
        areas.append(cv.contourArea(a_hull))
    small_areas = [area for area in areas if area <= 32 * 32]
    if len(small_areas) > small_prop * len(areas):
        size = random.randint(200, 600)
        return sliding_crop(img, size, size, overlap=0.1)
    # 1 是否目标数量超过100
    if len(anns) > max_objs:
        h, w, _ = img.shape
        size = random.randint(400, 800)
        if h < 1333 or w < 1333:
            size = 1333
        return sliding_crop(img, size, size, overlap=0.1)
    return []


def crop_for_large_img(img, overlap=0.1, size_th=1333):
    # 2 图片宽或高超过size_th
    h, w, _ = img.shape
    if h > size_th or w > size_th:
        size = random.randint(400, size_th)
        try:
            return sliding_crop(img, size, size, overlap=overlap)
        except TypeError:
            return []
    return []


def crop_for_protected(img, anns, size_th=1024):
    # 3 滑窗裁剪中被破坏的大目标
    def cal_edge(objs, size):
        """求所有segmentation的外接矩形，也可以求所有bbox的外接矩形"""
        obbs = np.array([obj['segmentation'][0]
                         for obj in objs]).reshape(-1, 2)
        x1, y1 = np.min(obbs[..., 0]), np.min(obbs[..., 1])
        x2, y2 = np.max(obbs[..., 0]), np.max(obbs[..., 1])
        if size < (x2 - x1 + 1) * 2: size = (x2 - x1 + 1) * 2
        x_tolerate = size - (x2 - x1 + 1)
        y_tolerate = size - (y2 - y1 + 1)
        if x_tolerate < 0 or y_tolerate < 0:
            return ()
        new_x1 = random.randint(x1 - x_tolerate // 2, x1)
        new_y1 = random.randint(y1 - y_tolerate // 2, y1)
        new_x2 = random.randint(x2, x2 + x_tolerate // 2)
        new_y2 = random.randint(y2, y2 + y_tolerate // 2)
        h, w, _ = img.shape
        new_x1 = 0 if new_x1 < 0 else new_x1
        new_y1 = 0 if new_y1 < 0 else new_y1
        new_x2 = w - 1 if new_x2 >= w else new_x2
        new_y2 = h - 1 if new_y2 >= h else new_y2
        return new_x1, new_y1, new_x2, new_y2
    img_boxes = []
    all_protect_crop = cal_edge(anns, size_th)
    if len(all_protect_crop) > 0:
        img_boxes.append(all_protect_crop)
    if len(anns) > 1:   # 为每个大实例做裁剪
        each_protect_crops = [cal_edge([obj], size_th) for obj in anns
                              if obj['bbox'][2] * obj['bbox'][3] > 96 * 96]
        img_boxes += [crop for crop in each_protect_crops if len(crop) > 0]
    return img_boxes


class CropImageInOrder(CropMethod):
    def __init__(self, crop_w=1024, crop_h=1024, overlap=0.2, iof_th=0.7,
                 size_th=1333):
        """

        Args:
            crop_w (int): 滑动裁剪的宽
            crop_h: 滑动裁剪的高
            overlap: 滑动裁剪时的重合率
            iof_th: 实例在图像内的大小占自己本身大小的比例的阈值，小于此大小将被过滤
            size_th: 宽或高超过此阈值，认为应该被crop
        """
        super().__init__()
        assert 1.0 >= iof_th >= 0.5
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.overlap = overlap
        self.iof_th = iof_th
        self.size_th = size_th
        self.img_boxes = []

    def crop(self, img, anns=None):
        self.img_boxes = sliding_crop(img, self.crop_w, self.crop_h,
                                      self.overlap)
        return self.img_boxes


class CropImageProtected(CropMethod):

    def __init__(self, iof_th=0.7, size_th=1024):
        super().__init__(iof_th)
        self.size_th = size_th
        self.img_boxes = []

    def crop(self, img, anns=None):
        self.img_boxes = crop_for_protected(img, anns, self.size_th)
        return self.img_boxes


class CropImageAdaptive(CropMethod):
    def __init__(self, overlap=0.1, iof_th=0.7, small_prop=0.5, max_objs=100,
                 size_th=1333, strict_size=True):
        """

        Args:
            iof_th: 实例在图像内的大小占自己本身大小的比例的阈值，小于此大小将被过滤
            small_prop: 一张图中小实例占实例总数比例
            max_objs: 一张图允许最大的实例数
            overlap: 顺序裁剪时的重合率
            size_th: 宽或高超过此阈值，认为应该被crop
        """
        super().__init__(iof_th)
        assert 1.0 >= iof_th >= 0.5
        self.iof_th = iof_th
        self.small_prop = small_prop
        self.max_objs = max_objs
        self.overlap = overlap
        self.size_th = size_th
        self.strict_size = strict_size
        self.img_boxes = []

    def crop(self, img, anns=None):
        """
        可能crop算法可能需要根据标签信息
        """
        assert anns is not None, "自适应crop必须提供标签信息"
        # 1 是否小目标比例超过small_prop,或,是否目标数量超过max_objs
        self.img_boxes = crop_for_small_intensive(
            img, anns, small_prop=self.small_prop,
            max_objs=self.max_objs)
        # 2 图片宽或高超过size_th
        if len(self.img_boxes) == 0:
            self.img_boxes = crop_for_large_img(img)
        if len(self.img_boxes) == 0:
            self.img_boxes = [(0, 0, img.shape[1] - 1, img.shape[0] - 1)]
            return  # 不需要裁剪
        if not self.strict_size:
            self.img_boxes.append((0, 0, img.shape[1] - 1, img.shape[0] - 1))
        # 以下操作不保证可以增加self.img_boxes元素
        gt_boxes = [obj['bbox'] for obj in anns]
        if len(gt_boxes) == 0:
            return  # 没有被破话的anns了
        # 3.1 iof阈值筛选被破坏的
        iof = self.cal_iof(gt_boxes)
        ids_in = set(np.where(iof >= self.iof_th)[0])
        out = set([i for i in range(len(anns))]) - ids_in
        # 3.2 ann's box宽高筛选被破坏的
        for ids in ids_in:
            max_len = max(anns[ids]['bbox'][2:4])
            if max_len > 96:
                out.add(ids)
        ann_croped = [anns[i] for i in out]
        # 同时满足3.1&&3.2才进行保护裁剪
        if len(ann_croped) > 0:
            self.img_boxes += crop_for_protected(img, ann_croped)
        return self.img_boxes
