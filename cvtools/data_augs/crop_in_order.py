# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/7/9 13:05
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import cv2.cv2 as cv
import copy
import os.path as osp
import numpy as np
import random
from collections import defaultdict
from shapely.geometry import Polygon

import cvtools
from cvtools.utils.boxes import x1y1wh_to_x1y1x2y2x3y3x4y4


def sliding_crop(img, crop_w, crop_h, overlap=0.1):
    h, w, c = img.shape
    crop_imgs = []
    starts = []
    # fix crop bug!
    y_stop = False
    for sy in range(0, h, int(crop_h * (1. - overlap))):
        x_stop = False
        for sx in range(0, w, int(crop_w * (1. - overlap))):
            ex = sx + crop_w
            if ex > w:
                ex = w
                sx = w - crop_w
                if sx < 0:
                    sx = 0
                x_stop = True
            ey = sy + crop_h
            if ey > h:
                ey = h
                sy = h - crop_h
                if sy < 0:
                    sy = 0
                y_stop = True
            # sy, ey, sx, ex = int(sy), int(ey), int(sx), int(ex)
            crop_imgs.append(img[sy:ey, sx:ex])
            starts.append((sx, sy))
            if x_stop:
                break
        if y_stop:
            break
    return crop_imgs, starts


def crop_for_small_intensive(img, ann, small_prop=0.5, max_objs=100):
    # 1 是否小目标数量超过50%
    areas = []
    for obj in ann:
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
    if len(ann) > max_objs:
        h, w, _ = img.shape
        size = random.randint(400, 800)
        if h < 1333 or w < 1333:
            size = 1333
        return sliding_crop(img, size, size, overlap=0.1)
    return None


def crop_for_large_img(img, ann=None, th=1333):
    # 2 图片宽或高超过1333
    h, w, _ = img.shape
    if h > th or w > th:
        size = random.randint(800, 1333)
        return sliding_crop(img, size, size, overlap=0.1)
    return None


def crop_for_protected(img, ann):
    # 3 滑窗裁剪中被破坏的大目标
    def cal_edge(objs):
        obbs = np.array([obj['segmentation'][0]
                         for obj in objs]).reshape(-1, 2)
        x1, y1 = np.min(obbs[..., 0]), np.min(obbs[..., 1])
        x2, y2 = np.max(obbs[..., 0]), np.max(obbs[..., 1])
        h, w, _ = img.shape
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        x2 = w - 1 if x2 > w else x2
        y2 = h - 1 if y2 > h else y2
        new_x1 = random.randint(0, x1)
        new_y1 = random.randint(0, y1)
        new_x2 = random.randint(x2, w)
        new_y2 = random.randint(y2, h)
        return new_x1, new_y1, new_x2, new_y2
    crop_imgs, starts = [], []
    x1, y1, x2, y2 = cal_edge(ann)
    crop_imgs.append(img[y1:y2, x1:x2])
    starts.append((x1, y1))
    if len(ann) > 1:
        crops = [(cal_edge([obj])) for obj in ann
                 if obj['area'] > 96 * 96]
        crop_imgs += [img[crop[1]:crop[3], crop[0]:crop[2]]
                      for crop in crops]
        starts += [crop[:2] for crop in crops]
    return crop_imgs, starts


def polygen_overlap(polygen1, polygen2, mode='iou', shape=(1333, 1333)):
    assert mode in ['iou', 'iof']
    polygen1 = np.array(polygen1, dtype=np.int32).reshape(-1, 2)
    polygen2 = np.array(polygen2, dtype=np.int32).reshape(-1, 2)
    im1 = np.zeros(shape, dtype="uint8")
    im2 = np.zeros(shape, dtype="uint8")
    mask1 = cv.fillPoly(im1, polygen1, 255)
    mask2 = cv.fillPoly(im2, polygen2, 255)
    masked_and = cv.bitwise_and(mask1, mask2, mask=im1)
    masked_or = cv.bitwise_or(mask1, mask2)
    # or_area = np.sum(np.float32(np.greater(masked_or, 0)))
    and_area = np.sum(np.float32(np.greater(masked_and, 0)))
    if mode == 'iou':
        union = np.sum(np.float32(np.greater(masked_or, 0)))
    else:
        union = np.sum(np.float32(mask1))
    iou = and_area / union
    return iou


class CropInOder(object):
    """为DOTA数据集设计的裁剪方案：
    需要滑窗裁剪的条件(顺序重要)：
        1 小目标数量超过50%，crop=400x400（实验确定）；or
        1 目标数量超过100（结合mmdet单张最大检测数量制定）；or
        2 图片宽或高超过1333（实验确定）；
    需要保护裁剪的条件：
        3 滑窗裁剪中被破坏的大目标；
    测试集裁剪方案：
        1 滑窗裁剪；
        2 不剪裁
    """
    def __init__(self,
                 img_prefix=None,
                 ann_file=None,
                 iof_th=0.7,
                 small_prop=0.5,
                 max_objs=100,
                 overlap=0.1,
                 size_th=1333):
        assert 1.0 >= iof_th >= 0.5
        self.img_prefix = img_prefix
        if ann_file is not None:
            self.ann_file = ann_file
            self.coco_dataset = cvtools.load_json(ann_file)
            self.COCO = cvtools.COCO(ann_file)
            self.catToAnns = defaultdict(list)
            if 'annotations' in self.coco_dataset:
                for ann in self.coco_dataset['annotations']:
                    self.catToAnns[ann['category_id']].append(ann)
        self.iof_th = iof_th
        self.small_prop = small_prop
        self.max_objs = max_objs
        self.overlap = overlap
        self.size_th = size_th
        self.imgs = list()
        self.img_to_objs = list()

    def cal_iof(self, gt_boxes, crop_imgs, starts):
        """iof: 行是gt_boxes，列是crop_imgs"""
        gt_boxes = cvtools.x1y1wh_to_x1y1x2y2(np.array(gt_boxes))
        img_boxes = []
        for i, crop_img in enumerate(crop_imgs):
            sx, sy = starts[i]
            h, w, _ = crop_img.shape
            img_box = cvtools.x1y1wh_to_x1y1x2y2(
                np.array([[sx, sy, w, h]]))
            img_boxes.append(img_box)
        img_boxes = np.array(img_boxes).reshape(-1, 4)
        iof = cvtools.bbox_overlaps(gt_boxes, img_boxes, mode='iof')
        return iof

    def crop(self, img, ann):
        # 1 是否小目标数量超过50%, 是否目标数量超过100
        result = crop_for_small_intensive(
            img, ann, small_prop=self.small_prop,
            max_objs=self.max_objs)
        # 2 图片宽或高超过1333
        if result is None:
            result = crop_for_large_img(img, ann)
        if result is None:
            return [img], [(0, 0)]
        crop_imgs, starts = result
        gt_boxes = [obj['bbox'] for obj in ann]
        # 3.1 iof阈值筛选被破坏的
        iof = self.cal_iof(gt_boxes, crop_imgs, starts)
        ids_in = set(np.where(iof >= self.iof_th)[0])
        out = set([i for i in range(len(ann))]) - ids_in
        # 3.2 ann's box宽高筛选被破坏的
        for ids in ids_in:
            max_len = max(ann[ids]['bbox'][2:4])
            if max_len > 800:
                out.add(ids)
        ann_croped = [ann[i] for i in out]
        if len(ann_croped) > 0:
            crops_protected, starts_protected = \
                crop_for_protected(img, ann_croped)
            crop_imgs += crops_protected
            starts += starts_protected
        return crop_imgs, starts

    # iof大于阈值的予以保留
    def match_img_objs(self, crop_imgs, starts, objs):
        reserved, edged = defaultdict(), defaultdict()
        boxes = [obj['bbox'] for obj in objs]
        if len(boxes) == 0:
            return None
        iof = self.cal_iof(boxes, crop_imgs, starts)
        for i, crop_img in enumerate(crop_imgs):
            sx, sy = starts[i]
            h, w, _ = crop_img.shape
            ex, ey = sx + w, sy + h
            index_reserved = set(np.where(iof[..., i] == 1.0)[0])
            objs_reserved = [objs[i] for i in index_reserved]
            index_edged = set(np.where((iof[..., i] > self.iof_th) &
                                       (iof[..., i] < 1.0))[0])
            objs_edged = [objs[i] for i in index_edged]
            if len(objs_reserved) > 0:
                reserved[(sx, sy, ex, ey)] = objs_reserved
            if len(objs_edged) > 0:
                edged[(sx, sy, ex, ey)] = objs_edged
        if len(edged) > 0:
            new_edged = self.deal_edged_boxes(edged)
            for img_coor in reserved.keys():
                if img_coor in new_edged.keys():
                    reserved[img_coor] += new_edged[img_coor]
        self.img_to_objs.append(reserved)

    # 处理位于剪裁图像边缘的框
    def deal_edged_boxes(self, cropToObjs):
        newCropToObjs = defaultdict()
        for img_box, objs in cropToObjs.items():
            # min_x, max_x = img_box[0], img_box[2]
            # min_y, max_y = img_box[1], img_box[3]
            img_x1y1wh = cvtools.x1y1x2y2_to_x1y1wh(img_box)
            img_poly = np.array(
                x1y1wh_to_x1y1x2y2x3y3x4y4(img_x1y1wh)
            ).reshape(-1, 2)
            img_poly = Polygon(img_poly).convex_hull
            new_objs = []
            for obj in objs:
                new_obj = copy.deepcopy(obj)
                # box_x1y1x2y2 = cvtools.x1y1wh_to_x1y1x2y2(obj['bbox'])
                # bbox = np.array(box_x1y1x2y2).reshape(-1, 2)
                # bbox[..., 0] = np.clip(bbox[..., 0], min_x, max_x-1)
                # bbox[..., 1] = np.clip(bbox[..., 1], min_y, max_y-1)
                # obj['bbox'] = cvtools.x1y1x2y2_to_x1y1wh(bbox.reshape(-1).tolist())
                segm = np.array(obj['segmentation']).reshape(-1, 2)
                segm_poly = Polygon(segm).convex_hull
                segm_poly = img_poly.intersection(segm_poly)
                # 注意不能原位修改obj，否则会影响到其它crop_img的obj
                new_obj['bbox'] = cvtools.x1y1x2y2_to_x1y1wh(segm_poly.bounds)
                # segm[..., 0] = np.clip(segm[..., 0], min_x, max_x)
                # segm[..., 1] = np.clip(segm[..., 1], min_y, max_y)
                a = np.array(list(segm_poly.exterior.coords)).reshape(-1, 2)
                segm_hull = cv.convexHull(a.astype(np.float32), clockwise=False)
                new_obj['segmentation'] = [np.array(segm_hull).reshape(-1).tolist()]
                new_objs.append(new_obj)
            newCropToObjs[img_box] = new_objs
        return newCropToObjs

    def crop_with_label(self, save_root):
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        if cvtools._DEBUG:
            roidb = roidb[:10]
        for i, entry in enumerate(roidb):
            image_name = entry['file_name']
            image_file = osp.join(self.img_prefix, image_name)
            ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
            ann = self.COCO.loadAnns(ann_ids)
            if len(ann) == 0:
                print('{} ann is None.'.format(image_name))
                continue

            # read image
            img = cvtools.imread(image_file)
            if img is None:
                print('{} is None.'.format(image_file))
                continue

            print('crop image %d of %d: %s' %
                  (i, len(roidb), image_name))

            # crop image
            crop_imgs, starts = self.crop(img, ann)

            # handling the box at the edge of the cropped image
            self.imgs.append(image_name)
            self.match_img_objs(crop_imgs, starts, ann)

        # save crop results
        self.save_crop_labeltxt(save_root)

    # Note： 原始objs不能修改，否则可能影响其它objs
    def save_crop_labeltxt(self, save_root):
        cvtools.makedirs(osp.join(save_root, 'images'))
        cvtools.makedirs(osp.join(save_root, 'labelTxt+crop'))
        for i, image_name in enumerate(self.imgs):
            crops = []
            img_name_no_suffix, img_suffix = osp.splitext(image_name)
            imgToObjs = self.img_to_objs[i]
            for crop_i, img_coor in enumerate(imgToObjs):
                crop_objs = imgToObjs[img_coor]
                if len(crop_objs) == 0:
                    continue
                # write crop results to txt
                txt_name = '_'.join(
                    [img_name_no_suffix] + [str(crop_i)] +
                    list(map(str, img_coor))) + '.txt'
                txt_content = ''
                for crop_obj in crop_objs:
                    if len(crop_obj) == 0:
                        continue
                    polygen = np.array(crop_obj['segmentation'][0]).reshape(-1, 2)
                    polygen = polygen - np.array(img_coor[:2]).reshape(-1, 2)
                    line = list(map(str, polygen.reshape(-1)))
                    cat = self.COCO.cats[crop_obj['category_id']]['name']
                    diffcult = str(crop_obj['difficult'])
                    line.append(cat)
                    line.append(diffcult)
                    txt_content += ' '.join(line) + '\n'
                cvtools.strwrite(
                    txt_content, osp.join(save_root, 'labelTxt+crop', txt_name))
            if len(crops) > 0:
                draw_img = cvtools.draw_boxes_texts(
                    img, crops, line_width=3, box_format='x1y1x2y2')
                cvtools.imwrite(
                    draw_img, osp.join(save_root, 'images', img_name_no_suffix+'.jpg'))

    def vis_box(self, save_root, has_box=True, has_segm=True, has_crop=True):
        for i, image_name in enumerate(self.imgs):
            print('Visualize image %d of %d: %s' %
                  (i, len(self.imgs), image_name))
            # read image
            image_file = osp.join(self.img_prefix, image_name)
            img = cvtools.imread(image_file)
            if img is None:
                print('{} is None.'.format(image_file))
                continue
            imgToObjs = self.img_to_objs[i]
            for crop_i, img_coor in enumerate(imgToObjs):
                if has_crop:
                    img = cvtools.draw_boxes_texts(
                        img, img_coor, colors='random', line_width=2)
                objs = imgToObjs[img_coor]
                if has_box:
                    boxes = [obj['bbox'] for obj in objs]
                    img = cvtools.draw_boxes_texts(
                        img, boxes, colors=[(255, 0, 0)]*len(boxes), box_format='x1y1wh')
                if has_segm:
                    segms = [obj['segmentation'][0] for obj in objs]
                    img = cvtools.draw_boxes_texts(
                        img, segms, box_format='polygen')
            to_file = osp.join(save_root, 'images', image_name)
            cvtools.imwrite(img, to_file)

    def crop_for_test(self, w, h, save=None):
        imgs = cvtools.get_images_list(self.img_prefix)
        if cvtools._DEBUG:
            imgs = imgs[:10]
        self.test_dataset = defaultdict(list)
        for i, image_file in enumerate(imgs):
            image_name = osp.basename(image_file)
            img = cvtools.imread(image_file)  # support chinese
            if img is None:
                print('{} is None.'.format(image_file))
                continue
            print('crop image %d of %d: %s' %
                  (i, len(imgs), image_name))

            crop_imgs, starts = sliding_crop(img, w, h)

            for crop_img, start in zip(crop_imgs, starts):
                crop_rect = start[0], start[1], \
                            start[0]+crop_img.shape[1], start[1]+crop_img.shape[0]
                self.test_dataset[image_name].append(crop_rect)
        if save is not None:
            cvtools.save_json(self.test_dataset, save)
        return self.test_dataset


if __name__ == '__main__':
    # only test crop method
    img_file = 'D:/data/DOTA/val/images/P0003.png'
    img = cv.imread(img_file)
    imgs, starts = sliding_crop(img, 800, 800)
