# -*- encoding:utf-8 -*-
# @Time    : 2019/12/25 15:17
# @Author  : jiang.g.f
# @File    : merge_crop_dets.py
# @Software: PyCharm
import numpy as np
from collections import defaultdict
import os.path as osp

import cvtools
from cvtools.ops.nms import py_cpu_nms


class MergeCropDetResults(object):
    """
    将裁剪数据集检测结果转成原始数据集检测结果
    Args:
        gts (str or COCO): DOTA-COCO兼容格式, img_info中必须含crop字段
        results (str or dict): 子图检测结果，保存在文件中，
            results:
            {
                image_id: dets,  # image_id必须是anns中有效的id
                image_id: dets,
                ...
            }
            dets:
            {
                类别：[[位置坐标，得分], [...], ...],     # np.array
                类别: [[位置坐标，得分], [...], ...],
                ...
            }，
            位置坐标支持左上角右下角两个点表示和四个点的表示方式
        num_coors (int): 4表示左上角右下角box， 8表示四个点的box
    """
    def __init__(self, gts, results, num_coors=4):
        assert num_coors in (4, 8), "不支持的检测位置表示"
        self.coco = gts
        self.results = results
        if cvtools.is_str(gts):
            self.coco = cvtools.COCO(gts)
        if cvtools.is_str(results):
            self.results = cvtools.load_pkl(results)
        self.num_coors = num_coors
        self.img_to_results = {}    # merge返回结果保存

    def crop_bbox_map_back(self, bb, crop_start):
        """子图检测坐标转换为相对于整图的坐标"""
        bb_shape = bb.shape
        original_bb = bb.reshape(-1, 2) + np.array(crop_start).reshape(-1, 2)
        return original_bb.reshape(bb_shape)

    def match_part_img_results(self, results):
        """结合子图的结果，映射回原图，应用nms, 生成一张整图的结果"""
        img_to_results = defaultdict(list)
        for image_id, dets in results.items():
            img_info = self.coco.imgs[image_id]
            labels = cvtools.concat_list(
                [[j]*len(det) for j, det in dets.items()])    # label从0开始
            scores = cvtools.concat_list([det[:, self.num_coors]
                                          for det in dets.values()])
            bboxes = np.vstack([det[:, :self.num_coors] for det in dets.values()
                                if len(det) > 0])
            if 'crop' in img_info:
                bboxes = self.crop_bbox_map_back(bboxes, img_info['crop'][:2])
            assert len(bboxes) == len(labels)
            if len(labels) > 0:
                result = [bboxes, labels, scores]
                img_to_results[img_info['file_name']].append(result)
        return img_to_results

    def merge(self, nms_method=py_cpu_nms, nms_th=0.15):
        """

        Args:
            nms_method: nms方法必须与bbox表示方式对应，
                默认是左上角右下角表示的水平矩形框
            nms_th: nms阈值

        Returns: 合并后的检测结果
            {
                filename: [bboxes, labels, scores],
                filename: [bboxes, labels, scores],
                ...
            }
        """
        img_to_results = self.match_part_img_results(self.results)
        for filename, result in img_to_results.items():
            bboxes = np.vstack([bb[0] for bb in result]).astype(np.int)
            labels = np.hstack([bb[1] for bb in result])
            scores = np.hstack([bb[2] for bb in result])
            dets = np.hstack([bboxes, scores[:, np.newaxis]]).astype(np.float32)
            ids = nms_method(dets, nms_th)
            img_to_results[filename] = [bboxes[ids], labels[ids], scores[ids]]
        self.img_to_results = img_to_results
        return img_to_results

    def img_results_to_cat_results(self, img_results):
        cat_results = defaultdict(list)
        for filename in img_results:
            bboxes = img_results[filename][0]
            cats = img_results[filename][1]
            scores = img_results[filename][2]
            for ind, cat in enumerate(cats):
                cat_results[cat].append([filename, scores[ind], bboxes[ind]])
        return cat_results

    def save_dota_det_format(self, save_path):
        class_names = [cat['name'] for cat in self.coco.cats.values()]
        self.cat_results = self.img_results_to_cat_results(self.img_to_results)
        path = osp.join(save_path, 'Task1_{}.txt')
        for cat_id, result in self.cat_results.items():
            lines = []
            for filename, score, bbox in result:
                filename = osp.splitext(filename)[0]
                bbox = list(map(str, list(bbox)))
                score = str(round(score, 3))
                lines.append(' '.join([filename] + [score] + bbox))
            cvtools.write_list_to_file(
                lines, path.format(class_names[cat_id-1]))


def crop_bbox_map_back(bb, crop_start):
    """子图检测坐标转换为相对于整图的坐标"""
    bb_shape = bb.shape
    original_bb = bb.reshape(-1, 2) + np.array(crop_start).reshape(-1, 2)
    return original_bb.reshape(bb_shape)


class CroppedDets:
    """New! 将裁剪数据集检测结果转成原始数据集检测结果"""

    def __init__(self, cropped_dets, img_infos, num_coors=4):
        """

        Args:
            img_infos (str or COCO): DOTA-COCO兼容格式, img_info中须含crop字段
                仅需要images和categories字段，无须annotations
                "categories": [
                    {
                        "id": 1,
                        "name": "large-vehicle",
                        "supercategory": "large-vehicle"
                    },
                    ...,
                    {
                        "id": 15,
                        "name": "storage-tank",
                        "supercategory": "storage-tank"
                    }
                ],
                images": [
                    {
                        "file_name": "P0126.png",
                        "id": 1,
                        "width": 455,
                        "height": 387,
                        "crop": [0, 0, 454, 386]
                    },
                    ...,
                ]
            cropped_dets (str or dict): 子图检测结果，保存在文件中，
                results:
                {
                    image_id: dets,  # image_id必须是anns中有效的id
                    image_id: dets,
                    ...
                }
                dets:
                {
                    cls_id：[[位置坐标，得分], [...], ...],     # np.array
                    cls_id: [[位置坐标，得分], [...], ...],
                    ...
                }，
                位置坐标支持左上角右下角两个点表示和四个点的表示方式
            num_coors (int): 4表示左上角右下角box， 8表示四个点的box
        """
        assert num_coors in (4, 8), "不支持的检测位置表示"
        self.cropped_dets = cropped_dets
        self.img_infos = img_infos
        if cvtools.is_str(cropped_dets):
            self.cropped_dets = cvtools.load_pkl(cropped_dets)
        if cvtools.is_str(img_infos):
            self.img_infos = cvtools.COCO(img_infos)
        self.num_coors = num_coors
        self.img_to_dets = self.merge()

    def merge(self, nms_method=py_cpu_nms, nms_th=0.15):
        """

        Args:
            nms_method: nms方法必须与bbox表示方式对应，
                默认是左上角右下角表示的水平矩形框
            nms_th: nms阈值

        Returns: 合并后的检测结果
            {
                filename: [bboxes, labels, scores],
                filename: [bboxes, labels, scores],
                ...
            }
        """
        img_to_dets = dict()  # 保存merge后结果

        # 按img_name汇集dets，并转换到原图坐标
        tmp_img_dets = defaultdict(list)
        for image_id, dets in self.cropped_dets.items():
            img_info = self.img_infos.imgs[image_id]
            if 'crop' in img_info:
                start_point = img_info['crop'][:2]
                img_name = img_info['file_name']
            else:
                # 待测试
                filename, ext = osp.splitext(img_info['file_name'])
                crop_info = filename.split('_')[-4:]
                start_point = (int(crop_info[0]), int(crop_info[1]))
                img_name = filename[:-4] + ext
            cls_det = dict()  # 单张子图所有类别检测结果
            for cls_id, det in dets.items():
                if len(det) > 0:
                    bboxes = crop_bbox_map_back(det[:, :-1], start_point)
                    det = np.hstack([bboxes, det[:, [-1]]])
                cls_det[cls_id] = det
            # 用于容纳一张大图的所有子图检测结果
            tmp_img_dets[img_name].append(cls_det)

        # 按类别合并子图结果
        for img_name, original_dets in tmp_img_dets.items():
            """
            {
                img1_name: [  # 大图1
                    dets,     # 子图1结果，dict
                    dets,     # 子图2结果，dict
                    ...,
                ]，
                img2_name: [  # 大图2
                    dets,     # 子图1结果，dict
                    dets,     # 子图2结果，dict
                    ...,
                ]
            }
            """
            # original_dets（List）：单张大图中所有子图所有类别检测结果
            original_dets_by_cls = dict()  # 按类别存放original_dets
            for sub_dets in original_dets:
                for cls, det in sub_dets.items():
                    if cls not in original_dets_by_cls:
                        original_dets_by_cls[cls] = [det]
                    else:
                        original_dets_by_cls[cls].append(det)
            for cls, dets in original_dets_by_cls.items():
                original_dets_by_cls[cls] = np.vstack(dets)
            img_to_dets[img_name] = original_dets_by_cls

        # 同一类在大图上做NMS
        for img_name, img_dets in img_to_dets.items():
            for cls, dets in img_dets.items():
                ids = nms_method(dets, nms_th)
                img_dets[cls] = dets[ids]
            img_to_dets[img_name] = img_dets

        return img_to_dets

    def to_original_dets_list(self):
        # 类别list序号顺序必须与cls_id顺序一致
        # 字典循环访问时，需要先对image_id和cls_id排序，以防止访问顺序不对
        pass

    def to_original_dets_dict(self, to_file='dets_dict.pkl'):
        cvtools.dump_pkl(self.img_to_dets, to_file)
        print('dets are saved in {}'.format(to_file))


if __name__ == '__main__':
    ann_file = '../../tests/data/DOTA/eval/val_dota_crop800_no_anns.json'
    croped_dets = '../../tests/data/DOTA/eval/dets.pkl'
    dets = CroppedDets(croped_dets, ann_file)
    dets.to_original_dets_dict(
        '../../tests/data/DOTA/eval/ori_dets_dict.pkl')
