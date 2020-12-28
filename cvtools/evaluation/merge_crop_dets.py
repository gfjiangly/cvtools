# -*- encoding:utf-8 -*-
# @Time    : 2019/12/25 15:17
# @Author  : jiang.g.f
# @File    : merge_crop_dets.py
# @Software: PyCharm
import numpy as np
from collections import defaultdict
import os.path as osp
import re

import cvtools
from cvtools.ops.nms import py_cpu_nms
from cvtools.ops.polynms import poly_nms


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

    def __init__(self, 
                 cropped_dets, 
                 img_infos, 
                 num_coors,
                 nms_method=None,
                 nms_th=0.1,
                 classes=None):
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
        assert num_coors in (4, 8), "不支持的检测结果表示"
        self.num_coors = num_coors
        if nms_method is None:
            if self.num_coors == 4:
                self.nms_method = py_cpu_nms
            else:
                self.nms_method = poly_nms
        else:
            self.nms_method = nms_method
        self.nms_th = nms_th
        self.cropped_dets = cropped_dets
        self.img_infos = img_infos
        if cvtools.is_str(cropped_dets):
            self.cropped_dets = cvtools.load_pkl(cropped_dets)
        if cvtools.is_str(img_infos):
            self.img_infos = cvtools.COCO(img_infos)
        assert isinstance(self.cropped_dets, (list, dict))
        if isinstance(self.cropped_dets, list):
            self.cropped_dets_list = self.cropped_dets
            self.cropped_dets_dict = self.list_to_dict(self.cropped_dets)
        else:
            self.cropped_dets_dict = self.cropped_dets
            self.cropped_dets_list = self.dict_to_list(self.cropped_dets)
        self.ori_dets_dict = self.merge()
        self.ori_dets_list = self.dict_to_list(self.ori_dets_dict)
        self.new_classes_map = self.remap_classes(classes)
    
    def remap_classes(self, classes=None):
        old_classes = [cat['name'] for cat in self.img_infos.cats.values()]
        if classes is not None:
            new_classes_map = dict(zip(old_classes, classes))
        else:
            new_classes_map = dict(zip(old_classes, old_classes))
        return new_classes_map
    
    def list_to_dict(self, results_list):
        results_dict = {}
        for idx in range(len(self.img_infos.dataset['images'])):
            img_info = self.img_infos.dataset['images'][idx]
            image_id = img_info['id']
            result = results_list[idx]
            cls_to_dets = {}
            for cls_id, cls_det in zip(self.img_infos.cats, result):
                cls_to_dets[cls_id] = cls_det
            results_dict[image_id] = cls_to_dets
        return results_dict

    def dict_to_list(self, results_dict):
        results_list = []
        for cls_dets in results_dict.values():
            cls_results = []
            for dets in cls_dets.values():
                cls_results.append(dets)
            results_list.append(cls_results)
        return results_list

    def merge(self):
        """

        Args:
            nms_method: nms方法必须与bbox表示方式对应，
                默认是左上角右下角表示的水平矩形框
            nms_th: nms阈值

        Returns: 合并后的检测结果
            {
                ori_img1_name: {
                    cls1: dets,      # dets: np.array [坐标(长度为4或8)，得分]
                    cls2: dets,
                    ...
                },
                ori_img2_name: {
                    cls1: dets,
                    cls2: dets,
                    ... 
                },
                ...
            }
        """
        # 按ori_img_name汇集dets，并转换到原图坐标，注意image_id是子图id
        tmp_img_dets = defaultdict(list)
        for image_id, dets in self.cropped_dets_dict.items():
            img_info = self.img_infos.imgs[image_id]
            filename, _ = osp.splitext(img_info['file_name'])
            if 'crop' in img_info:
                start_point = img_info['crop'][:2]
                ori_img_name = filename
            else:
                # 待测试
                # crop_info = filename.split('_')[-4:]
                ori_img_name = filename.split('__')[0]
                pattern1 = re.compile(r'__\d+___\d+')
                x_y = re.findall(pattern1, filename)
                crop_info = re.findall(r'\d+', x_y[0])
                start_point = (int(crop_info[0]), int(crop_info[1]))
            cls_det = dict()  # 单张子图所有类别检测结果
            for cls_id, det in dets.items():
                if len(det) > 0:
                    bboxes = crop_bbox_map_back(det[:, :-1], start_point)
                    det = np.hstack([bboxes, det[:, [-1]]])
                cls_det[cls_id] = det
            # 用于容纳一张大图的所有子图检测结果
            tmp_img_dets[ori_img_name].append(cls_det)

        # 按类别合并子图结果
        ori_img_cls_dets = dict()
        for ori_img_name, ori_dets in tmp_img_dets.items():
            """tmp_img_dets:
            {
                ori_img1_name: [        # 大图1
                    cls_dets,           # 子图1结果，dict
                    cls_dets,           # 子图2结果，dict
                    ...,
                ]，
                ori_img2_name: [    # 大图2
                    cls_dets,           # 子图1结果，dict
                    cls_dets,           # 子图2结果，dict
                    ...,
                ]
            }
            """
            # original_dets（List）：单张大图中所有子图所有类别检测结果
            ori_dets_by_cls = dict()  # 按类别存放original_dets
            for sub_dets in ori_dets:
                for cls_id, det in sub_dets.items():
                    if cls_id not in ori_dets_by_cls:
                        ori_dets_by_cls[cls_id] = [det]
                    else:
                        ori_dets_by_cls[cls_id].append(det)
            for cls_id, dets in ori_dets_by_cls.items():
                ori_dets_by_cls[cls_id] = np.vstack(dets)
            ori_img_cls_dets[ori_img_name] = ori_dets_by_cls

        # 在大图上按类别做NMS
        for ori_img_name, cls_dets in ori_img_cls_dets.items():
            for cls_id, dets in cls_dets.items():
                ids = self.nms_method(dets, self.nms_th)
                cls_dets[cls_id] = dets[ids]
            ori_img_cls_dets[ori_img_name] = cls_dets

        return ori_img_cls_dets

    def to_original_dets_list(self, to_file='dets_list.pkl'):
        cvtools.dump_pkl(self.ori_dets_list, to_file)
        print('original dets are saved in {}'.format(to_file))

    def to_original_dets_dict(self, to_file='dets_dict.pkl'):
        cvtools.dump_pkl(self.ori_dets_dict, to_file)
        print('original dets are saved in {}'.format(to_file))
    
    def save_dota_format(self, save_dir):
        cat_results = defaultdict(lambda: defaultdict())
        for ori_img_name, cls_dets in self.ori_dets_dict.items():
            for cls_id, dets in cls_dets.items():
                cat_results[cls_id][ori_img_name] = dets
        for cls_id, img_dets in cat_results.items():
            cls_name = self.img_infos.cats[cls_id]['name']
            cls_name = self.new_classes_map[cls_name]
            lines = []
            for ori_img_name, dets in img_dets.items():
                sorted_ind = np.argsort(-dets[:, -1])
                dets = dets[sorted_ind]
                for det in dets:
                    bbox = list(map(str, det[:-1]))
                    score = str(det[-1])
                    lines.append(' '.join([ori_img_name] + [score] + bbox))
            cvtools.write_list_to_file(
                lines, osp.join(save_dir, cls_name + '.txt'))
        print('original dets are saved in {}'.format(save_dir))


if __name__ == '__main__':
    # # 测试无标注json和保存原图结果
    # ann_file = '../../tests/data/DOTA/eval/DOTA_val1024.json'
    # croped_dets = '../../tests/data/DOTA/eval/color_val1024_cropped_dets.pkl'
    # # ann_file = '../../tests/data/DOTA/eval/val_dota_crop800_no_anns.json'
    # # croped_dets = '../../tests/data/DOTA/eval/dets.pkl'
    # dets = CroppedDets(croped_dets, ann_file, num_coors=8)
    # dets.to_original_dets_dict(
    #     '../../tests/data/DOTA/eval/ori_dets_dict.pkl')
    # dets.to_original_dets_list(
    #     '../../tests/data/DOTA/eval/ori_dets_list.pkl')
    # dets.save_dota_format(
    #     '../../tests/data/DOTA/eval/Task1_results_nms')

    # convert to dota txt format
    ann_file = '/media/data/DOTA/dota1_1024/val1024/DOTA_val1024.json'
    cropped_dets = '/code/test_AerialDetection/AerialDetection/work_dirs/retinanet_obb_r50_fpn_1x_dota_1gpus_adapt_over1/val_cropped_dets.pkl'
    dets = CroppedDets(cropped_dets, ann_file, num_coors=8, classes=[
        'large-vehicle', 'swimming-pool',
        'helicopter', 'bridge',
        'plane', 'ship',
        'soccer-ball-field', 'basketball-court',
        'ground-track-field', 'small-vehicle',
        'harbor', 'baseball-diamond',
        'tennis-court', 'roundabout',
        'storage-tank'])
    dets.save_dota_format('/code/test_AerialDetection/AerialDetection/work_dirs/retinanet_obb_r50_fpn_1x_dota_1gpus_adapt_over1/Task1_results_nms')
