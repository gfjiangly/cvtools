# -*- encoding:utf-8 -*-
# @Time    : 2019/12/25 15:17
# @Author  : jiang.g.f
# @File    : merge_dets.py
# @Software: PyCharm
import numpy as np
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm
import os.path as osp

import cvtools


class MergeCropDetResults(object):

    def __init__(self, anns_file, results, num_coors=4, num_workers=0):
        """

        Args:
            anns_file (str): DOTA-COCO兼容格式, img_info中必须含crop字段
            results (str): 子图检测结果，保存在文件中，
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
            num_workers (int): 进程数，默认值0为表示不使用多进程
        """
        assert num_coors in (4, 8), "不支持的检测位置表示"
        self.coco = cvtools.COCO(anns_file)
        self.anns = self.coco.anns
        if cvtools.is_str(results):
            self.results = cvtools.load_pkl(results)
        else:
            self.results = results
        self.num_coors = num_coors
        self.num_workers = num_workers
        self.img_to_results = {}    # merge返回结果保存

    def crop_bbox_map_back(self, bb, crop_start):
        """子图检测坐标转换为相对于整图的坐标"""
        bb_shape = bb.shape
        original_bb = bb.reshape(-1, 2) + np.array(crop_start).reshape(-1, 2)
        return original_bb.reshape(bb_shape)

    def match_part_img_results(self, results, coor_len=4):
        """结合子图的结果，映射回原图，应用nms, 生成一张整图的结果"""
        img_to_results = defaultdict(list)
        for image_id, dets in results.items():
            img_info = self.coco.imgs[image_id]
            labels = cvtools.concat_list(
                [[j]*len(det) for j, det in dets.items()])    # label从0开始
            scores = cvtools.concat_list([det[:, coor_len]
                                          for det in dets.values()])
            bboxes = np.vstack([det[:, :coor_len] for det in dets.values()
                                if len(det) > 0])
            if 'crop' in img_info:
                bboxes = self.crop_bbox_map_back(bboxes, img_info['crop'][:2])
            assert len(bboxes) == len(labels)
            if len(labels) > 0:
                result = [bboxes, labels, scores]
                img_to_results[img_info['file_name']].append(result)
        return img_to_results

    def merge(self, nms_method=cvtools.py_cpu_nms, nms_th=0.15):
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
        img_to_results = defaultdict(list)
        if self.num_workers > 0:
            pool = Pool(processes=self.num_workers)
            num = len(self.anns) // self.num_workers
            anns_group = [self.anns[i:i + num]
                          for i in range(0, len(self.anns), num)]
            results_group = [self.results[i:i + num] for i in
                             range(0, len(self.results), num)]
            res = []
            for anns, results in tqdm(zip(anns_group, results_group)):
                res.append(pool.apply_async(self.match_part_img_results,
                                            args=(anns, self.results,)))
            pool.close()
            pool.join()
            for item in res:
                img_to_results.update(item.get())
        else:
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
        class_names = [cat['name'] for cat in self.coco.cats]
        self.cat_results = self.img_results_to_cat_results(self.img_to_results)
        path = osp.join(save_path, 'Task1_{}.txt')
        for cat_id, result in self.cat_results.items():
            lines = []
            for filename, score, bbox in result:
                filename = osp.splitext(filename)[0]
                bbox = list(map(str, list(bbox)))
                score = str(round(score, 3))
                lines.append(' '.join([filename] + [score] + bbox))
            cvtools.write_list_to_file(lines, path.format(class_names[cat_id]))


class ConvertGT2Dets(object):
    """
        将DOTA-COCO兼容格式GT转成检测结果表达形式results，保存成pkl
        results:
        {
            image_id: dets,  # image_id必须是anns中有效的id
            image_id: dets,
            ...
        }
        dets:
        {
            类别：[[位置坐标，得分], [...], ...],
            类别: [[位置坐标，得分], [...], ...],
            ...
        }，
    """
    def __init__(self, anns_file):
        self.coco = cvtools.COCO(anns_file)
        self.results = defaultdict()    # 动态创建嵌套字典

    def convert(self, to_file=None):
        for img_id, img_info in self.coco.imgs.items():
            dets = defaultdict(list)
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                x1y1x2y2 = cvtools.x1y1wh_to_x1y1x2y2(ann['bbox'])
                dets[ann['category_id']].append(x1y1x2y2 + [1.])
            for cls, det in dets.items():
                dets[cls] = np.array(det)
            self.results[img_id] = dets
        if to_file is not None:
            cvtools.dump_pkl(self.results, to_file)
        return self.results
