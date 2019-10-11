# -*- encoding:utf-8 -*-
# @Time    : 2019/9/9 20:46
# @Author  : gfjiang
# @Site    : 
# @File    : size_analysis.py
# @Software: PyCharm
from collections import defaultdict

import cvtools
from cvtools.cocotools.coco import COCO
from cvtools.utils.misc import sort_dict


class SizeAnalysis(object):
    """
    for small objects: area < 32^2
    for medium objects: 32^2 < area < 96^2
    for large objects: area > 96^2
    see http://cocodataset.org/#detection-eval
        and https://arxiv.org/pdf/1405.0312.pdf
    """
    def __init__(self, coco, size_range=(32, 96)):
        if isinstance(coco, str):
            coco = COCO(coco)
        assert isinstance(coco, COCO)
        self.coco = coco
        self.size_range = size_range
        # self.low_limit = 0
        # self.up_limit = 100000000
        self.createIndex()

    def createIndex(self):
        self.catToDatasets = []
        catToImgs = sort_dict(self.coco.catToImgs)
        for cat, img_ids in catToImgs.items():
            img_ids = set(img_ids)
            categories = [cat_info for cat_info in
                          self.coco.dataset['categories']
                          if cat_info['id'] == cat]
            images = [img_info for img_info in
                      self.coco.dataset['images']
                      if img_info['id'] in img_ids]
            annotations = [ann_info for ann_info in
                           self.coco.dataset['annotations']
                           if ann_info['category_id'] == cat]
            self.catToDatasets.append(
                {'info': self.coco.dataset['info'],
                 'categories': categories,
                 'images': images,
                 'annotations': annotations}
            )

    def stats_size_per_cat(self, to_file='size_per_cat_data.json'):
        self.cat_size = defaultdict(list)
        for cat_id, dataset in enumerate(self.catToDatasets):
            self.cat_size[dataset['categories'][0]['name']] = [
                ann_info['bbox'][2]*ann_info['bbox'][2]
                for ann_info in dataset['annotations']]
        self.cat_size = dict(
            sorted(self.cat_size.items(), key=lambda item: len(item[1])))
        g2_data = []
        for cat_name, sizes in self.cat_size.items():
            data_dict = dict()
            data_dict['Category'] = cat_name
            data_dict['small'] = len(
                [size for size in sizes
                 if pow(self.size_range[0], 2) >= size])
            data_dict['medium'] = len(
                [size for size in sizes
                 if pow(self.size_range[1], 2) >= size > pow(self.size_range[0], 2)])
            data_dict['large'] = len(
                [size for size in sizes
                 if size > pow(self.size_range[1], 2)])
            g2_data.append(data_dict)
        cvtools.save_json(g2_data, to_file)

    def stats_objs_per_img(self, to_file='stats_num.json'):
        total_anns = 0
        imgToNum = defaultdict()
        for cat_id, ann_ids in self.coco.catToImgs.items():
            imgs = set(ann_ids)
            total_anns += len(ann_ids)
            assert len(imgs) > 0
            cat_name = self.coco.cats[cat_id]['name']
            imgToNum[cat_name] = len(ann_ids) / float(len(imgs))
        imgToNum['total'] = total_anns / float(len(self.coco.imgs))
        print(imgToNum)
        cvtools.save_json(imgToNum, to_file)

    def stats_objs_per_cat(self, to_file='objs_per_cat_data.json'):
        cls_to_num = list()
        for cat_id in self.coco.catToImgs:
            item = dict()
            item['name'] = self.coco.cats[cat_id]['name']
            item['value'] = len(self.coco.catToImgs[cat_id])
            cls_to_num.append(item)
        cvtools.save_json(cls_to_num, to_file=to_file)


if __name__ == '__main__':
    size_analysis = SizeAnalysis('coco/instances_train2017.json')
    size_analysis.stats_size_per_cat(to_file='coco/size_per_cat_data.json')
    size_analysis.stats_objs_per_cat(to_file='coco/objs_per_cat_data.json')
