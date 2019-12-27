# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/6/10 10:09
# e-mail   : jgf0719@foxmail.com
# software : PyCharm

import json
from collections import defaultdict
import cvtools.utils.boxes as box_utils
# COCO API
# from pycocotools.coco import COCO


class COCO2Jiang(object):
    def __init__(self, root, files, phase, save_name='custom', prefix=None):
        self.root = root
        self.files = files
        self.phase = phase
        self.save_name = save_name
        self.prefix = prefix
        self.dataset = {}
        for json_file in files:
            # self.COCO = COCO(root+json_file)
            coco_json = json.load(open(root+json_file, 'r'))
            self.dataset.update(coco_json)
        self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def convert(self):
        line = ''
        for img_id in self.imgToAnns:
            file_name = self.prefix + self.imgs[img_id]['file_name']
            line += file_name + ' '
            for ann in self.imgToAnns[img_id]:
                bbox = ann['bbox']
                bbox = box_utils.x1y1wh_to_x1y1x2y2(bbox)
                cat = ann['category_id']
                bbox_str = ','.join(map(str, map(int, bbox)))
                line += bbox_str + ',' + str(cat) + ' '
            line += '\n'
        self.save(line)
        print('save in {}.'.format(self.save_name))

    def save(self, line):
        with open(self.save_name, 'w', encoding='utf-8') as fp:
            fp.write(line)


if __name__ == '__main__':
    name = 'jiang/train2017.txt'
    root_path = 'coco/annotations/coco/'
    # E:\label_convert\coco\annotations\coco
    files_list = ['instances_train2017.json']
    phase = 'train'
    prefix = '/media/data1/jgf/coco/train2017/'

    coco2jiang = COCO2Jiang(root_path, files_list, phase, save_name=name, prefix=prefix)
    coco2jiang.convert()
