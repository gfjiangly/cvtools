# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/7/17 16:51
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import os.path as osp
import copy

import cvtools


class SplitDataset(object):
    """coco-like datasets analysis"""
    def __init__(self, ann_file):
        self.ann_file = ann_file
        self.coco_dataset = cvtools.load_json(ann_file)
        self.COCO = cvtools.COCO(ann_file)

    def split_dataset(self, to_file='data.json', val_size=0.1):
        imgs_train, imgs_val = cvtools.split_dict(self.COCO.imgs, val_size)
        print('images: {} train, {} test.'.format(len(imgs_train), len(imgs_val)))

        path, name = osp.split(to_file)
        dataset = copy.deepcopy(self.coco_dataset)

        # deal train data
        dataset['images'] = list(imgs_train.values())  # bad design
        anns = []
        for key in imgs_train.keys():
            anns += self.COCO.imgToAnns[key]
        dataset['annotations'] = anns
        cvtools.save_json(dataset, to_file=osp.join(path, 'train_'+name))

        # deal test data
        dataset['images'] = list(imgs_val.values())
        anns = []
        for key in imgs_val.keys():
            anns += self.COCO.imgToAnns[key]
        dataset['annotations'] = anns
        cvtools.save_json(dataset, to_file=osp.join(path, 'val_'+name))


if __name__ == '__main__':
    ann_file = 'arcsoft/elevator_gender.json'
    split_data = SplitDataset(ann_file)
    split_data.split_dataset(to_file=ann_file, val_size=0.5)
