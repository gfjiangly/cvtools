# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/7/17 16:51
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import os.path as osp
import copy

import cvtools
from cvtools.utils.misc import sort_dict


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

    def split_cats(self):
        catToImgs = sort_dict(self.COCO.catToImgs)
        self.catToDatasets = []
        for cat, img_ids in catToImgs.items():
            img_ids = set(img_ids)
            categories = [cat_info for cat_info in
                          self.coco_dataset['categories']
                          if cat_info['id'] == cat]
            images = [img_info for img_info in
                      self.coco_dataset['images']
                      if img_info['id'] in img_ids]
            annotations = [ann_info for ann_info in
                           self.coco_dataset['annotations']
                           if ann_info['category_id'] == cat]
            self.catToDatasets.append(
                {'info': self.coco_dataset['info'],
                 'categories': categories,
                 'images': images,
                 'annotations': annotations}
            )

    def save_cat_datasets(self, to_file):
        for dataset in self.catToDatasets:
            cvtools.save_json(
                dataset,
                to_file=to_file.format(
                    dataset['categories'][0]['name'])
            )


if __name__ == '__main__':
    ann_file = 'dota/test_dota+crop800x800.json'
    split_data = SplitDataset(ann_file)
    split_data.split_cats()
    split_data.save_cat_datasets(
        to_file='dota/test_dota_{}+crop800x800.json'
    )
