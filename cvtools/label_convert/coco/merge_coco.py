# -*- encoding:utf-8 -*-
# @Time    : 2019/8/31 17:52
# @Author  : gfjiang
# @Site    : 
# @File    : merge_coco.py
# @Software: PyCharm
from cvtools.utils.file import load_json, save_json


class MergeCOCO(object):
    """merge multiple coco-like datasets into one file"""

    def __init__(self, files):
        if not isinstance(files, (list, tuple)):
            raise TypeError('files must be a list, but got {}'.format(
                type(files)))
        assert len(files) > 1, 'least 2 files must be provided!'
        self.files = files
        self.merge_coco = load_json(self.files[0])
        self.img_ids = [img_info['id']
                        for img_info in self.merge_coco['images']]
        self.ann_ids = [img_info['id']
                        for img_info in self.merge_coco['annotations']]

    def update_img_ann_ids(self, images, anns):
        img_id_map = dict()
        img_max_id = max(self.img_ids)
        for i in range(len(images)):
            img_id_map[images[i]['id']] = \
                images[i]['id'] + img_max_id + 1
            images[i]['id'] += img_max_id + 1
        self.merge_coco['images'] += images

        ann_max_id = max(self.ann_ids)
        for i in range(len(anns)):
            anns[i]['id'] += ann_max_id + 1
            new_img_id = img_id_map[anns[i]['image_id']]
            anns[i]['image_id'] = new_img_id
            self.ann_ids.append(anns[i]['id'])
        self.merge_coco['annotations'] += anns

    def merge(self):
        for dataset in self.files[1:]:
            coco = load_json(dataset)
            self.update_img_ann_ids(
                coco['images'], coco['annotations'])

    def save(self, save='merge_coco.json'):
        save_json(self.merge_coco, save)


if __name__ == '__main__':
    dataset_files = ['../dota/val_dota+crop800x800.json',
                     '../dota/val_dota+orginal.json']
    merge_coco = MergeCOCO(dataset_files)
    merge_coco.merge()
    merge_coco.save(save='../dota/val_dota+orginal+crop800x800.json')

