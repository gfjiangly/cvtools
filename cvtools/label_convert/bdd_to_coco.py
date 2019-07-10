# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/5/17 10:17
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import os
import json
from tqdm import tqdm
import numpy as np
import cv2

from cvtools.utils.file import get_files_list


class BDD2COCO:
    def __init__(self, bdd_root, phase, save_name='custom', path_replace=None):
        self.bdd_root = bdd_root
        self.phase = phase
        self.save_name = save_name
        self.path_replace = path_replace
        # 这里是我需要的10个类别
        self.categorys = ['car', 'bus', 'person', 'bike', 'truck',
                          'motor', 'rider', 'traffic sign', 'traffic light']
        self.categorys_map = {}
        self.json_files = get_files_list(self.bdd_root+'labels/100k/'+self.phase, file_type='json')
        self.imageID = 1
        self.annID = 1
        self.coco_dataset = {
            "info": {
                "description": "bdd to coco dataset format.",
                "url": "http://www.gfjiang.com",
                "version": "0.1", "year": 2019,
                "contributor": "jiang",
                "date_created": "2019-05-09 09:11:52.357475"
            },
            "categories": [],
            "images": [], "annotations": []
        }

    def parse_json(self, json_file):
        """
        :param json_file: BDD00K数据集的一个json标签文件
        :return:
            返回一个，存储了一个json文件里面的方框坐标及其所属的类，
            形如：image_name, [[325.0, 342.0, 376.0, 384.0, 'car'], ...]
        """
        objs = []
        obj = []
        info = json.load(open(json_file))
        image_name = info['name']
        objects = info['frames'][0]['objects']
        for i in objects:
            if i['category'] in self.categorys:
                obj.append(int(i['box2d']['x1']))
                obj.append(int(i['box2d']['y1']))
                obj.append(int(i['box2d']['x2']))
                obj.append(int(i['box2d']['y2']))
                obj.append(i['category'])
                objs.append(obj)
                obj = []
        return image_name, objs

    def convert(self):
        for cls_id, cls_name in enumerate(self.categorys, start=1):
            self.categorys_map[cls_name] = cls_id
            self.coco_dataset['categories'].append({
                'id': cls_id,
                'name': cls_name,
                'supercategory': cls_name
            })

        for json_file in tqdm(self.json_files):
            image_name, objects = self.parse_json(json_file)
            image_name = self.bdd_root+'images/100k/'+self.phase+'/'+image_name+'.jpg'
            im = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), cv2.IMREAD_COLOR)
            if im is None:
                print('Waring: !!!can\'t read %s, continue this image' % image_name)
                continue
            height, width, _ = im.shape

            # 添加图像的信息到dataset中
            for key, value in self.path_replace.items():
                image_name = image_name.replace(key, value)
            self.coco_dataset["images"].append({
                'file_name': image_name,
                'id': self.imageID,
                'width': width,
                'height': height
            })

            for bbox in objects:
                cls_name = bbox[4]
                x1, y1, x2, y2 = bbox[:4]

                width = max(0., x2 - x1)
                height = max(0., y2 - y1)

                self.coco_dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': self.categorys_map[cls_name],  # 0 for backgroud
                    'id': self.annID,
                    'image_id': self.imageID,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })

                self.annID += 1
            self.imageID += 1
        self.save_json()

    def save_json(self):
        # 保存结果的文件夹
        folder = os.path.join(self.bdd_root, 'annotations')
        if not os.path.exists(folder):
            os.makedirs(folder)
        json_name = os.path.join(self.bdd_root, 'annotations/{name}_{phase}.json'.format(
            name=self.save_name, phase=self.phase))

        with open(json_name, 'w') as f:
            json.dump(self.coco_dataset, f)   # using indent=4 show more friendly


if __name__ == '__main__':
    name = 'custom'
    root_path = 'F:/bdd/bdd100k/'
    phase = 'train'

    # path_replace = {'F:/头肩检测分类/data/': '/home/arc-fsy8515/data/',
    #                 '/root/data/': '/home/arc-fsy8515/data/',
    #                 'E:/data': '/home/arc-fsy8515/data/'}
    path_replace = {'F:/': '/media/data1/jgf/',
                    '/root/data/': '/media/data1/jgf/',
                    'E:/data': '/media/data1/jgf/'}

    bdd2coco = BDD2COCO(root_path, phase, save_name=name, path_replace=path_replace)
    bdd2coco.convert()
