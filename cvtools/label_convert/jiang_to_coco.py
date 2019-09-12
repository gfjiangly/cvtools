# -*- coding:utf-8 -*-
# author: gfjiangly
# time: 2019/5/9 15:21
# e-mail: jgf0719@foxmail.com
# software: PyCharm

import json
import os
from tqdm import tqdm
from PIL import Image

from cvtools.utils.file import read_files_to_list
from cvtools.utils.timer import get_now_time_str
from cvtools.utils.file import read_key_value


class Jiang2COCO(object):
    """
    此转换的coco格式数据集与官方格式有一点差异：
    官方的image name仅仅是图片名，不包含路径，路径是组装起来查找的，图片名命令也是有规律的，
    但由于私有的数据集图片名和路径可能比较混乱，没有统一处理。因此在image name中包含了图片地址，
    使用coco api读取标签文件时须注意这一点。
    """
    def __init__(self, root, files, cls_map='', path_replace=None):
        self.root = root
        self.files = files
        self.path_replace = path_replace
        self.lines = read_files_to_list(self.files, root=self.root)
        self.cls_first = False
        self.cls_map = read_key_value(cls_map)
        self.imageID = 1
        self.annID = 1
        self.coco_dataset = {
            "info": {
                "description": "This is stable 0.0.0 version of the 2019 jiang's dataset format.",
                "url": "http://www.gfjiang.com",
                "version": "0.1", "year": 2019,
                "contributor": "jiang",
                "date_created": get_now_time_str()
            },
            "categories": [],
            "images": [], "annotations": []
        }

    def convert(self):
        for key, value in self.cls_map.items():
            self.coco_dataset['categories'].append({
                'id': int(key) + 1,
                'name': value,
                'supercategory': value
            })

        for line in tqdm(self.lines):
            line = line.strip().split()
            image_name = line[0]
            for key, value in self.path_replace.items():
                image_name = image_name.replace(key, value)
            # im = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), cv2.IMREAD_COLOR)
            # if im is None:
            #     print('Waring: !!!can\'t read %s, continue this image' % image_name)
            #     continue
            # height, width, _ = im.shape

            # read the image to get width and height
            try:
                # "PIL: Open an image file, without loading the raster data"
                im = Image.open(image_name)
                if im is None:
                    print('Waring: !!!can\'t read %s, continue this image' % image_name)
                    continue
                width, height = im.size
            except (FileNotFoundError, Image.DecompressionBombError) as e:
                print(e)    # Image.DecompressionBombError for the big size image
                continue

            # 添加图像的信息到dataset中
            self.coco_dataset["images"].append({
                'file_name': image_name,
                'id': self.imageID,
                'width': width,
                'height': height
            })

            for bbox in line[1:]:
                bbox = list(map(int, bbox.strip().split(',')))
                # 类别
                if self.cls_first:
                    coor_start = 1
                    cls_id = bbox[0]
                else:
                    coor_start = 0
                    cls_id = bbox[4]
                x1, y1, x2, y2 = map(float, bbox[coor_start:coor_start+4])
                width = max(0., x2 - x1)
                height = max(0., y2 - y1)

                self.coco_dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': int(cls_id) + 1,  # 0 for backgroud
                    'id': self.annID,
                    'image_id': self.imageID,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })

                self.annID += 1
            self.imageID += 1

    def save_json(self, to_file='cocolike.json'):
        # save json format results to disk
        dirname = os.path.dirname(to_file)
        if dirname != '' and not os.path.exists(dirname):
            os.makedirs(os.path.dirname(dirname))
        with open(to_file, 'w') as f:
            json.dump(self.coco_dataset, f)  # using indent=4 show more friendly
        print('!save {} finished'.format(to_file))


if __name__ == '__main__':

    root_path = 'jiang/label/train/'
    # files_list = ['elevator_20181230_convert_train.txt', 'elevator_20181231_convert_train.txt',
    #               'elevator_20190106_convert_train.txt', 'person_7421_train.txt']
    # phase = 'train'
    #
    # files_list = ['elevator_20181230_convert_test.txt', 'elevator_20181231_convert_test.txt',
    #               'elevator_20190106_convert_test.txt', 'person_1856_test.txt']
    # # files_list = ['elevator_20181230_convert_test.txt']
    # phase = 'test'
    #
    # path_replace = {'E:/头肩检测分类/data/': '/home/arc-fsy8515/data/',
    #                 '/root/data/': '/home/arc-fsy8515/data/',
    #                 'E:/data': '/home/arc-fsy8515/data/'}

    files_list = ['our_train.txt']
    path_place = {'/root/data/': 'F:/data/detection/'}
    jiang2coco = Jiang2COCO(root_path, files_list, path_replace=path_place,
                            cls_map='jiang/our_cat_id_map.txt')
    jiang2coco.convert()
    jiang2coco.save_json(to_file='jiang/our.json')




