# -*- encoding:utf-8 -*-
# @Time    : 2019/6/19 19:35
# @Author  : gfjiang
# @Site    : 
# @File    : rscup_to_coco.py
# @Software: PyCharm
import json
import os
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np

import cvtools.utils.file as file_utils
from cvtools.utils.timer import Timer, get_now_time_str


class Rscup2COCO(object):
    def __init__(self, label_root, image_root, path_replace=None, box_form='x1y1wh'):
        self.label_root = label_root
        self.image_root = image_root
        self.path_replace = path_replace
        self.box_form = box_form
        self.files = file_utils.get_files_list(label_root, basename=True)
        self.lines = []
        self.cls_map = file_utils.read_key_value('rscup2019/cat_id_map.txt')
        self.coco_dataset = {
            "info": {
                "description": "This is stable 1.0 version of the 2019 rscup race.",
                "url": "http://rscup.bjxintong.com.cn/#/theme/2",
                "version": "1.0", "year": 2019,
                "contributor": "rscup",
                "date_created": get_now_time_str()
            },
            "categories": [],
            "images": [], "annotations": []
        }
        self.imageID = 1
        self.annID = 1
        self.run_timer = Timer()

    def convert(self):
        for key, value in self.cls_map.items():
            self.coco_dataset['categories'].append({
                'id': int(value),
                'name': key,
                'supercategory': key
            })
        for file in tqdm(self.files):
            with open(os.path.join(self.label_root, file), 'r') as f:
                lines = f.readlines()
            # !only do once for one file label
            # image_name = file.replace('.txt', '.png')
            image_name = file.split('_')[0] + '.png'
            image_file = os.path.join(self.image_root, image_name)
            try:
                # self.run_timer.tic()
                "PIL: Open an image file, without loading the raster data"
                im = Image.open(image_file)
                # im = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), cv2.IMREAD_COLOR)
                # self.run_timer.toc(average=False)
                if im is None:
                    print('Waring: !!!can\'t read %s, continue this image' % image_file)
                    continue
                # height, width, _ = im.shape
                width, height = im.size
            except (FileNotFoundError, Image.DecompressionBombError) as e:
                print(e)
                continue

            # add image information to dataset
            for key, value in self.path_replace.items():
                image_name = image_name.replace(key, value)
            crop = list(map(int, os.path.basename(file).split('.')[0].split('_')[2:]))
            self.coco_dataset["images"].append({
                'file_name': image_name,    # relative path
                'id': self.imageID,
                'width': width,
                'height': height,
                'crop': crop
            })
            ignore = 0
            if height * width > 100000000:
                ignore = 1

            for line in lines:
                line = line.strip().split()
                if len(line) != 10:
                    continue
                polygon = list(map(float, [item.strip() for item in line[:8]]))
                a = np.array(polygon).reshape(4, 2)  # prepare for Polygon class input
                a_hull = cv2.convexHull(a.astype(np.float32), clockwise=False)  # 顺时针输出凸包

                if self.box_form == 'x1y1wh':
                    box = cv2.boundingRect(a_hull)
                    area = box[2] * box[3]
                elif self.box_form == 'xywha':
                    # self.run_timer.tic()
                    xywha = cv2.minAreaRect(a_hull)
                    area = xywha[1][0] * xywha[1][1]
                    box = list(xywha[0]) + list(xywha[1]) + [xywha[2]]
                    # x1y1x2y2x3y3x4y4 = cv2.boxPoints(xywha)
                    # self.run_timer.toc(average=False)
                elif self.box_form == 'x1y1x2y2x3y3x4y4':
                    # points order：left_up, left_down, right_down, right_up
                    # polygon = Polygon(a).convex_hull
                    # area = polygon.area
                    # polygon = np.array(polygon.exterior.coords[:]).reshape(1, -1).tolist()[0][:8]
                    # # points order：left_up, right_up, right_down, left_down
                    # polygon[2:4], polygon[6:8] = polygon[6:8], polygon[2:4]
                    area = cv2.contourArea(a_hull)
                    box = list(a_hull.reshape(-1).astype(np.float))
                else:
                    raise TypeError("not support {} box format!".format(self.box_form))

                box = list(map(lambda x: round(x, 2), box))
                cat = line[8].strip()
                difficult = int(line[9].strip())
                self.coco_dataset['annotations'].append({
                    'area': area,
                    'bbox': box,
                    'category_id': int(self.cls_map[cat]),  # 0 for backgroud
                    'id': self.annID,
                    'image_id': self.imageID,
                    'iscrowd': 0,
                    'ignore': ignore,
                    'difficult': difficult,
                    'segmentation': [polygon]
                })
                self.annID += 1
            self.imageID += 1
        # print('opencv: {}'.format(self.run_timer.total_time))

    def save_json(self, to_file='cocolike.json'):
        # save json format results to disk
        dirname = os.path.dirname(to_file)
        if dirname != '' and not os.path.exists(dirname):
            os.makedirs(os.path.dirname(dirname))
        with open(to_file, 'w') as f:
            json.dump(self.coco_dataset, f)  # using indent=4 show more friendly
        print('!save {} finished'.format(to_file))


if __name__ == '__main__':
    label_root = 'F:/data/rssrai2019_object_detection/crop/val/labelTxt+crop/'
    image_root = 'F:/data/rssrai2019_object_detection/val/images/'
    path_replace = {'\\': '/'}
    rscup_to_coco = Rscup2COCO(label_root, image_root, path_replace=path_replace, box_form='x1y1wh')
    rscup_to_coco.convert()
    rscup_to_coco.save_json('rscup2019/rscup2019_x1y1wh_polygen_crop.json')
