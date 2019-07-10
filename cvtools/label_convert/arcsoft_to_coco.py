# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/6/28 13:41
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import os
import json
from tqdm import tqdm
from PIL import Image

from cvtools.utils import *
from .arcsoft.attribute_combination import *


class Arcsoft2COCO(object):
    """convert arcsoft format label to standard coco format."""
    def __init__(self, path, path_replace=None, img_suffix='.jpg'):
        self.path = path
        self.path_replace = path_replace
        self.img_suffix = img_suffix
        self.img_list = get_files_list(self.path, file_type=img_suffix)
        self.txt_list = get_files_list(self.path, file_type='.txt')
        # you could comment this sentence if you don't want check integrity of images and labels.
        # assert len(self.img_list) == len(self.txt_list)
        self.cls_map = read_key_value('arcsoft/cat_id_map.txt')
        self.coco_dataset = {
            "info": {
                "description": "This is unstable 0.0.0 version of the 2019 Projects Data.",
                "url": "http://www.arcsoft.com",
                "version": "1.0", "year": 2019,
                "contributor": "arcsoft",
                "date_created": get_now_time_str()
            },
            "categories": [],  # Not added yet
            "images": [], "annotations": []
        }
        self.imageID = 1
        self.annID = 1
        self.run_timer = Timer()

    def convert(self, label_processor=rect_reserved):
        # the latter content covers the previous content, if the id is repeated.
        id_cats = {value: key for key, value in self.cls_map.items()}
        for key, value in id_cats.items():
            self.coco_dataset['categories'].append({
                'id': int(key),
                'name': value,
                'supercategory': value
            })
        for txt_file in tqdm(self.txt_list):
            img_file = txt_file.replace('.txt', self.img_suffix)
            # read the image to get width and height
            try:
                # "PIL: Open an image file, without loading the raster data"
                im = Image.open(img_file)
                if im is None:
                    print('Waring: !!!can\'t read %s, continue this image' % img_file)
                    continue
                width, height = im.size
            except (FileNotFoundError, Image.DecompressionBombError) as e:
                print(e)    # Image.DecompressionBombError for the big size image
                continue

            # add image information to dataset
            if self.path_replace is not None:
                for key, value in self.path_replace.items():
                    img_file = img_file.replace(key, value)
            self.coco_dataset["images"].append({
                'file_name': img_file,
                'id': self.imageID,
                'width': width,
                'height': height
            })

            # read txt label
            labels = read_arcsoft_txt_format(txt_file)
            if len(labels) == 0:
                continue
            for label in labels:
                label = label_processor(label)    # change here for specific labels
                if len(label['bbox']) == 0:
                    continue    # may be not happened
                label['id'] = self.annID
                label['image_id'] = self.imageID
                label['iscrowd'] = 0
                if 'category' not in label.keys():
                    label['category'] = 'head'
                try:
                    label['category_id'] = int(self.cls_map[label['category']])  # 0 for backgroud
                except KeyError:
                    print('skip file: {}'.format(txt_file))
                    break
                self.coco_dataset['annotations'].append(label)
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
    path_replace = {'\\': '/', 'F:/data/detection/elevator/gender/': '/media/data1/jgf/elevator/'}
    arcsoft_to_coco = Arcsoft2COCO('F:/data/detection/elevator/gender',
                                   path_replace=path_replace, img_suffix='.jpg')
    arcsoft_to_coco.convert(label_processor=gender_reserved)
    arcsoft_to_coco.save_json('arcsoft/gender_elevator.json')
