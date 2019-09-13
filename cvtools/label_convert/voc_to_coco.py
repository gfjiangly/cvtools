import os.path as osp
import xml.etree.ElementTree as ET
import json

import cvtools


class VOC2COCO(object):
    """convert voc-like dataset to coco-like dataset"""

    def __init__(self,
                 root,
                 mode='train',
                 cls='cat.txt',
                 cls_replace=None):
        self.root = root
        self.mode = mode
        self.cls_replace = cls_replace
        self.voc_classes = cvtools.read_files_to_list(cls)
        self.cls_map = {name: i + 1 for i, name in enumerate(self.voc_classes)}

        file = osp.join(root, 'ImageSets/Main/{}.txt'.format(mode))
        self.imgs = cvtools.read_files_to_list(file)
        self.img_paths = [
            'JPEGImages/{}.jpg'.format(img_name)    # relative path
            for img_name in self.imgs
        ]
        self.xml_paths = [
            osp.join(root, 'Annotations/{}.xml'.format(img_name))
            for img_name in self.imgs]

        # coco format dataset definition
        self.coco_dataset = {
            "info": {
                "description": "The PASCAL Visual Object Classes.",
                "url": "http://host.robots.ox.ac.uk/pascal/VOC/",
                "version": "1.0", "year": 2007,
                "contributor": "VOC",
                "date_created": cvtools.get_now_time_str()
            },
            "categories": [],
            "images": [], "annotations": []
        }
        self.imageID = 1
        self.annID = 1

    def convert(self):
        for key, value in self.cls_map.items():
            cls = self.cls_replace[key] \
                if key in self.cls_replace.keys() else key
            self.coco_dataset['categories'].append({
                'id': int(value),
                'name': cls,
                'supercategory': cls
            })
        for index, xml_path in enumerate(self.xml_paths):
            print('parsing xml {} of {}: {}.xml'.format(
                index, len(self.imgs), self.imgs[index]))
            img_path = self.img_paths[index]
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            if not cvtools.isfile_casesensitive(
                    osp.join(self.root, img_path)):
                print("No images found in {}".format(
                    osp.join(self.root, img_path)))

            img_info = {
                'file_name': img_path,  # relative path
                'id': self.imageID,
                'width': w,
                'height': h,
            }
            self.coco_dataset["images"].append(img_info)

            for obj in root.findall('object'):
                name = obj.find('name').text
                difficult = int(obj.find('difficult').text)
                bnd_box = obj.find('bndbox')
                bbox = [
                    int(bnd_box.find('xmin').text),
                    int(bnd_box.find('ymin').text),
                    int(bnd_box.find('xmax').text),
                    int(bnd_box.find('ymax').text)
                ]
                bbox = cvtools.x1y1x2y2_to_x1y1wh(bbox)
                area = bbox[2] * bbox[3]
                try:
                    cls_id = self.cls_map[name]
                except KeyError:
                    print('ignore {} in {}.xml'.format(name, self.imgs[index]))
                    continue
                self.coco_dataset['annotations'].append({
                    'area': area,
                    'bbox': bbox,
                    'category_id': cls_id,  # 0 for backgroud
                    'id': self.annID,
                    'image_id': self.imageID,
                    'iscrowd': 0,
                    'ignore': 0,
                    'difficult': difficult,
                    'segmentation': []
                })
                self.annID += 1
            self.imageID += 1

    def save_json(self, to_file='cocolike.json'):
        # save json format results to disk
        cvtools.utils.makedirs(to_file)
        with open(to_file, 'w') as f:
            json.dump(self.coco_dataset, f)
        print('!save {} finished'.format(to_file))


if __name__ == '__main__':
    mode = 'trainval'
    root = 'D:/data/hat_detect/SHWD/VOC2028'
    cls_replace = {'person': 'head'}
    voc_to_coco = VOC2COCO(root, mode=mode,
                           cls='hat_detect/cls.txt',
                           cls_replace=cls_replace)
    voc_to_coco.convert()
    to_file = 'D:/data/hat_detect/SHWD/VOC2028/' \
              'json/{}_shwd.json'.format(mode)
    voc_to_coco.save_json(to_file=to_file)
