import os.path as osp
import xml.etree.ElementTree as ET
import json

import cvtools
from cvtools.evaluation.class_names import get_classes


class VOC2COCO(object):
    """convert voc-like dataset to coco-like dataset

    Args:
        root (str): path include images, xml, file list
        mode (str): 'train', 'val', 'trainval', 'test'. used to find file list.
        cls (str or list): class name in a file or a list.
        cls_replace (dict): a dictionary for replacing class name. if not needed,
            you can just ignore it.
        use_xml_name (bool): image filename source, if true, using filename
            in xml for the image, otherwise using file list name for the image.
    """

    def __init__(self,
                 root,
                 mode='train',
                 cls=get_classes('voc'),
                 cls_replace=None,
                 use_xml_name=True,
                 read_test=False):
        self.root = root
        self.mode = mode
        self.cls_replace = cls_replace
        self.use_xml_name = use_xml_name
        self.read_test = read_test
        if isinstance(cls, str):
            self.voc_classes = cvtools.read_files_to_list(cls)
        else:
            self.voc_classes = cls
        self.cls_map = {name: i + 1 for i, name in
                        enumerate(self.voc_classes)}

        file = osp.join(root, 'ImageSets/Main/{}.txt'.format(mode))
        self.imgs = cvtools.read_files_to_list(file)
        self.img_paths = [
            'JPEGImages/{}.jpg'.format(img_name)  # relative path
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

    def convert(self, filter_objs=None):
        for key, value in self.cls_map.items():
            cls = key
            if self.cls_replace and key in self.cls_replace.keys():
                cls = self.cls_replace[key]
            self.coco_dataset['categories'].append({
                'id': int(value),
                'name': cls,
                'supercategory': cls
            })
        for index, xml_path in enumerate(self.xml_paths):
            print('parsing xml {} of {}: {}.xml'.format(
                index, len(self.imgs), self.imgs[index]))
            img_path = self.img_paths[index]
            try:
                tree = ET.parse(xml_path)
            except FileNotFoundError:
                print('file {} is not found!'.format(xml_path))
                continue
            root = tree.getroot()
            if self.use_xml_name:
                img_path = 'JPEGImages/{}'.format(
                    root.find('filename').text)
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            if not cvtools.isfile_casesensitive(
                    osp.join(self.root, img_path)):
                print("No images found in {}".format(
                    osp.join(self.root, img_path)))
            else:
                if self.read_test:
                    try:
                        cvtools.imread(osp.join(self.root, img_path))
                    except Exception as e:
                        print(e, 'filter images {}'.format(img_path))
                        continue

            img_info = {
                'file_name': img_path,  # relative path
                'id': self.imageID,
                'width': w,
                'height': h,
            }
            self.coco_dataset["images"].append(img_info)

            objects = root.findall('object')
            if filter_objs:
                objects = filter_objs(objects)
            for obj in objects:
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
    mode = 'train'
    root = 'D:/data/hat_detect/hat_V1.0'
    voc_to_coco = VOC2COCO(root, mode=mode,
                           cls=['head', 'hat'], read_test=True)
    voc_to_coco.convert()
    to_file = 'D:/data/hat_detect/hat_V1.0/' \
              'json/{}_hat_V1.json'.format(mode)
    voc_to_coco.save_json(to_file=to_file)
