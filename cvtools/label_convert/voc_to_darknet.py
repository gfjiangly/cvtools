import xml.etree.ElementTree as ET
import os.path as osp

import cvtools
from cvtools.evaluation.class_names import get_classes


class VOC2DarkNet(object):

    def __init__(self,
                 voc_root,
                 mode,
                 classes=get_classes('voc'),
                 use_xml_name=True,
                 read_test=False):
        self.voc_root = voc_root
        self.mode = mode
        self.use_xml_name = use_xml_name
        self.read_test = read_test
        if isinstance(classes, str):
            self.classes = cvtools.read_files_to_list(classes)
        else:
            self.classes = classes
        self.label_path = osp.join(self.voc_root, 'labels')
        cvtools.makedirs(self.label_path)

        file = osp.join(voc_root, 'ImageSets/Main/{}.txt'.format(mode))
        self.imgs = cvtools.read_files_to_list(file)
        self.img_paths = [
            'JPEGImages/{}.jpg'.format(img_name)  # relative path
            for img_name in self.imgs
        ]
        self.xml_paths = [
            osp.join(voc_root, 'Annotations/{}.xml'.format(img_name))
            for img_name in self.imgs]

    def convert(self):
        valid_imgs = []
        for index, xml_path in enumerate(self.xml_paths):
            print('parsing xml {} of {}: {}.xml'.format(
                index+1, len(self.imgs), self.imgs[index]))
            try:
                tree = ET.parse(xml_path)
            except FileNotFoundError:
                print('file {} is not found!'.format(xml_path))
                continue

            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            img_path = self.img_paths[index]
            if not self.use_xml_name:
                img_path = 'JPEGImages/{}'.format(root.find('filename').text)
            img_path = self.check_image(img_path)
            if img_path is None:
                print('{} failed to pass inspection'.format(self.xml_paths))
                continue

            objects = root.findall('object')
            if len(objects) == 0:
                print('Image {} has no object'.format(img_path))
                continue

            save_file = osp.join(self.label_path, '{}.txt'.format(self.imgs[index]))
            save_context = []
            valid_imgs.append(img_path)

            for obj in objects:
                cls = obj.find('name').text
                difficult = int(obj.find('difficult').text)
                if cls not in self.classes or difficult == 1:
                    continue
                cls_id = self.classes.index(cls)
                bnd_box = obj.find('bndbox')
                b = [
                    float(bnd_box.find('xmin').text),
                    float(bnd_box.find('xmax').text),
                    float(bnd_box.find('ymin').text),
                    float(bnd_box.find('ymax').text)
                ]
                bb = self.convert_box((w, h), b)
                save_context.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))
            cvtools.write_list_to_file(save_context, save_file)
        cvtools.write_list_to_file(valid_imgs, osp.join(self.voc_root, 'darknet_imglist.txt'))

    def convert_box(self, size, box):
        dw = 1./(size[0])
        dh = 1./(size[1])
        x = (box[0] + box[1])/2.0 - 1.
        y = (box[2] + box[3])/2.0 - 1.
        w = box[1] - box[0]
        h = box[3] - box[2]
        if x < 0 or y < 0. or w > size[0] or h > size[1]:
            return None
        x = 0. if x < 0 else x
        y = 0. if y < 0 else y
        w = size[0] if w > size[0] else w
        h = size[1] if h > size[1] else h
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return x, y, w, h

    def check_image(self, img_path):
        file = osp.join(self.voc_root, img_path)
        if not cvtools.isfile_casesensitive(file):
            image_types = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
            for suffix in image_types:
                img_path = osp.splitext(img_path)[0] + suffix
                file = osp.join(self.voc_root, img_path)
                if cvtools.isfile_casesensitive(file):
                    break
            if not cvtools.isfile_casesensitive(file):
                print("No images found in {}".format(osp.basename(img_path)))
                return None
        else:
            if self.read_test:
                try:
                    img = cvtools.imread(osp.join(self.voc_root, img_path))
                except Exception as e:
                    print(e, 'filter images {}'.format(img_path))
                    return None
                if img is None:
                    print('image {} is None'.format(img_path))
                    return None
        return img_path


if __name__ == '__main__':
    mode = 'trainval'
    root = 'D:/data/hat_detect/hat_V2.0'
    voc_to_darknet = VOC2DarkNet(root,
                                 mode=mode,
                                 classes=['head', 'hat'],
                                 use_xml_name=True,
                                 read_test=False)
    voc_to_darknet.convert()

