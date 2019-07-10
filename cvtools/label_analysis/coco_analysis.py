# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/6/24 9:57
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import copy
import cv2
import os
import numpy as np
from tqdm import tqdm

from cvtools.pycocotools.coco import COCO
from cvtools.utils import *
from .crop_in_order import CropInOder


_DEBUG = False


class COCOAnalysis(object):
    """coco-like datasets analysis"""
    def __init__(self, img_prefix, ann_file):
        self.img_prefix = img_prefix
        self.ann_file = ann_file
        self.coco_dataset = load_json(ann_file)
        # coco api已经帮我们写了一个COCO数据集表示类，需要什么数据从中取出即可
        self.COCO = COCO(ann_file)

    def split_dataset(self, to_file='data.json', val_size=0.1):
        imgs_train, imgs_val = split_dict(self.COCO.imgs, val_size)
        print('images: {} train, {} test.'.format(len(imgs_train), len(imgs_val)))
        name, suffix = os.path.splitext(to_file)
        dataset = copy.deepcopy(self.coco_dataset)
        # deal train data
        dataset['images'] = list(imgs_train.values())   # bad design
        anns = []
        for key in imgs_train.keys():
            anns += self.COCO.imgToAnns[key]
        dataset['annotations'] = anns
        save_json(dataset, to_file=name+'_train'+suffix)
        # deal test data
        dataset['images'] = list(imgs_val.values())
        anns = []
        for key in imgs_val.keys():
            anns += self.COCO.imgToAnns[key]
        dataset['annotations'] = anns
        save_json(dataset, to_file=name+'_val'+suffix)

    def cluster_boxes(self, save_root):
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        if _DEBUG:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))[:10]
        else:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        print(len(roidb))
        areas_relative = []
        areas_absolute = []
        for entry in roidb:
            ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
            objs = self.COCO.loadAnns(ann_ids)
            image_area = float(entry['width']*entry['height'])
            # Sanitize bboxes -- some are invalid
            for obj in objs:
                if 'ignore' in obj and obj['ignore'] == 1:
                    continue
                areas_absolute.append(obj['area'])
                areas_relative.append(obj['area'] / image_area)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        draw_hist(areas_absolute, bins=1000, x_label="Area", y_label="Box Quantity", title="Box Overall Distribution",
                  save_name='whole.png')
        draw_hist(areas_relative, bins=1000, x_label="Area", y_label="Box Quantity", title="Box Overall Distribution",
                  save_name=save_root+'whole_relative.png')
        areas_absolute = np.array(areas_absolute).reshape(-1, 1)
        areas_relative = np.array(areas_relative).reshape(-1, 1)
        areas_absolute_centers = k_means_cluster(areas_absolute, n_clusters=3)
        areas_relative_centers = k_means_cluster(areas_relative, n_clusters=3)
        print(areas_absolute_centers, areas_relative_centers)
        np.savetxt(save_root+'areas_absolute_centers.txt', areas_absolute_centers)
        np.savetxt(save_root+'areas_relative_centers.txt', areas_relative_centers)

    def cluster_boxes_cat(self, save_root):
        # TODO: 按类别统计box分布
        self.catToBoxes = [[] for _ in range(len(self.COCO.cats))]
        for key, ann in self.COCO.anns.items():
            # xywha = x1y1x2y2x3y3x4y4_to_xywha(ann['bbox'])
            # x1y1x2y2x3y3x4y4 = xywha_to_x1y1x2y2x3y3x4y4(xywha)
            self.catToBoxes[ann['category_id']-1].append(ann['bbox'])
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        for idx, cat in enumerate(self.catToBoxes):
            cat = np.array(cat)
            area = (cat[:, 2] * cat[:, 3])
            draw_hist(cat, bins=10, x_label="Area", y_label="Box Quantity",
                      title="Box Over Class Distribution",
                      save_name=save_root+self.COCO.cats[idx+1]['name']+'.png')
            if len(area) > 3:
                rbbox_centers = k_means_cluster(area.reshape(-1, 1), n_clusters=3)
                print(rbbox_centers)

    def stats_class_distribution(self, save_file):
        draw_cats = []
        makedirs(save_file)
        with open(save_file, 'w', encoding='utf8') as f:
            for cat_id in self.COCO.catToImgs:
                cat_name = self.COCO.cats[cat_id]['name']
                cat_num = len(self.COCO.catToImgs[cat_id])
                draw_cats += [cat_name]*cat_num
                line_str = '{}: {}\n'.format(cat_name, cat_num)
                f.write(line_str)
        draw_class_distribution(draw_cats, save_name=save_file.replace('txt', 'png'))

    def vis_boxes(self, save_root, box_format='x1y1wh'):
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        if _DEBUG:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))[:10]
        else:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        print('{} images.'.format(len(roidb)))
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        for entry in roidb:
            print('Visualize {}'.format(entry['file_name']))
            image_name = entry['file_name']
            image_file = os.path.join(self.img_prefix, image_name)
            img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)   # support chinese
            # img = cv2.imread(image_file)  # not support chinese
            if 'crop' in entry:
                img = img[entry['crop'][1]:entry['crop'][3], entry['crop'][0]:entry['crop'][2]]
                image_name = '_'.join([os.path.splitext(image_name)[0]] + list(map(str, entry['crop'])) +
                                      [os.path.splitext(image_name)[1]])
            if img is None:
                print('{} is None.'.format(image_file))
                continue
            ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
            objs = self.COCO.loadAnns(ann_ids)
            if len(objs) == 0:
                continue
            # Sanitize bboxes -- some are invalid
            for obj in objs:
                if 'ignore' in obj and obj['ignore'] == 1:
                    continue
                if 'bbox' not in obj:
                    obj['bbox'] = []
                class_name = self.COCO.cats[obj['category_id']]['name'] if 'category_id' in obj else ''
                img = draw_box_text(img, obj['bbox'], class_name, box_format=box_format)
            cv2.imwrite(os.path.join(save_root, image_name), img)

    def crop_in_order(self, save_root, box_format='x1y1wh'):
        crop = CropInOder(width_size=1920, height_size=1080, overlap=0.1)
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        if _DEBUG:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))[:10]
        else:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        print('{} images.'.format(len(roidb)))
        if not os.path.exists(save_root+'/images'):
            os.makedirs(save_root+'/images')
        if not os.path.exists(save_root+'/labelTxt+crop'):
            os.makedirs(save_root+'/labelTxt+crop')

        for entry in tqdm(roidb):
            print('crop {}'.format(entry['file_name']))
            image_name = entry['file_name']
            image_file = os.path.join(self.img_prefix, image_name)
            img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)  # support chinese
            # img = cv2.imread(image_file)  # not support chinese
            if img is None:
                print('{} is None.'.format(image_file))
                continue
            ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
            objs = self.COCO.loadAnns(ann_ids)
            boxes = [obj['bbox'] for obj in objs]
            # labels = [obj['category_id'] for obj in objs]
            crop_imgs, starts, new_ann_ids = crop(img, np.array(boxes), np.array(ann_ids))
            for crop_i, crop_img in enumerate(crop_imgs):
                img_name, img_suffix = os.path.splitext(image_name)
                # new_img_name = img_name + '_' + str(crop_i) + img_suffix
                # cv2.imwrite(os.path.join(save_root, 'images', new_img_name), crop_img)
                sx, sy = starts[crop_i]
                h, w, _ = crop_img.shape
                ex, ey = sx + w, sy + h
                txt_name = '_'.join([img_name]+[str(crop_i)]+list(map(str, [sx, sy, ex, ey]))) + '.txt'
                txt_content = ''
                crop_objs = self.COCO.loadAnns(new_ann_ids[crop_i])
                if len(crop_objs) == 0:
                    continue
                for crop_obj in crop_objs:
                    polygen = np.array(crop_obj['segmentation'][0]).reshape(-1, 2)
                    polygen = polygen - np.array(starts[crop_i]).reshape(-1, 2)
                    line = list(map(str, polygen.reshape(-1)))
                    cat = self.COCO.cats[crop_obj['category_id']]['name']
                    diffcult = str(crop_obj['difficult'])
                    line.append(cat)
                    line.append(diffcult)
                    txt_content += ' '.join(line) + '\n'
                with open(os.path.join(save_root, 'labelTxt+crop', txt_name), 'w') as f:
                    f.write(txt_content)


if __name__ == '__main__':
    img_prefix = 'F:/data/rssrai2019_object_detection/val/images'
    ann_file = 'E:/label_convert/rscup2019/rscup2019_x1y1wh_polygen_crop.json'
    coco_analysis = COCOAnalysis(img_prefix, ann_file)
    # coco_analysis.crop_in_order('F:/data/rssrai2019_object_detection/crop/val', box_format='x1y1wh')
    coco_analysis.vis_boxes('rscup2019/vis_rscup_crop/', box_format='x1y1wh')
    # coco_analysis.split_dataset(to_file='Arcsoft/gender_elevator/gender_elevator.json', val_size=1./3.)
    # coco_analysis.stats_class_distribution('Arcsoft/gender_elevator/class_distribution/class_distribution.txt')
    # coco_analysis.cluster_boxes('rscup2019/rbbox_distribution/')
    # coco_analysis.cluster_boxes_cat('rscup2019/rbbox_distribution/')
