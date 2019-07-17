# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/6/24 9:57
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import copy
import cv2
import os.path as osp
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import cvtools
from cvtools.label_analysis.crop_in_order import CropInOder


_DEBUG = False


class COCOAnalysis(object):
    """coco-like datasets analysis"""
    def __init__(self, img_prefix, ann_file):
        self.img_prefix = img_prefix
        self.ann_file = ann_file
        self.coco_dataset = cvtools.load_json(ann_file)
        self.COCO = cvtools.COCO(ann_file)

    def split_dataset(self, to_file='data.json', val_size=0.1):
        imgs_train, imgs_val = cvtools.split_dict(self.COCO.imgs, val_size)
        print('images: {} train, {} test.'.format(len(imgs_train), len(imgs_val)))
        name, suffix = osp.splitext(to_file)
        dataset = copy.deepcopy(self.coco_dataset)
        # deal train data
        dataset['images'] = list(imgs_train.values())   # bad design
        anns = []
        for key in imgs_train.keys():
            anns += self.COCO.imgToAnns[key]
        dataset['annotations'] = anns
        cvtools.save_json(dataset, to_file=name+'_train'+suffix)
        # deal test data
        dataset['images'] = list(imgs_val.values())
        anns = []
        for key in imgs_val.keys():
            anns += self.COCO.imgToAnns[key]
        dataset['annotations'] = anns
        cvtools.save_json(dataset, to_file=name+'_val'+suffix)

    def cluster_analysis(self, save_root, cluster_names=('bbox', )):
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        if _DEBUG:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))[:10]
        else:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        print(len(roidb))
        cluster_dict = defaultdict(list)
        for entry in roidb:
            ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
            objs = self.COCO.loadAnns(ann_ids)
            # Sanitize bboxes -- some are invalid
            for obj in objs:
                if 'ignore' in obj and obj['ignore'] == 1:
                    continue
                if 'area' in cluster_names:
                    cluster_dict['area'].append(obj['area'])
        cvtools.makedirs(save_root)
        for cluster_name, cluster_value in cluster_dict.items():
            if len(cluster_value) == 0:
                continue
            cvtools.draw_hist(cluster_value, bins=1000, x_label=cluster_name, y_label="Quantity",
                              title=cluster_name,
                              save_name=osp.join(save_root, cluster_name+'.png'))
            cluster_value = np.array(cluster_value).reshape(-1, 1)
            cluster_value_centers = cvtools.k_means_cluster(cluster_value, n_clusters=3)
            np.savetxt(osp.join(save_root, cluster_name+'.txt'), cluster_value_centers)

    def cluster_boxes_cat(self, save_root, cluster_names=('bbox', )):
        cluster_dict = defaultdict(lambda: defaultdict(list))
        for key, ann in self.COCO.anns.items():
            if 'area' in cluster_names:
                cat_name = self.COCO.cats[ann['category_id']]['name']
                cluster_dict['area'][cat_name].append(ann['area'])
        cvtools.makedirs(save_root)
        for cluster_name, cluster_values in cluster_dict.items():
            cluster_results = defaultdict(lambda: defaultdict(list))
            for cat, cluster_value in cluster_values.items():
                if len(cluster_value) > 3:
                    centers = cvtools.k_means_cluster(np.array(cluster_value).reshape(-1, 1), n_clusters=3)
                    cluster_results[cluster_name][cat].append(list(centers.reshape(-1)))
            cvtools.save_json(cluster_results, osp.join(save_root, '{}_by_cat.json'.format(cluster_name)))

    def stats_class_distribution(self, save_file):
        cls_to_num = dict()
        draw_cats = []
        for cat_id in self.COCO.catToImgs:
            cat_name = self.COCO.cats[cat_id]['name']
            cat_num = len(self.COCO.catToImgs[cat_id])
            draw_cats += [cat_name] * cat_num
            cls_to_num[cat_name] = cat_num
        cls_to_num = dict(sorted(cls_to_num.items(), key=lambda item: item[1]))
        cvtools.write_key_value(cls_to_num, save_file)
        cvtools.draw_class_distribution(draw_cats, save_name=save_file.replace('txt', 'png'))

    def vis_boxes_by_cat(self, save_root, vis_cats=None, vis='bbox', box_format='x1y1wh'):
        catImgs = copy.deepcopy(self.COCO.catToImgs)
        catImgs = {cat: set(catImgs[cat]) for cat in catImgs}
        for cat_id, image_ids in catImgs.items():
            cat_name = self.COCO.cats[cat_id]['name']
            if vis_cats is not None and cat_name not in vis_cats:
                continue
            print('Visualize %s' % cat_name)
            if _DEBUG:
                roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))[:10]
            else:
                roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
            for i, entry in enumerate(roidb):
                print('Visualize image %d of %d: %s' % (i, len(roidb), entry['file_name']))
                image_name = entry['file_name']
                image_file = osp.join(self.img_prefix, image_name)
                img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)  # support chinese
                # img = cv2.imread(image_file)  # not support chinese
                image_name = osp.splitext(image_name)[0]
                if 'crop' in entry:
                    img = img[entry['crop'][1]:entry['crop'][3], entry['crop'][0]:entry['crop'][2]]
                    image_name = '_'.join([image_name] + list(map(str, entry['crop'])))
                if img is None:
                    print('{} is None.'.format(image_file))
                    continue
                ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
                objs = self.COCO.loadAnns(ann_ids)
                for obj in objs:
                    if obj['category_id'] != cat_id:
                        continue
                    if 'ignore' in obj and obj['ignore'] == 1:
                        continue
                    vis_obj = []
                    if vis in obj:
                        vis_obj = obj[vis]
                    class_name = [cat_name if 'category_id' in obj else '']
                    img = cvtools.draw_boxes_texts(img, vis_obj, class_name, box_format=box_format)
                # save in jpg format for saving storage
                cvtools.imwrite(img, osp.join(save_root, cat_name, image_name + '.jpg'))

    def vis_boxes(self, save_root, vis='bbox', box_format='x1y1wh'):
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        if _DEBUG:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))[:10]
        else:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        print('{} images.'.format(len(roidb)))
        cvtools.makedirs(save_root)
        for i, entry in enumerate(roidb):
            print('Visualize image %d of %d: %s' % (i, len(roidb), entry['file_name']))
            image_name = entry['file_name']
            image_file = osp.join(self.img_prefix, image_name)
            img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)   # support chinese
            # img = cv2.imread(image_file)  # not support chinese
            image_name = osp.splitext(image_name)[0]
            if 'crop' in entry:
                img = img[entry['crop'][1]:entry['crop'][3], entry['crop'][0]:entry['crop'][2]]
                image_name = '_'.join([image_name] + list(map(str, entry['crop'])))
            if img is None:
                print('{} is None.'.format(image_file))
                continue
            ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
            objs = self.COCO.loadAnns(ann_ids)
            if len(objs) == 0:
                continue
            # Sanitize bboxes -- some are invalid
            for obj in objs:
                vis_obj = []
                if 'ignore' in obj and obj['ignore'] == 1:
                    continue
                if vis in obj:
                    vis_obj = obj[vis]
                class_name = self.COCO.cats[obj['category_id']]['name'] if 'category_id' in obj else ''
                img = cvtools.draw_boxes_texts(img, vis_obj, class_name, box_format=box_format)
            # save in jpg format for saving storage
            cvtools.imwrite(img, osp.join(save_root, image_name + '.jpg'))

    def crop_in_order(self, save_root, box_format='x1y1wh'):
        crop = CropInOder(width_size=1920, height_size=1080, overlap=0.1)
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        if _DEBUG:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))[:10]
        else:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        print('{} images.'.format(len(roidb)))

        cvtools.makedirs(save_root+'/images')
        cvtools.makedirs(save_root+'/labelTxt+crop')

        for entry in tqdm(roidb):
            print('crop {}'.format(entry['file_name']))
            image_name = entry['file_name']
            image_file = osp.join(self.img_prefix, image_name)
            img_name_no_suffix, img_suffix = osp.splitext(image_name)
            img = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)  # support chinese
            # img = cv2.imread(image_file)  # not support chinese
            if img is None:
                print('{} is None.'.format(image_file))
                continue
            ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
            objs = self.COCO.loadAnns(ann_ids)
            boxes = [obj['bbox'] for obj in objs]
            # labels = [obj['category_id'] for obj in objs]
            if len(boxes) == 0:
                continue
            crop_imgs, starts, new_ann_ids = crop(img, np.array(boxes), np.array(ann_ids))
            # crops = []
            for crop_i, crop_img in enumerate(crop_imgs):
                # new_img_name = img_name + '_' + str(crop_i) + img_suffix
                # cv2.imwrite(os.path.join(save_root, 'images', new_img_name), crop_img)
                sx, sy = starts[crop_i]
                h, w, _ = crop_img.shape
                ex, ey = sx + w, sy + h
                # crops.append([sx+3, sy+3, ex-3, ey-3])
                txt_name = '_'.join([img_name_no_suffix] +
                                    [str(crop_i)]+list(map(str, [sx, sy, ex, ey]))) + '.txt'
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
                cvtools.strwrite(txt_content, osp.join(save_root, 'labelTxt+crop', txt_name))
            # if len(crops) > 0:
            #     draw_img = cvtools.draw_boxes_texts(img, crops, line_width=3, box_format='x1y1x2y2')
            #     cvtools.imwrite(draw_img, osp.join(save_root, 'images', img_name_no_suffix+'.jpg'))


if __name__ == '__main__':
    img_prefix = 'D:/data/rssrai2019_object_detection/train/images'
    ann_file = '../label_convert/rscup/train_crop1920x1080_rscup_x1y1wh_polygen.json'
    coco_analysis = COCOAnalysis(img_prefix, ann_file)
    # coco_analysis.crop_in_order('rscup/crop/train', box_format='x1y1wh')
    # coco_analysis.vis_boxes_by_cat('rscup/vis_rscup/', vis_cats=('helipad', ),
    #                                vis='segmentation', box_format='x1y1x2y2x3y3x4y4')
    # coco_analysis.vis_boxes('rscup/vis_rscup_crop/', vis='segmentation', box_format='x1y1x2y2x3y3x4y4')
    coco_analysis.vis_boxes('rscup/vis_rscup_crop_box/', vis='bbox', box_format='x1y1wh')
    # coco_analysis.split_dataset(to_file='Arcsoft/gender_elevator/gender_elevator.json', val_size=1./3.)
    # coco_analysis.stats_class_distribution('rscup/class_distribution/class_distribution.txt')
    # coco_analysis.cluster_analysis('rscup/bbox_distribution/', cluster_names=('area', ))
    # coco_analysis.cluster_boxes_cat('rscup/bbox_distribution/', cluster_names=('area', ))
