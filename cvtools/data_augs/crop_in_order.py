# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/7/9 13:05
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import cv2
import copy
import os.path as osp
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import cvtools


class CropInOder(object):

    def __init__(self, img_prefix=None, ann_file=None,
                 width_size=1920, height_size=1080, overlap=0.):
        self.img_prefix = img_prefix
        if ann_file is not None:
            self.ann_file = ann_file
            self.coco_dataset = cvtools.load_json(ann_file)
            self.COCO = cvtools.COCO(ann_file)
            self.catToAnns = defaultdict(list)
            if 'annotations' in self.coco_dataset:
                for ann in self.coco_dataset['annotations']:
                    self.catToAnns[ann['category_id']].append(ann)

        assert 1920 >= width_size >= 800 and 1080 >= height_size >= 800 and 0.5 >= overlap >= 0.
        self.width_size = int(width_size)
        self.height_size = int(height_size)
        self.overlap = overlap

    def crop(self, img, boxes=None, labels=None, iof_th=0.8):
        h, w, c = img.shape
        crop_imgs = []
        starts = []
        # fix crop bug!
        y_stop = False
        for sy in range(0, h, int(self.height_size*(1.-self.overlap))):
            x_stop = False
            for sx in range(0, w, int(self.width_size * (1. - self.overlap))):
                ex = sx + self.width_size
                if ex > w:
                    ex = w
                    sx = w - self.width_size
                    if sx < 0:
                        sx = 0
                    x_stop = True
                ey = sy + self.height_size
                if ey > h:
                    ey = h
                    sy = h - self.height_size
                    if sy < 0:
                        sy = 0
                    y_stop = True
                # sy, ey, sx, ex = int(sy), int(ey), int(sx), int(ex)
                crop_imgs.append(img[sy:ey, sx:ex])
                starts.append((sx, sy))
                if x_stop:
                    break
            if y_stop:
                break

        # if boxes is not None and labels is not None and \
        #         len(labels) > 0 and len(crop_imgs) > 1:
        #     assert len(boxes) == len(labels)
        #     gt_boxes = cvtools.x1y1wh_to_x1y1x2y2(boxes)
        #     return_imgs = []
        #     return_starts = []
        #     # return_boxes = []
        #     return_labels = []
        #     for i, crop_img in enumerate(crop_imgs):
        #         crop_h, crop_w, _ = crop_img.shape
        #         crop_x1, crop_y1 = starts[i]
        #         img_box = cvtools.x1y1wh_to_x1y1x2y2(np.array([[crop_x1, crop_y1, crop_w, crop_h]]))
        #         iof = cvtools.bbox_overlaps(gt_boxes.reshape(-1, 4),
        #                                     img_box.reshape(-1, 4), mode='iof').reshape(-1)
        #         ids_in = iof >= 1.  # 完全在crop区域内的
        #         ids_crop = (iof > iof_th) & (iof < 1.)  # 部分在crop区域内的，经处理可接受的
        #         labels_in = labels[ids_in]
        #         if len(labels_in) > 0:
        #             return_imgs.append(crop_img)
        #             return_starts.append(starts[i])     # fix not skip start bug
        #             return_labels.append([labels_in, labels[ids_crop]])
        #     return return_imgs, return_starts, return_labels
        #
        # return crop_imgs, starts, [labels]
        return crop_imgs, starts

    # 处理位于剪裁图像边缘的框，iof大于阈值的处理后予以保留
    def deal_edged_boxes(self, ann_ids, crop_imgs, starts, iof_th=0.8):
        objs = self.COCO.loadAnns(ann_ids)
        ann_ids = np.array(ann_ids)
        boxes = [obj['bbox'] for obj in objs]
        segms = [obj['segmentation'] for obj in objs]
        if len(boxes) == 0:
            return None, None
        assert len(boxes) == len(ann_ids)
        gt_boxes = cvtools.x1y1wh_to_x1y1x2y2(np.array(boxes))
        img_to_objs = defaultdict()
        obj_to_num = defaultdict()
        for i, crop_img in enumerate(crop_imgs):
            sx, sy = starts[i]
            h, w, _ = crop_img.shape
            ex, ey = sx + w, sy + h

            img_box = cvtools.x1y1wh_to_x1y1x2y2(np.array([[sx, sy, w, h]]))
            iof = cvtools.bbox_overlaps(gt_boxes.reshape(-1, 4),
                                        img_box.reshape(-1, 4), mode='iof').reshape(-1)
            ids_in = iof >= 1.  # 完全在crop区域内的
            in_crop_objs = self.COCO.loadAnns(ann_ids[ids_in])
            img_to_objs[(sx, sy, ex, ey)] = in_crop_objs
            objs_stats = in_crop_objs

            ids_crop = (iof > iof_th) & (iof < 1.)  # 部分在crop区域内的，经处理可接受的
            crop_id = ann_ids[ids_crop]
            # TODO：对于超出边界的boxes和segms处理
            # 要判断是大目标还是小目标？
            part_crop_objs = self.COCO.loadAnns(crop_id)
            img_to_objs[(sx, sy, ex, ey)] += self.COCO.loadAnns(crop_id)

            objs_stats += part_crop_objs
            for obj in objs_stats:
                if obj['id'] in obj_to_num:
                    obj_to_num[obj['id']] += 1
                else:
                    obj_to_num[obj['id']] = 1
        return img_to_objs, obj_to_num

    def crop_with_label(self, save_root='./', iof_th=0.5):
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        if cvtools._DEBUG:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))[:10]
        else:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        print('{} images.'.format(len(roidb)))

        cvtools.makedirs(save_root+'/images')
        cvtools.makedirs(save_root+'/labelTxt+crop')

        stats = defaultdict(crop_objs=0, total_objs=0, missing_objs=0, total_croped_images=0)
        for entry in tqdm(roidb):
            if cvtools._DEBUG:
                print('crop {}'.format(entry['file_name']))
            # read image
            image_name = entry['file_name']
            image_file = osp.join(self.img_prefix, image_name)
            img = cvtools.imread(image_file)
            if img is None:
                print('{} is None.'.format(image_file))
                continue

            # crop image
            crop_imgs, starts = self.crop(img)

            # handling the box at the edge of the cropped image
            ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
            img_to_objs, obj_to_num = self.deal_edged_boxes(ann_ids, crop_imgs, starts, iof_th=iof_th)
            if img_to_objs is None:
                continue

            # stats
            for _, num in obj_to_num.items():
                stats['crop_objs'] += num
            stats['total_objs'] += len(ann_ids)
            stats['missing_objs'] += len(set(ann_ids) - set(obj_to_num.keys()))
            for obj in img_to_objs.values():
                if len(obj) > 0:
                    stats['total_croped_images'] += 1

            # save results
            # self.save_crop_labeltxt(image_name, img_to_objs, save_root)

        # save stats values
        total_images = len(roidb)
        stats['total_images'] = len(roidb)
        stats['objs_per_croped_image'] = stats['total_croped_images'] / float(total_images)
        stats['objs_per_image'] = stats['total_objs'] / float(total_images)
        cvtools.save_json(stats, to_file='stats.json')

    # Note： 原始objs不能修改，否则可能影响其它objs
    def save_crop_labeltxt(self, image_name, img_to_objs, save_root):
        crops = []
        img_name_no_suffix, img_suffix = osp.splitext(image_name)
        for crop_i, img_coor in enumerate(img_to_objs):
            crop_objs = img_to_objs[img_coor]
            if len(crop_objs) == 0:
                continue
            # write crop results to txt
            txt_name = '_'.join([img_name_no_suffix] +
                                [str(crop_i)]+list(map(str, img_coor))) + '.txt'
            txt_content = ''
            for crop_obj in crop_objs:
                if len(crop_obj) == 0:
                    continue
                polygen = np.array(crop_obj['segmentation'][0]).reshape(-1, 2)
                polygen = polygen - np.array(img_coor[:2]).reshape(-1, 2)
                line = list(map(str, polygen.reshape(-1)))
                cat = self.COCO.cats[crop_obj['category_id']]['name']
                diffcult = str(crop_obj['difficult'])
                line.append(cat)
                line.append(diffcult)
                txt_content += ' '.join(line) + '\n'
            cvtools.strwrite(txt_content, osp.join(save_root, 'labelTxt+crop', txt_name))
        if len(crops) > 0:
            draw_img = cvtools.draw_boxes_texts(img, crops, line_width=3, box_format='x1y1x2y2')
            cvtools.imwrite(draw_img, osp.join(save_root, 'images', img_name_no_suffix+'.jpg'))

    def crop_for_test(self, save):
        from collections import defaultdict
        imgs = cvtools.get_images_list(self.img_prefix)
        self.test_dataset = defaultdict(list)
        for image_file in tqdm(imgs):
            if cvtools._DEBUG:
                print('crop {}'.format(image_file))
            image_name = osp.basename(image_file)
            img = cvtools.imread(image_file)  # support chinese
            if img is None:
                print('{} is None.'.format(image_file))
                continue
            crop_imgs, starts = self.crop(img)
            for crop_img, start in zip(crop_imgs, starts):
                crop_rect = start[0], start[1], start[0]+crop_img.shape[1], start[1]+crop_img.shape[0]
                self.test_dataset[image_name].append(crop_rect)
        cvtools.save_json(self.test_dataset, save)


if __name__ == '__main__':
    crop_in_order = CropInOder()
    img_file = 'F:/data/rssrai2019_object_detection/val/images/P0060.png'
    img = cv2.imread(img_file)
    img = crop_in_order.crop(img)
