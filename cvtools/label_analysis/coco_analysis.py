# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/6/24 9:57
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import copy
import os.path as osp
import numpy as np
from collections import defaultdict

import cvtools
from .size_analysis import SizeAnalysis


class COCOAnalysis(object):
    """coco-like datasets analysis"""
    def __init__(self, img_prefix, ann_file=None):
        self.img_prefix = img_prefix
        if ann_file is not None:
            self.ann_file = ann_file
            self.coco_dataset = cvtools.load_json(ann_file)
            self.COCO = cvtools.COCO(ann_file)
            # 组合优于继承
            self.size_analysis = SizeAnalysis(self.COCO)
            self.catToAnns = defaultdict(list)
            if 'annotations' in self.coco_dataset:
                for ann in self.coco_dataset['annotations']:
                    self.catToAnns[ann['category_id']].append(ann)

    def stats_size_per_cat(self, to_file='size_per_cat_data.json'):
        self.size_analysis.stats_size_per_cat(to_file=to_file)

    def stats_objs_per_img(self, to_file='stats_num.json'):
        self.size_analysis.stats_objs_per_img(to_file=to_file)

    def stats_objs_per_cat(self, to_file='objs_per_cat_data.json'):
        self.size_analysis.stats_objs_per_cat(to_file=to_file)

    def get_weights_for_balanced_classes(self, to_file='weighted_samples.pkl'):
        weights = self.size_analysis.get_weights_for_balanced_classes(to_file)
        return weights

    # TODO: plan to fix
    def cluster_analysis(self,
                         save_root,
                         name_clusters=('bbox', ),
                         n_clusters=(3,),
                         by_cat=False):
        if by_cat:
            self._cluster_by_cat(save_root, name_clusters, n_clusters)
        assert len(name_clusters) == len(n_clusters)
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        print('roidb: {}'.format(len(roidb)))
        cluster_dict = defaultdict(list)
        for entry in roidb:
            ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
            objs = self.COCO.loadAnns(ann_ids)
            # Sanitize bboxes -- some are invalid
            for obj in objs:
                if 'ignore' in obj and obj['ignore'] == 1:
                    continue
                if 'area' in name_clusters:
                    cluster_dict['area'].append(obj['area'])
                if 'w-vs-h' in name_clusters:
                    cluster_dict['w-vs-h'].append(
                        obj['bbox'][2] / float(obj['bbox'][3]))
        cvtools.makedirs(save_root)
        print('start cluster analysis...')
        for i, cluster_name in enumerate(cluster_dict.keys()):
            cluster_value = cluster_dict[cluster_name]
            assert len(cluster_value) >= n_clusters[i]
            value_arr = np.array(cluster_value)
            percent = np.percentile(value_arr, [1, 50, 99])
            value_arr = value_arr[percent[2] > value_arr]
            cvtools.draw_hist(value_arr, bins=1000,
                              x_label=cluster_name,
                              y_label="Quantity",
                              title=cluster_name, density=False,
                              save_name=osp.join(save_root, cluster_name+'.png'))
            cluster_value = np.array(value_arr).reshape(-1, 1)
            cluster_value_centers = cvtools.DBSCAN_cluster(cluster_value,
                                                           metric='manhattan')
            np.savetxt(osp.join(save_root, cluster_name+'.txt'),
                       np.around(cluster_value_centers, decimals=0))
        print('cluster analysis finished!')

    def _cluster_by_cat(self,
                        save_root,
                        name_clusters=('bbox', ),
                        n_clusters=(3,)):
        assert len(name_clusters) == len(n_clusters)
        cluster_dict = defaultdict(lambda: defaultdict(list))
        for key, ann in self.COCO.anns.items():
            cat_name = self.COCO.cats[ann['category_id']]['name']
            if 'area' in name_clusters:
                cluster_dict[cat_name]['area'].append(ann['area'])
            if 'w-vs-h' in name_clusters:
                cluster_dict[cat_name]['w-vs-h'].append(
                    ann['bbox'][2] / float(ann['bbox'][3]))
        cvtools.makedirs(save_root)
        for cat_name, cluster_value in cluster_dict.items():
            cluster_values = cluster_dict[cat_name]
            cluster_results = defaultdict(lambda: defaultdict(list))
            for i, cluster_name in enumerate(cluster_values.keys()):
                if len(cluster_value) < n_clusters[i]:
                    continue
                centers = cvtools.k_means_cluster(
                    np.array(cluster_value).reshape(-1, 1),
                    n_clusters=n_clusters[i])
                cluster_results[cluster_name][cat_name].append(
                    list(centers.reshape(-1)))
            cvtools.save_json(
                cluster_results,
                osp.join(save_root, 'cluster_{}.json'.format(cat_name))
            )

    def read_img_or_crop(self, entry):
        image_name = entry['file_name']
        image_file = osp.join(self.img_prefix, image_name)
        try:
            img = cvtools.imread(image_file)
        except FileNotFoundError:
            print('image {} is not found!'.format(image_file))
            img = None
        image_name = osp.splitext(image_name)[0]
        if 'crop' in entry:
            img = img[entry['crop'][1]:entry['crop'][3]+1,
                  entry['crop'][0]:entry['crop'][2]+1]
            image_name = '_'.join([image_name] + list(map(str, entry['crop'])))
        return img, image_name

    def vis_objs(self, img, objs, vis='bbox', box_format='x1y1wh'):
        for obj in objs:
            vis_obj = []
            if 'ignore' in obj and obj['ignore'] == 1: continue
            if vis in obj: vis_obj = obj[vis]
            class_name = self.COCO.cats[obj['category_id']]['name'] \
                if 'category_id' in obj else ''
            img = cvtools.draw_boxes_texts(
                img, vis_obj, class_name, box_format=box_format)
        return img

    def vis_instances(self,
                      save_root,
                      vis='bbox',   # or segm
                      vis_cats=None,
                      output_by_cat=False,
                      box_format='x1y1wh'):
        """Visualise bbox and polygon in annotation.

        包含某一类的图片上所有类别均会被绘制。

        Args:
            save_root (str): path for saving image.
            vis (str): 'bbox' or 'segmentation'
            vis_cats (list): categories to be visualized
            output_by_cat (bool): output visual images by category.
            box_format (str): 'x1y1wh' or 'polygon'
        """
        assert vis in ('bbox', 'segmentation')
        assert box_format in ('x1y1wh', 'polygon')
        if vis_cats is not None or output_by_cat:
            self._vis_instances_by_cat(
                save_root, vis=vis, vis_cats=vis_cats, box_format=box_format)
            return
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        if cvtools._DEBUG:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))[:10]
        else:
            roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        print('{} images.'.format(len(roidb)))
        cvtools.makedirs(save_root)
        for i, entry in enumerate(roidb):
            print('Visualize image %d of %d: %s' %
                  (i, len(roidb), entry['file_name']))
            img, image_name = self.read_img_or_crop(entry)
            if img is None:
                print('{} is None.'.format(image_name))
                continue
            ann_ids = self.COCO.getAnnIds(imgIds=entry['id'],
                                          iscrowd=None)
            objs = self.COCO.loadAnns(ann_ids)
            if len(objs) == 0: continue
            # Sanitize bboxes -- some are invalid
            img = self.vis_objs(img, objs, vis=vis, box_format=box_format)
            cvtools.imwrite(img, osp.join(save_root, image_name + '.jpg'))

    def _vis_instances_by_cat(self,
                              save_root,
                              vis_cats=None,
                              vis='bbox',
                              box_format='x1y1wh'):
        """Visualise bbox and polygon in annotation by categories.

        包含某一类的图片上所有类别均会被绘制。

        Args:
            save_root (str): path for saving image.
            vis (str): 'bbox' or 'segmentation'
            vis_cats (list): categories to be visualized
            box_format (str): 'x1y1wh' or 'polygon'
        """
        catImgs = copy.deepcopy(self.COCO.catToImgs)
        catImgs = {cat: set(catImgs[cat]) for cat in catImgs}
        for cat_id, image_ids in catImgs.items():
            cat_name = self.COCO.cats[cat_id]['name']
            if vis_cats is not None and cat_name not in vis_cats:
                continue
            print('Visualize %s' % cat_name)
            roidb = self.COCO.loadImgs(image_ids)   # 不会修改原始数据
            if cvtools._DEBUG:
                roidb = roidb[:10]
            for i, entry in enumerate(roidb):
                print('Visualize image %d of %d: %s' %
                      (i, len(roidb), entry['file_name']))
                img, image_name = self.read_img_or_crop(entry)
                if img is None:
                    print('{} is None.'.format(image_name))
                    continue
                ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
                objs = self.COCO.loadAnns(ann_ids)
                img = self.vis_objs(img, objs, vis=vis, box_format=box_format)
                # save in jpg format for saving storage
                cvtools.imwrite(
                    img, osp.join(save_root, cat_name, image_name + '.jpg'))
