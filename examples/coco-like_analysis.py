# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/7/29 9:22
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import cvtools


# imgage folder
img_prefix = 'F:/data/rssrai2019_object_detection/train/images'
# position you save in dataset convertion.
ann_file = '../label_convert/rscup/train_rscup_x1y1wh_polygen.json'
coco_analysis = cvtools.COCOAnalysis(img_prefix, ann_file)

coco_analysis.vis_boxes('rscup/vis_rscup_whole/', vis='segmentation', box_format='x1y1x2y2x3y3x4y4')

coco_analysis.vis_boxes_by_cat('rscup/vis_rscup/', vis_cats=('helipad', ),
                                vis='segmentation', box_format='x1y1x2y2x3y3x4y4')

coco_analysis.stats_class_distribution('rscup/class_distribution/class_distribution.txt')

coco_analysis.cluster_analysis('rscup/bbox_distribution/', cluster_names=('area', ))

# and so on...
