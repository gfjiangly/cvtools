# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/7/29 9:22
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import cvtools


# imgage folder
img_prefix = 'F:/data/rssrai2019_object_detection/train/images'
# position you save in dataset convertion.
ann_file = '../label_convert/rscup/train_dota_x1y1wh_polygen.json'
coco_analysis = cvtools.COCOAnalysis(img_prefix, ann_file)

save = 'rscup/vis_rscup_whole/'
coco_analysis.vis_instances(save,
                            vis='segmentation',
                            box_format='x1y1x2y2x3y3x4y4')

save = 'rscup/class_distribution/class_distribution.txt'
coco_analysis.stats_class_distribution(save)

save = 'rscup/bbox_distribution/'
coco_analysis.cluster_analysis(save, name_clusters=('area', ))

# and so on...
