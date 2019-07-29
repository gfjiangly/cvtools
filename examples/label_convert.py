# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/7/10 17:13
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import cvtools


# label folder
label_root = 'F:/data/rssrai2019_object_detection/train/labelTxt/'
# imgage folder
image_root = 'F:/data/rssrai2019_object_detection/train/images/'
# what you want to repalece in path string. if not, you can ignore this parameter.
path_replace = {'\\': '/'}
rscup_to_coco = cvtools.Rscup2COCO(label_root, image_root, path_replace=path_replace, box_form='x1y1wh')
rscup_to_coco.convert()
rscup_to_coco.save_json('rscup/train_rscup_x1y1wh_polygen.json')
