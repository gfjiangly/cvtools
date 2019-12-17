cvtools
=======

[![Travis CI Build](https://travis-ci.com/gfjiangly/cvtools.svg?branch=master)](https://travis-ci.com/gfjiangly/cvtools)
[![PyPI - Python Version](https://img.shields.io/pypi/v/cvtoolss)](https://pypi.org/project/cvtoolss)
[![Documentation Status](https://readthedocs.org/projects/cvtools/badge/?version=latest)](https://cvtools.readthedocs.io/zh/latest/?badge=latest)
[![GitHub - License](https://img.shields.io/github/license/open-mmlab/mmcv.svg)](https://github.com/gfjiangly/cvtools/blob/master/LICENSE)

Computer Vision Tool Library

Introduction
------------

cvtools is a helpful python library for computer vision.

It provides the following functionalities.

- Dataset Conversion(voc to coco, bdd to coco, ...)
- Data Augmentation(random mirror, random sample crop, ...)
- Dataset Analysis(visualization, cluster analysis, ...)
- Image processing(crop, resize, ...)
- Useful utilities (iou, timer, ...)
- Universal IO APIs

See the [documentation](https://cvtools.readthedocs.io/zh/latest) for more features and usage.

Installation
------------
Try and start with
```bash
pip install cvtoolss
```
Note: There are two s at the end.

or install from source
```bash
git clone https://github.com/gfjiangly/cvtools.git
cd cvtools
pip install .  # (add "-e" if you want to develop or modify the codes)
```


example
-------
convert voc-like dataset to coco-like dataset
```python
import cvtools


mode = 'train'
root = 'D:/data/VOCdevkit/VOC2007'
# The cls parameter is a file containing categories,
# one category string is one line
voc_to_coco = cvtools.VOC2COCO(root, mode=mode,
                               cls='voc/cls.txt')
voc_to_coco.convert()
voc_to_coco.save_json(to_file='voc/{}.json'.format(mode))

```
convert dota dataset to coco-like dataset.
```python
import cvtools


# convert dota dataset to coco dataset format
# label folder
label_root = '/media/data/DOTA/train/labelTxt/'
# imgage folder
image_root = '/media/data/DOTA/train/images/'

dota_to_coco = cvtools.DOTA2COCO(label_root, image_root)

dota_to_coco.convert()

save = 'dota/train_dota_x1y1wh_polygen.json'
dota_to_coco.save_json(save)
```

coco-like dataset analysis
```python
import cvtools


# imgage folder
img_prefix = '/media/data/DOTA/train/images'
# position you save in dataset convertion.
ann_file = '../label_convert/dota/train_dota_x1y1wh_polygen.json'
coco_analysis = cvtools.COCOAnalysis(img_prefix, ann_file)

save = 'dota/vis_dota_whole/'
coco_analysis.vis_instances(save, 
                            vis='segmentation', 
                            box_format='polygon')

# Size distribution analysis for each category
save = 'size_per_cat_data.json'
coco_analysis.stats_size_per_cat(save)

# Average number of targets per image for each category
save = 'stats_num.json'
coco_analysis.stats_objs_per_img(save)

# Analysis of target quantity per category
save = 'objs_per_cat_data.json'
coco_analysis.stats_objs_per_cat(save)

save = 'dota/bbox_distribution/'
coco_analysis.cluster_analysis(save, name_clusters=('area', ))

# and so on...
```