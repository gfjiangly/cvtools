cvtools
=======
Computer Vision Tool Library

Introduction
------------

cvtools is a helpful python library for computer vision.

It provides the following functionalities.

- Universal IO APIs
- Dataset Conversion(voc to coco, bdd to coco, ...)
- Dataset Analysis(visualization, cluster analysis, ...)
- Data Augmentation(random mirror, random sample crop, ...)
- Image processing(crop, resize, ...)
- Useful utilities (iou, timer, ...)


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
convert rscup competition dataset to coco dataset format.
```python
import cvtools


# convert dota dataset to coco dataset format
# label folder
label_root = 'F:/data/DOTA/train/labelTxt/'
# imgage folder
image_root = 'F:/data/DOTA/train/images/'
# what you want to repalece in path string.
# if not, you can ignore this parameter.
path_replace = {'\\': '/'}
dota_to_coco = cvtools.DOTA2COCO(label_root,
                                 image_root,
                                 path_replace=path_replace,
                                 box_form='x1y1wh')

dota_to_coco.convert()

save = 'dota/train_dota_x1y1wh_polygen.json'
dota_to_coco.save_json(save)
```

coco-like dataset analysis
```python
import cvtools


# imgage folder
img_prefix = 'F:/data/DOTA/train/images'
# position you save in dataset convertion.
ann_file = '../label_convert/dota/train_dota_x1y1wh_polygen.json'
coco_analysis = cvtools.COCOAnalysis(img_prefix, ann_file)

save = 'dota/vis_dota_whole/'
coco_analysis.vis_instances(save, 
                            vis='segmentation', 
                            box_format='x1y1x2y2x3y3x4y4')

save = 'dota/class_distribution/class_distribution.txt'
coco_analysis.stats_class_distribution(save)

save = 'dota/bbox_distribution/'
coco_analysis.cluster_analysis(save, name_clusters=('area', ))

# and so on...
```