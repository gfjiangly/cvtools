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


# label folder
label_root = 'F:/data/rssrai2019_object_detection/train/labelTxt/'
# imgage folder
image_root = 'F:/data/rssrai2019_object_detection/train/images/'
# what you want to repalece in path string. 
# if not, you can ignore this parameter.
path_replace = {'\\': '/'}
rscup_to_coco = cvtools.Rscup2COCO(label_root, 
                                   image_root, 
                                   path_replace=path_replace, 
                                   box_form='x1y1wh')

rscup_to_coco.convert()

save = 'rscup/train_rscup_x1y1wh_polygen.json'
rscup_to_coco.save_json(save)
```

coco-like dataset analysis
```python
import cvtools


# imgage folder
img_prefix = 'F:/data/rssrai2019_object_detection/train/images'
# position you save in dataset convertion.
ann_file = '../label_convert/rscup/train_rscup_x1y1wh_polygen.json'
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
```