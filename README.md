# cvtools
Computer Vision Tool Codes

# install
```bash
pip install cvtoolss
```

# example
convert rscup compitition dataset to coco dataset format.
```bash
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
```

coco-like dataset analysis
```bash
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

```