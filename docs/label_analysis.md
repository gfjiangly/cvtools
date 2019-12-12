## 标签分析

目前仅提供COCO格式标签分析，其它格式数据集需先转为COCO格式（或与COCO兼容的格式）才能使用此库分析。


### COCO格式标签分析

加载COCO格式标签

```python
import cvtools


# imgage folder
img_prefix = '/media/data/DOTA/train/images'
# position you save in dataset convertion.
ann_file = '../label_convert/dota/train_dota_x1y1wh_polygen.json'
coco_analysis = cvtools.COCOAnalysis(img_prefix, ann_file)

```

可视化标签，支持绘制bbox和segmentation，可以指定bbox格式。
bbox格式支持：

- x1y1wh(默认)
- polygon(segmentation模式使用)

```python

save = 'dota/vis_dota_whole/'
coco_analysis.vis_instances(save, 
                            vis='segmentation', 
                            box_format='x1y1x2y2x3y3x4y4')

```

统计每个类别不同size占比和数量，size定义同COCO
```python

# Size distribution analysis for each category
save = 'size_per_cat_data.json'
coco_analysis.stats_size_per_cat(save)

# Average number of targets per image for each category
save = 'stats_num.json'
coco_analysis.stats_objs_per_img(save)

# Analysis of target quantity per category
save = 'objs_per_cat_data.json'
coco_analysis.stats_objs_per_cat(save)

```


统计每个类别单张图平均有多少实例数，统计维度是图片
```python

# Average number of targets per image for each category
save = 'stats_num.json'
coco_analysis.stats_objs_per_img(save)

# Analysis of target quantity per category
save = 'objs_per_cat_data.json'
coco_analysis.stats_objs_per_cat(save)

```


统计每个类别有多少实例数，统计维度是类别
```python

# Analysis of target quantity per category
save = 'objs_per_cat_data.json'
coco_analysis.stats_objs_per_cat(save)

```