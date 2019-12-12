## 数据集格式转换


### VOC转COCO


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


### DOTA转COCO

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