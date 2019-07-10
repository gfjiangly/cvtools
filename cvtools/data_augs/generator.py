# -*- coding:utf-8 -*-
# author: gfjiangly
# time: 2019/5/6 18:34
# e-mail: jgf0719@foxmail.com
# software: PyCharm

import os
import cv2
from tqdm import tqdm
from multiprocessing import Pool

from augmentations import *
from utils.file_utils import read_files_to_list, makedirs
from utils.image_utils import draw_rect_test_labels


class DataAugmentation(object):
    def __init__(self, mean=(127, 127, 127)):     # mean值应该通过对自己的数据集聚类得到，但是似乎影响不大，暂时没修改
        self.mean = mean
        self.augment = Compose([
            ConvertFromInts(),      # int->np.float32, for image
            PhotometricDistort(),   # 光度变形
            Expand(self.mean),      # 概率图像扩展
            RandomSampleCrop()      # 随机裁剪
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


def augment_images():
    # root = 'labels/test/'
    # data_list = ['elevator_20181230_convert_test.txt']
    root = 'labels/train/'
    data_list = ['elevator_20181230_convert_train.txt', 'elevator_20181231_convert_train.txt',
                 'elevator_20190106_convert_train.txt', 'person_7421_train.txt']
    dataset = read_files_to_list(root, data_list)
    transfer = DataAugmentation()
    save_path = 'generate_image2/'
    makedirs(save_path)
    annts_lines = ''

    # pool = Pool(processes=3)  # 创建进程池，指定最大并发进程数

    for line in tqdm(dataset):
        def augment_one_image(line):
            nonlocal annts_lines
            line = line.strip().split()
            image_name = line[0]
            boxes = []
            classes = []
            for label_str in line[1:]:
                bbox_cls_str = label_str.split(',')
                boxes.append([float(i) for i in bbox_cls_str][0:4])
                classes.append(int(bbox_cls_str[4]))
            boxes = np.array(boxes)
            classes = np.array(classes)
            new = save_path + image_name.replace('\\', '/').replace('.jpg', '').split('/')[-1] + '_{index}.jpg'
            for im_index in range(1, 5):    # 每张图片增强出4张
                new_image_name = new.format(index=im_index)
                if not os.path.isfile(new_image_name):
                    # im = cv2.imread(image_name)  # BGR 如果路径中含中文，读取不成功，img为None
                    im = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), cv2.IMREAD_COLOR)
                    img, boxes_trans, classes_trans = transfer(im, boxes, classes)
                    boxes_trans = boxes_trans.astype(np.int32)
                    classes_trans = classes_trans.astype(np.int32)
                    # print('save %s...' % new_image_name)
                    cv2.imwrite(new_image_name, img)
                    annts_lines += new_image_name + ' '
                    for box, cls in zip(boxes_trans, classes_trans):
                        annts_lines += ','.join(map(str, box)) + ',' + str(cls) + ' '
                    annts_lines += '\n'
        # pool.apply_async(augment_one_image, args=(line,))
        augment_one_image(line)
    # pool.close()  # 关闭进程池，阻止更多的任务提交到进程池Pool
    # pool.join()  # 主进程等待进程池中的进程执行完毕，回收进程池

    new_annots = 'labels/gen/gen_annots.txt'
    print('save %s...' % new_annots)
    with open(new_annots, 'w') as f:
        f.write(annts_lines)

    print('draw boxes in images...')
    draw_rect_test_labels(new_annots, 'temp/')


if __name__ == '__main__':
    augment_images()

