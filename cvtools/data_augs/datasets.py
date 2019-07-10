# -*- coding:utf-8 -*-
# author: gfjiangly
# time: 2019/5/6 18:40
# e-mail: jgf0719@foxmail.com
# software: PyCharm
"""
General Dataset Classes
"""
import random
import torch
import torch.utils.data as data
import cv2
import numpy as np


class AnnotationTransform(object):
    """Transforms a ELEVATOR annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    check annotations

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of ELEVATOR's 2 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self):
        pass

    def __call__(self, target, width, height, ignore_class=False):
        """
        Arguments:
            target (annotation) : the target annotation
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """

        scale = np.array([width, height, width, height]).astype(np.float32)
        res = []
        for box in target:
            box = box.split(',')
            box[:4] = list(map(lambda x: float(x) - 1., box[:4]))
            if box[2] - box[0] < 2. or box[3] - box[1] < 2.:
                continue
            # if box[2] - box[0] < 10. or box[3] - box[1] < 10.:
            #     continue
            box[4] = int(box[4])
            bbox = np.array(box[0:4])/scale
            # if box[2] - box[0] < 0.02 or box[3] - box[1] < 0.02:
            #     continue
            if bbox.any() < 0. or bbox.any() > 1.:
                pass
            bbox[bbox < 0] = 0.
            bbox[bbox > 1] = 1.
            bbox = list(bbox)
            if ignore_class:
                box[4] = 0
            bbox.append(box[4])
            res += [bbox]  # [xmin, ymin, xmax, ymax, label_ind]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class GeneralDataset(data.Dataset):
    """ELEVATOR Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, data_list,
                 transform=None, target_transform=AnnotationTransform(),
                 dataset_name='ELEVATOR'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self.label_files = data_list
        self.ids = list()
        for label_file in self.label_files:
            with open(self.root+label_file, 'r', encoding='gbk') as f:
                self.ids += f.readlines()

    def __getitem__(self, index):
        # im, gt, h, w = self.pull_item(index)
        try:
            # 如果发生异常，就将异常的数据丢弃掉
            im, gt, h, w = self.pull_item(index)
        except:
            # 对于诸如样本损坏或数据集加载异常等情况,就随机取一张图片代替：
            new_index = random.randint(0, len(self) - 1)
            return self[new_index]
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        img_id = img_id.split()
        if len(img_id) > 1:
            target = img_id[1:]
        else:
            print("waring: !image ", img_id[0], "has no box")
            target = None
        img = cv2.imread(img_id[0])     # 如果路径中含中文，读取不成功，img为None
        if img is None:
            print("waring: !can't read image ", img_id[0])
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height, ignore_class=False)   # 须实现__call__方法

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(img_id.split()[0], cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        return img_id

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
