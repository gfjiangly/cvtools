#import torch
import cv2
import numpy as np
import types
from numpy import random

import cvtools


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


# crop用到
def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels


class ResizeFilled(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        h, w, _ = image.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))  # h, w, c
        # Add padding
        image = np.pad(image, pad, 'constant', constant_values=0.)
        padded_h, padded_w, _ = image.shape
        # Resize
        image = cv2.resize(image, (self.size, self.size))
        # cv2.imwrite("temp.jpg", image+(104, 117, 123))
        boxes[:, [0, 2]] = (boxes[:, [0, 2]]*w + pad[1][0]) / float(padded_w)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]]*h + pad[0][0]) / float(padded_h)

        # for box in boxes:
        #     x1, y1, x2, y2 = box*300
        #     x1 = max(0, np.floor(x1 + 0.5).astype('int32'))
        #     y1 = max(0, np.floor(y1 + 0.5).astype('int32'))
        #     x2 = min(image.shape[1], np.floor(x2 + 0.5).astype('int32'))    # 这里image是ndarray，size是标量，shape才是向量
        #     y2 = min(image.shape[0], np.floor(y2 + 0.5).astype('int32'))
        #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # import time
        # cv2.imwrite("results/"+time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))+".jpg", image)
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


# 随机对图像每个像素添加一个值，
# 该添加值是从 [-delat, delta] 中随机选取的. 默认的 delta 值是 32.
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            # image += delta
            image = image.astype(np.int)
            np.add(image, delta, out=image, casting='unsafe')
            image = np.clip(image, 0, 255)
            image = image.astype(np.uint8)
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


# resize后，相当于局部放大了图像
# 此函数确保该图像块至少与一个 groundtruth box 有重叠，
# 至少一个 gt box 的中心位于该图像块中.
# 这样可以避免不包含明显的前景目标的图像块用于网络训练
class RandomSampleCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():     # 目前只对min做了限制，0.1, 0.3, 0.7, 0.9
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


# resize后，相当于整体缩小了图像
class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness(delta=32)
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return im, boxes, labels
        # return self.rand_light_noise(im, boxes, labels)


class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):     # mean值应该通过对自己的数据集聚类得到，但是似乎影响不大，暂时没修改
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),      # int->np.float32
            ToAbsoluteCoords(),     # Absolute Coords
            PhotometricDistort(),   # 光度变形
            Expand(self.mean),      # 概率图像扩展
            RandomSampleCrop(),     # 随机裁剪
            RandomMirror(),         # 随机镜像
            ToPercentCoords(),      # [0, 1] Relative Coords
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class RandomMirror(object):
    def __call__(self, image, boxes, both=True):
        # _, width, _ = image.shape
        # if random.randint(2):
        #     image = image[:, ::-1]
        #     boxes = boxes.copy()
        #     boxes[:, 0::2] = width - boxes[:, 2::-2]
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)     # 返回的数据类型一致性
        # 如果同时触发水平镜像和竖直镜像就等价于旋转180度
        flip_code = random.randint(4)
        if not both:
            flip_code = random.randint(3)
        if flip_code == 1:
            image, boxes = horizontal_mirror(image, boxes)
        elif flip_code == 2:
            image, boxes = vertical_mirror(image, boxes)
        elif flip_code == 3:
            image, boxes = horizontal_mirror(image, boxes)
            image, boxes = vertical_mirror(image, boxes)
        return image, boxes


def horizontal_mirror(image, boxes):
    """水平方向（flipping around the y-axis）镜像"""
    _, width, _ = image.shape
    image = cv2.flip(image, 1)  # 不修改原图，image内存连续
    # image = image[:, ::-1]    # image不内存连续
    boxes = boxes.copy()
    box_coor_len = boxes.shape[1]
    if box_coor_len == 4:   # 对角线形式，一般而言认为第一个点是左上点
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    elif box_coor_len == 8:  # polygon形式不改变点的相邻关系
        boxes[:, 0::2] = width - boxes[:, 0::2]
    else:
        raise NotImplementedError
    return image, boxes


class RandomHorMirror(object):
    """水平方向（flipping around the y-axis）镜像"""
    def __call__(self, image, boxes, r=True):
        """image参数被原位修改，boxes参数不会被修改"""
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)     # 返回的数据类型一致性
        if not r or random.randint(2):
            image, boxes = horizontal_mirror(image, boxes)
        return image, boxes


def vertical_mirror(image, boxes):
    height, _, _ = image.shape
    image = cv2.flip(image, 0)  # 不修改原图，image内存连续
    # image = image[::-1]  # image不内存连续
    box_coor_len = boxes.shape[1]
    boxes = boxes.copy()
    if box_coor_len == 4:
        boxes[:, 1::2] = height - boxes[:, 3::-2]
    elif box_coor_len == 8:
        boxes[:, 1::2] = height - boxes[:, 1::2]
    else:
        raise NotImplementedError
    return image, boxes


class RandomVerMirror(object):
    """竖直方向（flipping around the x-axis）镜像"""
    def __call__(self, image, boxes):
        """image参数被原位修改，boxes参数不会被修改"""
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)     # 返回的数据类型一致性
        if random.randint(2):
            image, boxes = vertical_mirror(image, boxes)
        return image, boxes


def rotate_90(img, bboxes=None):
    """顺时针旋转90度

        bboxes支持的格式
            两点式: x1y1x2y2,
            四点式：x1y1x2y2x3y3x4y4

    Args:
        img(np.ndarray): the format of opencv image
        bboxes(list or np.ndarray): supported format
            two points: x1y1x2y2,
            four points: x1y1x2y2x3y3x4y4
    """
    h, w, _ = img.shape
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 1)
    if bboxes is None:
        return new_img
    else:
        # bounding box 的变换: 一个图像的宽高是W,H,
        # 如果顺时90度转换，那么原来的原点(0, 0)到了 (H, 0) 这个最右边的顶点了，
        # 设图像中任何一个转换前的点(x1, y1), 转换后，x1, y1是表示到 (H, 0)这个点的距离，
        # 所以我们只要转换回到(0, 0) 这个点的距离即可！
        # 所以+90度转换后的点为 (H-y1, x1), -90度转换后的点为(y1, W-x1)
        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes)
        else:
            bboxes = bboxes.copy()
        box_coor_len = bboxes.shape[1]
        x_axis = range(0, box_coor_len, 2)
        y_axis = range(1, box_coor_len, 2)
        ori_axis = range(box_coor_len)
        new_axis = cvtools.concat_list(zip(y_axis, x_axis))
        bboxes[:, ori_axis] = bboxes[:, new_axis]
        bboxes[:, x_axis] = h - bboxes[:, x_axis]
        return new_img, bboxes


def rotate_270(img, bboxes=None):
    """逆时针旋转90度

        bboxes支持的格式
            两点式: x1y1x2y2,
            四点式：x1y1x2y2x3y3x4y4

    Args:
        img(np.ndarray): the format of opencv image
        bboxes(list or np.ndarray): supported format
            two points: x1y1x2y2,
            four points: x1y1x2y2x3y3x4y4
    """
    h, w, _ = img.shape
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 0)
    if bboxes is None:
        return new_img
    else:
        # bounding box 的变换: 一个图像的宽高是W,H,
        # 如果顺时90度转换，那么原来的原点(0, 0)到了 (H, 0) 这个最右边的顶点了，
        # 设图像中任何一个转换前的点(x1, y1), 转换后，x1, y1是表示到 (H, 0)这个点的距离，
        # 所以我们只要转换回到(0, 0) 这个点的距离即可！
        # 所以+90度转换后的点为 (H-y1, x1), -90度转换后的点为(y1, W-x1)
        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes)
        else:
            bboxes = bboxes.copy()
        box_coor_len = bboxes.shape[1]
        x_axis = range(0, box_coor_len, 2)
        y_axis = range(1, box_coor_len, 2)
        ori_axis = range(box_coor_len)
        new_axis = cvtools.concat_list(zip(y_axis, x_axis))
        bboxes[:, ori_axis] = bboxes[:, new_axis]
        bboxes[:, y_axis] = w - bboxes[:, y_axis]
        return new_img, bboxes


def rotate_180(img, bboxes=None):
    """顺时针旋转180度

        bboxes支持的格式
            两点式: x1y1x2y2,
            四点式：x1y1x2y2x3y3x4y4

    Args:
        img(np.ndarray): the format of opencv image
        bboxes(list or np.ndarray): supported format
            two points: x1y1x2y2,
            four points: x1y1x2y2x3y3x4y4
    """
    h, w, _ = img.shape
    new_img = cv2.flip(img, -1)
    if bboxes is None:
        return new_img
    else:
        # bounding box 的变换: 一个图像的宽高是W,H,
        # 如果顺时90度转换，那么原来的原点(0, 0)到了 (H, 0) 这个最右边的顶点了，
        # 设图像中任何一个转换前的点(x1, y1), 转换后，x1, y1是表示到 (H, 0)这个点的距离，
        # 所以我们只要转换回到(0, 0) 这个点的距离即可！
        # 所以+90度转换后的点为 (H-y1, x1), -90度转换后的点为(y1, W-x1)
        if not isinstance(bboxes, np.ndarray):
            bboxes = np.array(bboxes)
        else:
            bboxes = bboxes.copy()
        box_coor_len = bboxes.shape[1]
        x_axis = list(range(0, box_coor_len, 2))
        y_axis = list(range(1, box_coor_len, 2))
        bboxes[:, y_axis] = h - bboxes[:, y_axis]
        bboxes[:, x_axis] = w - bboxes[:, x_axis]
        return new_img, bboxes


class RandomRotate(object):
    """随机旋转0度、90度、180度、270度
    """
    def __call__(self, image, boxes, rotate=None):
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)
        if not rotate:
            rotate = random.choice([None, 'rotate_90', 'rotate_180', 'rotate_270'])
        if rotate:
            image, boxes = eval(rotate)(image, boxes)
        return image, boxes
