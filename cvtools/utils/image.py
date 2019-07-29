# -*- coding:utf-8 -*-
# author: gfjiangly
# time: 2019/5/6 18:55
# e-mail: jgf0719@foxmail.com
# software: PyCharm

import os
import os.path as osp
import sys
import numpy as np
import cv2.cv2 as cv
from PIL import Image, ImageDraw
import copy
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm

import cvtools


from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED
}


# 使用PIL lazy方式读图像，防止读大图像死机; 支持中文路径
def imread(img_or_path, flag='color'):
    """Read an image.

    Args:
        img_or_path (ndarray or str): Either a numpy array or image path.
            If it is a numpy array (loaded image), then it will be returned
            as is.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.

    Returns:
        ndarray: Loaded image array.
    """
    if isinstance(img_or_path, np.ndarray):
        return img_or_path
    elif isinstance(img_or_path, str):
        flag = imread_flags[flag] if isinstance(flag, str) else flag
        cvtools.check_file_exist(img_or_path, 'img file does not exist: {}'.format(img_or_path))
        try:
            "PIL: Open an image file, without loading the raster data"
            Image.open(img_or_path)
            # im = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), cv2.IMREAD_COLOR)
        except (FileNotFoundError, Image.DecompressionBombError) as e:
            print(e)
            return None
        return cv.imdecode(np.fromfile(img_or_path, dtype=np.uint8), flag)
    else:
        raise TypeError('"img" must be a numpy array or a filename')


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError('"img" must be a numpy array!')
    if auto_mkdir:
        cvtools.makedirs(file_path)
    # return cv.imwrite(file_path, img, params)
    # support path included chinese
    return cv.imencode(osp.splitext(file_path)[-1], img, params)[1].tofile(file_path)


# detect_line: out_boxes, out_scores, out_classes
def save_rect_image(image_path, detect_line, th_h=0.9, th_l=0.5):
    image = Image.open(image_path)
    path_h = os.getcwd()+'/results/crop/' + str(th_h) + '/'
    path_l = os.getcwd()+'/results/crop/' + str(th_l) + '/'
    out_boxes, out_scores, out_classes = detect_line
    if not os.path.exists(path_h):
        os.makedirs(path_h, mode=0o777)
    if not os.path.exists(path_l):
        os.makedirs(path_l, mode=0o777)
    _, file_name = os.path.split(image_path)
    filename, _ = os.path.splitext(file_name)
    for i, c in reversed(list(enumerate(out_classes))):
        box = out_boxes[i]
        top, left, bottom, right = box
        img = image.crop((left, top, right, bottom))
        score = out_scores[i]
        if score > th_h:
            img.save(path_h+'_'.join([filename, str(score), str(i)])+'.jpg')
        elif score > th_l:
            img.save(path_l+'_'.join([filename, str(score), str(i)])+'.jpg')


# 批量绘制矩形框，src为txt标签文件,dst为图片输出目录
def draw_rect_test_labels(src, dst, first=sys.maxsize):
    colour = ['blue', 'red', 'green', 'white']
    if not os.path.exists(dst):
        os.mkdir(dst)
    with open(src, 'r') as fp:
        for count, line in enumerate(fp.readlines()):
            if count < first:
                line = line.strip().split()
                im = Image.open(line[0])
                draw = ImageDraw.Draw(im)
                boxes = line[1:]
                for box in boxes:
                    bbox_coordinates_str = box.split(',')
                    coordinates = [float(i) for i in bbox_coordinates_str][0:4]
                    draw.rectangle(coordinates, None, colour[int(bbox_coordinates_str[4])])
                img_name = line[0].replace('\\', '/').split('/')[-1]
                im.save(dst+img_name)


def draw_boxes_texts(img, boxes, texts=None, colors=None, line_width=1, draw_start=True,
                     box_format='x1y1x2y2'):
    """support box format: x1y1x2y2(default), x1y1wh, xywh, xywha, x1y1x2y2x3y3x4y4"""
    if len(boxes) == 0:
        return img
    boxes = copy.deepcopy(boxes)
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
        if box_format != 'xywha':
            boxes = boxes.astype(np.int)
        if len(boxes.shape) == 1:
            boxes = [boxes]
    else:
        boxes = boxes.astype(np.int)
    if texts is not None and not isinstance(texts, (list, np.ndarray)):
        texts = [texts]
    if isinstance(img, Image.Image):
        img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    if not isinstance(img, np.ndarray):
        return
    # if colors is None:
    #     colors = [(255, 0, 0), (0, 255, 255)]
    text_color = (0, 255, 255)
    thickness = line_width
    font = cv.FONT_HERSHEY_SIMPLEX
    for idx, box in enumerate(boxes):
        box_color = (0, 0, 255) if colors is None else colors[idx]  # default color: red, BGR order
        if box_format == 'x1y1x2y2':
            cv.rectangle(img, tuple(box[0:2]), tuple(box[2:4]), box_color, thickness)
        elif box_format == 'x1y1wh':
            box[0:4] = cvtools.x1y1wh_to_x1y1x2y2(list(box[0:4]))
            cv.rectangle(img, tuple(box[0:2]), tuple(box[2:4]), box_color, thickness)
            pass
        elif box_format == 'xywh':
            box[0:4] = cvtools.xywh_to_x1y1x2y2(list(box[0:4]))
            cv.rectangle(img, tuple(box[0:2]), tuple(box[2:4]), box_color, thickness)
        elif box_format == 'xywha':
            rrect = tuple(box[:2]), tuple(box[2:4]), box[4]
            box = cv.boxPoints(rrect).astype(np.int)
            # box = np.int0(box)
            cv.drawContours(img, [box], 0, box_color, thickness)
            box = box.reshape((-1,))
        elif box_format == 'x1y1x2y2x3y3x4y4':
            cv.line(img, tuple(box[:2]), tuple(box[2:4]), box_color, thickness)
            cv.line(img, tuple(box[2:4]), tuple(box[4:6]), box_color, thickness)
            cv.line(img, tuple(box[4:6]), tuple(box[6:8]), box_color, thickness)
            cv.line(img, tuple(box[6:]), tuple(box[:2]), box_color, thickness)
        else:
            raise RuntimeError('not supported box format!')
        if draw_start:
            cv.circle(img, tuple(box[:2]), radius=5, color=text_color, thickness=-1)
        if texts is not None:
            cv.putText(img, texts[idx], (box[0]+2, box[1]-2), font, 0.5, text_color, 1)
    return img


def draw_class_distribution(y, save_name='class_distribution.png'):
    """绘制饼图,其中y是标签列表
    """
    # 设置matplotlib正常显示中文和负号
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    from collections import Counter
    target_stats = Counter(y)
    labels = list(target_stats.keys())
    sizes = list(target_stats.values())
    explode = tuple([0.1] * len(target_stats))
    fig, axes = plt.subplots(figsize=(10, 5), ncols=2)
    ax1, ax2 = axes.ravel()
    patches, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.0f%%',
                                        shadow=False, startangle=170)
    ax1.axis('equal')
    # 重新设置字体大小
    proptease = fm.FontProperties()
    proptease.set_size('xx-small')
    # font size include: ‘xx-small’,x-small’,'small’,'medium’,‘large’,‘x-large’,‘xx-large’ or number, e.g. '12'
    plt.setp(autotexts, fontproperties=proptease)
    plt.setp(texts, fontproperties=proptease)
    ax1.set_title('Class Distribution ', loc='center')
    # ax2 只显示图例（legend）
    ax2.axis('off')
    ax2.legend(patches, labels, loc='center left', prop={'size': 10})
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)


def draw_hist(data, bins=10,
              x_label="区间", y_label="频数/频率", title="频数/频率分布直方图",
              save_name='hist.png', density=True):
    """
    绘制直方图
    data: 必选参数，绘图数据
    bins: 直方图的长条形数目，可选项，默认为10
    """
    # 设置matplotlib正常显示中文和负号
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
    # facecolor:长条形的颜色
    # edgecolor:长条形边框的颜色
    # alpha:透明度
    n, bins, patches = plt.hist(data, bins=bins, density=density, facecolor="blue", edgecolor='None')
    # 显示横轴标签
    plt.xlabel(x_label)
    # 显示纵轴标签
    plt.ylabel(y_label)
    # 显示图标题
    plt.title(title)
    plt.savefig(save_name, dpi=300)

