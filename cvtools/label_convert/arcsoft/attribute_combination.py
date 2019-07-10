# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/6/28 16:00
# e-mail   : jgf0719@foxmail.com
# software : PyCharm

from utils.boxes import x1y1x2y2_to_x1y1wh


def head_reserved(label):
    if not isinstance(label, dict):
        raise RuntimeError('only support dict type!')
    head = {}
    if 'Face_Type' in label.keys() and label['Face_Type'] == 'Head':
        head['bbox'] = x1y1x2y2_to_x1y1wh(list(map(int, label['rect'].split(','))))
        head['area'] = head['bbox'][3] * head['bbox'][2]
        # head['area'] = (head['bbox'][3] - head['bbox'][1]) * (head['bbox'][2] - head['bbox'][0])
    return head


def face_reserved(label):
    if not isinstance(label, dict):
        raise RuntimeError('only support dict type!')
    face = {}
    if 'Face_Type' in label.keys() and label['Face_Type'] == 'Alive':
        face['bbox'] = x1y1x2y2_to_x1y1wh(list(map(int, label['rect'].split(','))))
        face['area'] = face['bbox'][3] * face['bbox'][2]
        # face['area'] = (face['bbox'][3] - face['bbox'][1]) * (face['bbox'][2] - face['bbox'][0])
    return face


def rect_reserved(label):
    if not isinstance(label, dict):
        raise RuntimeError('only support dict type!')
    rect = {}
    if 'rect' in label.keys():
        bbox = x1y1x2y2_to_x1y1wh(list(map(int, label['rect'].split(','))))
        area = bbox[3] * bbox[2]
    elif 'faceRect' in label.keys():
        bbox = x1y1x2y2_to_x1y1wh(list(map(int, label['faceRect'].split(','))))
        area = bbox[3] * bbox[2]
    else:
        bbox = []
        area = 0
    rect['bbox'] = bbox
    rect['area'] = area

    if 'Action' in label.keys():
        rect['category'] = label['Action']
    elif 'nature' in label.keys():
        rect['category'] = label['nature']
    return rect


def gender_reserved(label):
    if not isinstance(label, dict):
        raise RuntimeError('only support dict type!')
    rect = {}
    if 'rect' in label.keys():
        bbox = x1y1x2y2_to_x1y1wh(list(map(int, label['rect'].split(','))))
        area = bbox[3] * bbox[2]
    elif 'faceRect' in label.keys():
        bbox = x1y1x2y2_to_x1y1wh(list(map(int, label['faceRect'].split(','))))
        area = bbox[3] * bbox[2]
    else:
        bbox = []
        area = 0
    rect['bbox'] = bbox
    rect['area'] = area

    if 'gender' in label.keys():
        rect['category'] = label['gender']
    elif '上半身衣着性别' in label.keys():
        rect['category'] = label['上半身衣着性别']
    elif '下身衣着性别' in label.keys():
        rect['category'] = label['下身衣着性别']
    # elif 'nature' in label.keys():
    #     rect['category'] = label['nature']
    return rect
