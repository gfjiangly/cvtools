# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/6/28 16:00
# e-mail   : jgf0719@foxmail.com
# software : PyCharm

import cvtools


def rect_reserved(label):
    if not isinstance(label, dict):
        raise RuntimeError('only support dict type!')

    ann = {}
    if 'rect' in label.keys():
        bbox = cvtools.x1y1x2y2_to_x1y1wh(list(map(int, label['rect'].split(','))))
    elif 'faceRect' in label.keys():
        bbox = cvtools.x1y1x2y2_to_x1y1wh(list(map(int, label['faceRect'].split(','))))
    else:
        bbox = []

    if len(bbox) > 0:
        area = bbox[3] * bbox[2]
    else:
        area = 0

    ann['bbox'] = bbox
    ann['area'] = area

    return ann


def face_reserved(label, have_assert=False):
    if have_assert:
        assert ('Face_Type' in label.keys() and label['Face_Type'] == 'Alive')
    ann = rect_reserved(label)
    ann['category'] = 'face'
    return ann


def head_reserved(label, have_assert=False):
    if have_assert:
        assert ('Face_Type' in label.keys() and label['Face_Type'] == 'Head') or \
               ('Action' in label.keys() and label['Action'] == 'Head')
    ann = rect_reserved(label)
    ann['category'] = 'head'
    return ann


def gender_reserved(label):
    ann = rect_reserved(label)

    if 'gender' in label.keys():
        ann['category'] = label['gender']
    elif '上半身衣着性别' in label.keys():
        ann['category'] = label['上半身衣着性别']
    elif '下身衣着性别' in label.keys():
        ann['category'] = label['下身衣着性别']
    # elif 'nature' in label.keys():
    #     rect['category'] = label['nature']
    return ann
