# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/6/10 11:13
# e-mail   : jgf0719@foxmail.com
# software : PyCharm
import numpy as np
import cv2.cv2 as cv


def x1y1wh_to_x1y1x2y2(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format.
    supported type: list, type and np.ndarray
    """
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2] - 1.)
        y2 = y1 + np.maximum(0., xywh[3] - 1.)
        return [x1, y1, x2, y2]
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1))
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')


def x1y1x2y2_to_x1y1wh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return [x1, y1, w, h]
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def x1y1wh_to_x1y1x2y2x3y3x4y4(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2 x3 y3 x4 y4] format.
    supported type: list, type and np.ndarray
    """
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x4 = x1 + np.maximum(0., xywh[2] - 1.)
        y4 = y1 + np.maximum(0., xywh[3] - 1.)
        return [x1, y1, x4, y1, x4, y4, x1, y4]
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        x1y1 = xywh[:, 0:2]
        x4y4 = xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1)
        return np.hstack(
            (x1y1, np.hstack((x4y4[..., 0], x1y1[..., 1])),
             x4y4, np.hstack((x1y1[..., 0], x4y4[..., 1])))
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')


def xywh_to_x1y1x2y2(xywh):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2
        x2, y2 = xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2
        return x1, y1, x2, y2
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2] - xywh[:, 2:4] / 2, xywh[:, 0:2] + xywh[:, 2:4] / 2)
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')


def x1y1x2y2_to_xywh(x1y1x2y2):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    if isinstance(x1y1x2y2, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(x1y1x2y2) == 4
        ct_x, ct_y = (x1y1x2y2[3] + x1y1x2y2[1]) / 2, (x1y1x2y2[2] + x1y1x2y2[0]) / 2
        w, h = x1y1x2y2[3] - x1y1x2y2[1] + 1, x1y1x2y2[2] - x1y1x2y2[0] + 1
        return ct_x, ct_y, w, h
    elif isinstance(x1y1x2y2, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            ((x1y1x2y2[:, 0:2] + x1y1x2y2[:, 2:4]) / 2, x1y1x2y2[:, 2:4] - x1y1x2y2[:, 0:2] + 1)
        )
    else:
        raise TypeError('Argument x1y1x2y2 must be a list, tuple, or numpy array.')


def x1y1wh_to_xywh(x1y1wh):
    """Convert [x1 y1 w h] box format to [x y w h] format.
    supported type: list, type and np.ndarray
    """
    if isinstance(x1y1wh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(x1y1wh) == 4
        w = x1y1wh[2]
        h = x1y1wh[3]
        x, y = x1y1wh[0] + w/2, x1y1wh[1] + h/2
        return x, y, w, h
    elif isinstance(x1y1wh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (x1y1wh[:, 0:2]+x1y1wh[:, 2:4]/2, x1y1wh[:, 2:4])
        )
    else:
        raise TypeError('Argument x1y1wh must be a list, tuple, or numpy array.')


# 已测试
def rotate_rect(rect, center, angle):
    """一个数学问题：2x2矩阵（坐标）与旋转矩阵相乘.
    在笛卡尔坐标系中，angle>0, 逆时针旋转; angle<0, 顺时针旋转
    return: x1y1x2y2x3y3x4y4 format box
    """
    assert 180 > angle > -180
    # 顺时针排列坐标
    x1, y1, x3, y3 = rect
    x2, y2, x4, y4 = x3, y1, x1, y3
    x, y = center
    # anti-clockwise to clockwise arc
    cosA = np.cos(np.pi / 180. * angle)
    sinA = np.sin(np.pi / 180. * angle)

    x1n = (x1 - x) * cosA - (y1 - y) * sinA + x
    y1n = (x1 - x) * sinA + (y1 - y) * cosA + y

    x2n = (x2 - x) * cosA - (y2 - y) * sinA + x
    y2n = (x2 - x) * sinA + (y2 - y) * cosA + y

    x3n = (x3 - x) * cosA - (y3 - y) * sinA + x
    y3n = (x3 - x) * sinA + (y3 - y) * cosA + y

    x4n = (x4 - x) * cosA - (y4 - y) * sinA + x
    y4n = (x4 - x) * sinA + (y4 - y) * cosA + y

    return [x1n, y1n, x2n, y2n, x3n, y3n, x4n, y4n]     # 顺时针方向


def rotate_rects(rects, centers, angle):
    """一个数学问题：坐标矩阵与旋转矩阵相乘.
    在笛卡尔坐标系中，angle>0, 逆时针旋转; angle<0, 顺时针旋转
    return: x1y1x2y2x3y3x4y4 format box"""
    if not isinstance(rects, np.ndarray):
        rects = np.array([rects])
    if not isinstance(centers, np.ndarray):
        centers = np.array([centers])
    assert len(centers) == len(rects) > 0
    # assert 180 > angle > -180

    # 顺时针排列坐标
    rects = np.hstack((rects[:, :2],
                       rects[:, 2:3], rects[:, 1:2],
                       rects[:, 2:4],
                       rects[:, 0:1], rects[:, 3:4]))
    coors = rects.reshape(-1, 2)

    cosA = np.tile(np.cos(np.pi / 180. * angle), (4, 1))
    sinA = np.tile(np.sin(np.pi / 180. * angle), (4, 1))

    x, y = coors[:, 0:1], coors[:, 1:2]
    x0, y0 = np.tile(centers[:, 0:1], (4, 1)), np.tile(centers[:, 1:2], (4, 1))

    xn = (x - x0) * cosA - (y - y0) * sinA + x0
    yn = (x - x0) * sinA + (y - y0) * cosA + y0

    return np.hstack((xn, yn)).reshape(-1, 8)     # 顺时针方向


def xywha_to_x1y1x2y2x3y3x4y4(xywha):
    """用旋转的思路做变换是最通用和最简单的"""
    if isinstance(xywha, (list, tuple)):
        assert len(xywha) == 5
        return rotate_rect(xywha[:4], xywha[:2], xywha[4:5])
    elif isinstance(xywha, np.ndarray):
        return rotate_rects(xywha[:, :4], xywha[:, :2], xywha[:, 4:5])
    else:
        raise TypeError('Argument xywha must be a list, tuple, or numpy array.')


def get_minAreaRect(cnt):
    """返回：（中心(x,y), (宽,高), 旋转角度）"""
    assert isinstance(cnt, np.ndarray)
    cnt = cnt.reshape(-1, 2)
    # the clockwise output convex hull in the Cartesian coordinate system
    cnt_hull = cv.convexHull(cnt.astype(np.float32), clockwise=False)
    xywha = cv.minAreaRect(cnt_hull)
    return xywha


if __name__ == '__main__':
    pass
