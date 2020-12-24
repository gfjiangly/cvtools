import numpy as np

from cvtools.ops.polyiou import VectorDouble, iou_poly
from cvtools.ops.polyiou.polyiou_wrap import poly_overlaps
try:
    from .poly_nms import poly_gpu_nms
except ImportError as e:
    poly_gpu_nms = None
    print(e)


def poly_cpu_nms(dets, thresh):
    scores = dets[:, -1]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # calu iou
        ovr = poly_overlaps(np.array([dets[i]]), dets[order[1:]])
        ovr = np.squeeze(ovr)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def py_cpu_nms_poly_fast(dets, thresh):

    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = VectorDouble(
            [dets[i][0], dets[i][1], dets[i][2], dets[i][3],
             dets[i][4], dets[i][5], dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = dets[:, -1].argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
        inds = np.where(hbb_ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def poly_nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    # dets = dets.astype(np.float32)
    # if poly_gpu_nms is not None and not force_cpu:
    #     if dets.shape[0] == 0:
    #         return []
    #     return poly_gpu_nms(dets, thresh, device_id=0)
    # else:
        # raise NotImplemented("poly_nms的cpu版本暂未实现！")
    return py_cpu_nms_poly_fast(dets, thresh)
