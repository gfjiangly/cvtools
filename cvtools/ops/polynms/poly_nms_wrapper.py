import numpy as np

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


def poly_nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if poly_gpu_nms is not None and not force_cpu:
        if dets.shape[0] == 0:
            return []
        return poly_gpu_nms(dets, thresh, device_id=0)
    else:
        # raise NotImplemented("poly_nms的cpu版本暂未实现！")
        return poly_cpu_nms(dets, thresh)
