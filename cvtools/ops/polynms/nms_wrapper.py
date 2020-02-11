try:
    from .poly_nms import poly_gpu_nms
except ImportError:
    poly_gpu_nms = None


def poly_nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if poly_gpu_nms is not None and not force_cpu:
        if dets.shape[0] == 0:
            return []
        return poly_gpu_nms(dets, thresh, device_id=0)
    else:
        raise NotImplemented("poly_nms的cpu版本暂未实现！")
