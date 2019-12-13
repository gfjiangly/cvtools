# -*- encoding:utf-8 -*-
# @Time    : 2019/9/15 20:43
# @Author  : gfjiang
# @Site    : 
# @File    : f1_score.py
# @Software: PyCharm
import numpy as np
from collections import defaultdict

from .mean_ap import get_cls_results, tpfp_imagenet, tpfp_default
from cvtools.cocotools.coco import COCO
from cvtools.utils.boxes import x1y1wh_to_x1y1x2y2


def get_coco_anns(file, imgs=None):
    gt_bboxes = []
    gt_labels = []
    coco = COCO(file)
    if not imgs:
        imgs = [img_info['file_name']
                for img_info in coco.dataset['images']]
    imgToAnns = defaultdict(list)
    for ann in coco.dataset['annotations']:
        imgToAnns[coco.imgs[ann['image_id']]['file_name']].append(ann)
    anns = [imgToAnns[img] if img in imgToAnns.keys() else []
            for img in imgs]
    for ann in anns:
        bboxes = np.array([bbox['bbox'] for bbox in ann])
        if len(bboxes) > 0:
            bboxes = x1y1wh_to_x1y1x2y2(bboxes)
        labels = np.array([bbox['category_id'] for bbox in ann])
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    return gt_bboxes, gt_labels


def f1_score(det_results,
             gt_bboxes,
             gt_labels,
             gt_ignore=None,
             scale_ranges=None,
             iou_thr=0.5,
             dataset=None,
             print_summary=True):
    """Evaluate F1 score of a dataset.

    Args:
        det_results (list): a list of list, [[cls1_det, cls2_det, ...], ...]
        gt_bboxes (list): ground truth bboxes of each image, a list of K*4
            array.
        gt_labels (list): ground truth labels of each image, a list of K array
        gt_ignore (list): gt ignore indicators of each image, a list of K array
        scale_ranges (list, optional): [(min1, max1), (min2, max2), ...]
        iou_thr (float): IoU threshold
        dataset (None or str or list): dataset name or dataset classes, there
            are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc.
        print_summary (bool): whether to print the mAP summary

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(gt_bboxes) == len(gt_labels)
    if gt_ignore is not None:
        assert len(gt_ignore) == len(gt_labels)
        for i in range(len(gt_ignore)):
            assert len(gt_labels[i]) == len(gt_ignore[i])
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    eval_results = []
    cls_scores = []
    num_classes = len(det_results[0])  # positive class num
    gt_labels = [
        label if label.ndim == 1 else label[:, 0] for label in gt_labels
    ]
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gt_ignore = get_cls_results(
            det_results, gt_bboxes, gt_labels, gt_ignore, i)
        # calculate tp and fp for each image
        tpfp_func = (
            tpfp_imagenet if dataset in ['det', 'vid'] else tpfp_default)
        tpfp = [
            tpfp_func(cls_dets[j], cls_gts[j], cls_gt_ignore[j], iou_thr,
                      area_ranges) for j in range(len(cls_dets))
        ]
        tp, fp = tuple(zip(*tpfp))
        # calculate gt number of each scale, gts ignored or beyond scale
        # are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += np.sum(np.logical_not(cls_gt_ignore[j]))
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0] + 1) * (
                    bbox[:, 3] - bbox[:, 1] + 1)
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum(
                        np.logical_not(cls_gt_ignore[j]) &
                        (gt_areas >= min_area) & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        cls_scores.append(cls_dets[sort_inds][:, 4])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        f1 = 2 * precisions * recalls / (precisions + recalls)
        eval_results.append(f1)
    max_f1_ids = [f1s.argmax(axis=1)[0] for f1s in eval_results]
    best_f1 = [f1s[:, i] for i, f1s in zip(max_f1_ids, eval_results)]
    best_th = [scores[i] for i, scores in zip(max_f1_ids, cls_scores)]
    print('best f1: {}, best score threshold: {}'.format(best_f1, best_th))
    return eval_results, cls_scores


