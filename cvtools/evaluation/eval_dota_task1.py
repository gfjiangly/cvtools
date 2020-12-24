# -*- encoding:utf-8 -*-
# @Time    : 2019/10/30 16:56
# @Author  : gfjiang
# @Site    : 
# @File    : dota_evaluation_task1.py
# @Software: PyCharm
from argparse import ArgumentParser
import numpy as np
from terminaltables import AsciiTable
import matplotlib
import matplotlib.pyplot as plt

from cvtools.ops.polyiou import polyiou


def parse_gt(filename):
    """

    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if len(splitlines) < 9:
                    continue
                object_struct['name'] = splitlines[8]

                if len(splitlines) == 9:
                    object_struct['difficult'] = 0
                elif len(splitlines) == 10:
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects


def draw_precision_recall_curve(precisions, recalls, classes, idxs, legend_out=True, to_file=None):

    fig, ax = plt.subplots()
    # plt.rcParams['figure.figsize'] = (6.0, 3.5) # 设置figure_size尺寸
    # plt.rcParams['savefig.dpi'] = 300 #图片像素
    # plt.rcParams['figure.dpi'] = 300 #分辨率

    classes_sort = []
    for i in idxs:
        classes_sort.append(classes[i])
        mrec = np.concatenate(([0.], recalls[i], [1.]))
        mpre = np.concatenate(([0.], precisions[i], [0.]))

        # compute the precision envelope
        for k in range(mpre.size - 1, 0, -1):
            mpre[k - 1] = np.maximum(mpre[k - 1], mpre[k])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        idxs = np.where(mrec[1:] != mrec[:-1])[0]

        plt.plot(mrec[idxs], mpre[idxs])

    # plt.plot(recalls, precisions)
    # 显示图标题
    plt.title('Precision x Recall curve')
    # 显示横轴标签
    font = {
        'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 12,
    }
    plt.xlabel('recall', font)
    # 显示纵轴标签
    plt.ylabel('precision', font)
    # plt.xlim(0., 1.)
    if legend_out:
        plt.legend(labels=classes_sort, bbox_to_anchor=(1.05, 0.), loc=3, borderaxespad=0,
                prop={'size': 7})
        fig.subplots_adjust(right=0.7)
    else:
        plt.legend(labels=classes_sort, loc='best')
    
    # plt.show()
    if to_file:
        plt.savefig(to_file, dpi=300, bbox_inches='tight')


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # draw_precision_recall_curve(mpre[i], mrec[i], 'pr.jpg')

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             # cachedir,
             ovthresh=0.5,
             use_07_metric=False,
             det_thresh=0.3):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    # if not os.path.isdir(cachedir):
    #   os.mkdir(cachedir)
    # cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # print('imagenames: ', imagenames)
    # if not os.path.isfile(cachefile):
    # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        # print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))
        # if i % 100 == 0:
        #   print ('Reading annotation for {:d}/{:d}'.format(
        #      i + 1, len(imagenames)) )
        # save
        # print ('Saving cached annotations to {:s}'.format(cachefile))
        # with open(cachefile, 'w') as f:
        #   cPickle.dump(recs, f)
    # else:
    # load
    # with open(cachefile, 'r') as f:
    #   recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    splitlines = [x for x in splitlines if float(x[1]) > det_thresh]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    # conf_inds = np.where(confidence > det_thresh)[0]

    # print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    # print('check sorted_scores: ', sorted_scores)
    # print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    # print('check imge_ids: ', image_ids)
    # print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):
                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps

            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    # print('check fp:', int(np.sum(fp)))
    # print('check tp', int(np.sum(tp)))

    # print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def eval_dota_task1(det_path, ann_path, imageset_file, det_thresh=0.3):
    det_path += '/{:s}.txt'
    ann_path += '/{:s}.txt'

    # # For DOTA-v1.5
    # classnames = [
    #     'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
    #     'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
    #     'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter',
    #     'container-crane'
    # ]
    # For DOTA-v1.0
    classnames = [
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
        'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
        'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    ]
    cls_det_thres = {'plane': 0.01}
    classaps = []
    map = 0
    recs = []
    precs = []
    for classname in classnames:
        # print('classname:', classname)
        if classname in cls_det_thres:
            score_thres = cls_det_thres[classname]
        else:
            score_thres = det_thresh
        rec, prec, ap = voc_eval(det_path,
                                 ann_path,
                                 imageset_file,
                                 classname,
                                 ovthresh=0.5,
                                 use_07_metric=True,
                                 det_thresh=score_thres)
        map = map + ap
        # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        # print('ap: ', ap)
        classaps.append(ap)
        recs.append(rec)
        precs.append(prec)

        # umcomment to show p-r curve of each category
        # plt.figure(figsize=(8,4))
        # plt.xlabel('recall')
        # plt.ylabel('precision')
        # plt.plot(rec, prec)
        # plt.show()

    ap_idxs = np.array(classaps).argsort()[::-1]
    first3_last3 = np.concatenate([ap_idxs[:3], ap_idxs[-3:]])
    draw_precision_recall_curve(precs, recs, classnames, first3_last3, legend_out=False, to_file='pr_first3_last3_in.jpg')
    # draw_precision_recall_curve(precs, recs, classnames, ap_idxs[:3], legend_out=False, to_file='pr_first3.jpg')
    # draw_precision_recall_curve(precs, recs, classnames, ap_idxs[::-1][:3],  legend_out=False, to_file='pr_last3.jpg')

    map = map / len(classnames)
    # print('map:', map)
    classaps = 100 * np.array(classaps)
    # np.set_printoptions(precision=2)

    # print ap
    # header = ['mAP'] + classnames
    # body = [map*100] + classaps.tolist()
    # body = ['{:.2f}'.format(x) for x in body]
    # table_data = [header, body]
    # table = AsciiTable(table_data)
    # table.inner_footing_row_border = True
    # print(table.table)

    header = ['class', 'ap']
    table_data = [header]

    # sort_ids = np.argsort(classaps)
    sort_ids = range(len(classaps))
    import os.path as osp
    with open(osp.join(osp.dirname(det_path), 'results.txt'), 'w') as f:
        f.write('map: {}\n'.format(map))
        for id in sort_ids:
            line = '{}: {:.2f}'.format(classnames[id], classaps[id])
            # print(line, end=' ')
            f.write(line+'\n')
            table_data.append([classnames[id], '{:.2f}'.format(classaps[id])])
    table_data.append(['mAP', '{:.2f}'.format(map*100)])
    table = AsciiTable(table_data)
    table.inner_footing_row_border = True
    print(table.table)


def main():
    parser = ArgumentParser(description='DOTA Evaluation(PASCAL FORMAT)')
    parser.add_argument('det', help='result file path')
    parser.add_argument('ann', help='annotation file path')
    parser.add_argument('img_list', help='imageset file path')
    parser.add_argument('--det_thresh', type=float, default=0.05)
    args = parser.parse_args()
    eval_dota_task1(args.det, args.ann, args.img_list, args.det_thresh)


if __name__ == '__main__':
    # # det_path = r'../../tests/data/DOTA/eval/Task1_results_nms'
    # det_path = r'/code/AerialDetection/work_dirs/retinanet_obb_r50_fpn_1x_dota_1gpus_adapt/Task1_results_nms'
    # ann_path = r'/media/data/DOTA/val/labelTxt-v1.0/labelTxt'
    # imageset_file = r'/media/data/DOTA/val/val.txt'
    # eval_dota_task1(det_path, ann_path, imageset_file, det_thresh=0.05)
    main()
