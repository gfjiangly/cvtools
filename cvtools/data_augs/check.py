# -*- coding:utf-8 -*-
# author   : gfjiangly
# time     : 2019/5/10 14:10
# e-mail   : jgf0719@foxmail.com
# software : PyCharm

import os
from tqdm import tqdm
from cvtools.file_io.read import read_files_to_list, read_file_to_list


def check_annots(file, save):
    if isinstance(file, list):
        annots = read_files_to_list('', file)
    else:
        annots = read_file_to_list(file)    # 去掉了首尾换行
    filte_images = set()
    for line in tqdm(annots):
        image = line.split()[0]
        if not os.path.exists(image):   # 目前仅对标签check图片是否存在
            filte_images.add(line)
    annots = list(filter(lambda x: x not in filte_images, annots))
    with open(save, 'w') as f:
        for line in annots:
            f.write(line+'\n')


if __name__ == '__main__':
    data_list = [
        'labels/gen/Thread-0_gen_annots.txt',
        'labels/gen/Thread-1_gen_annots.txt',
        'labels/gen/Thread-2_gen_annots.txt',
        'labels/train/elevator_20181230_convert_train.txt',
        'labels/train/elevator_20181231_convert_train.txt',
        'labels/train/elevator_20190106_convert_train.txt',
        'labels/train/person_7421_train.txt'
    ]
    check_annots(data_list, 'check_51k.txt')
