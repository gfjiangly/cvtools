# -*- encoding:utf-8 -*-
# @Time    : 2019/3/1 22:10
# @Author  : gfjiang
# @Site    : 
# @File    : utils.py
# @Software: PyCharm
import os
import os.path as osp
import shutil
from tqdm import tqdm

import cvtools


def find_in_path(name, path):
    """Find a file in a search path"""
    # Adapted fom
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = osp.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


# 递归文件夹下所有文件夹，得到文件列表(含路径)
def _get_files_list(root_dir):
    """get all files under the given path.

    Args:
        root_dir(str): must use absolute path to get files.

    Returns:
        list: all files under the given path.
    """
    # for Linux, isdir cannot recognize ~ home path
    root_dir = osp.expanduser(root_dir)
    if not osp.isdir(root_dir):
        return [root_dir]
    files_list = []
    for lists in os.listdir(root_dir):  # recursive
        files_list += _get_files_list(osp.join(root_dir, lists))
    return files_list


# 递归路径输出特定类型文件列表
def get_files_list(root, file_type=None, basename=False):
    """file_type is a str or list."""
    root = osp.abspath(root)
    files_list = _get_files_list(root)
    if file_type is not None:
        if isinstance(file_type, str):
            file_type = [file_type]
        files_list = [file for type in file_type
                      for file in files_list
                      if type == osp.splitext(file)[1]]
    if basename:
        # 似乎不太符合最小惊讶原则
        files_list = [file.replace(root+os.sep, '')
                      for file in files_list]
        # files_list = [osp.basename(file) for file in files_list]
    return files_list


# 递归路径输出图片列表
def get_images_list(root_dir):
    return get_files_list(root_dir, file_type=['.jpg', '.jpeg', '.png'])


# 将list随机按比例分成两部分
def split_list(data_list, test_size=0.1):
    import random
    random.shuffle(data_list)
    train_list = data_list[int(len(data_list)*test_size):]
    test_list = data_list[0:int(len(data_list)*test_size)]
    return train_list, test_list


# 将多个txt数据随机按比例分成两部分, dst无须后缀
def split_data(root, files, dst, test_size=0.1):
    data_list = cvtools.read_files_to_list(root, files)
    train_list, test_list = split_list(data_list, test_size)
    # import time
    # now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    cvtools.write_list_to_file(train_list, dst+'_train.txt')
    cvtools.write_list_to_file(test_list, dst+'_test.txt')
    return train_list, test_list


# 将dict随机按比例分成两部分
def split_dict(data_dict, test_size=0.1):
    import random
    dict_key = list(data_dict.keys())
    random.shuffle(dict_key)
    train_list = dict_key[int(len(dict_key)*test_size):]
    test_list = dict_key[0:int(len(dict_key)*test_size)]
    train_dict = {}
    for key in train_list:
        train_dict[key] = data_dict[key]
    test_dict = {}
    for key in test_list:
        test_dict[key] = data_dict[key]
    return train_dict, test_dict


# 批量将文件名中空格替换为下划线
def replace_filename_space(src_root, dst_root):
    files = get_files_list(src_root)
    if not osp.exists(dst_root):
        os.mkdir(dst_root)
    for file in files:
        temp = file.split('/')[-1].replace(' ', '_')
        os.rename(file, dst_root+temp)


# 检测文件数据是否有重复行，空行排除
def check_rept(file):
    with open(file, 'r') as f:
        str_list = f.readlines()
    count_dict = {}
    blank_line = 0
    # 如果字典里有该单词则加1，否则添加入字典
    for str in str_list:
        if str == '\n' or str == '':    # 白名单
            blank_line += 1
            continue
        if str in count_dict:
            count_dict[str] += 1
        else:
            count_dict[str] = 1
    return len(count_dict) != (len(str_list)-blank_line)


def makedirs(path):
    """对os.makedirs进行扩展

    从路径中创建文件夹，可创建多层。如果仅是文件名，则无须创建，返回False；
    如果是已存在文件或路径，则无须创建，返回False

    Args:
        path: 路径，可包含文件名。纯路径最后一个字符需要是os.sep
    """
    if path is None or path == '':  # 空
        return False
    if osp.isfile(path):    # 是文件并且已存在
        return False
    # 不能使用os.sep，因为有时在windows平台下用户也会传入使用'/'分割的路径
    if '/' not in path and '\\' not in path:  # 不含路径
        return False
    path = osp.dirname(path)
    if osp.exists(path):
        return False
    try:
        os.makedirs(path)
    except Exception as e:
        print(e, 'make dirs failed!')
        return False
    return True


def sample_label_from_images(images_src, labels_src, dst):
    assert osp.exists(images_src)
    assert osp.exists(labels_src)
    images = _get_files_list(images_src)
    if not osp.exists(dst):
        os.makedirs(dst)
    for image in tqdm(images):
        image = osp.basename(image)
        filename, extension = osp.splitext(image)
        if extension == '.jpg':
            filename = osp.join(labels_src, filename + '.json')
            if osp.exists(filename):
                shutil.copy(filename, dst)
            else:
                print('!!!Warning: %s not exists' % filename)


# 文件夹名批量替换子串
def folder_name_replace(path, list_replace):
    if list_replace is None:
        return
    # 三重循环可能效率较低
    for root, dirs, _ in os.walk(path, topdown=True):
        for key, value in list_replace.items():
            for dir in dirs:
                if key not in dir:
                    continue
                try:
                    fold = osp.join(root, dir)
                    new_fold = osp.join(root, dir.replace(key, value))
                    os.rename(fold, new_fold)  # change inplace
                except Exception as e:
                    print(e)


def files_name_replace(path, file_type=None, folder=False, list_replace=None):
    file_list = get_files_list(path, file_type)
    for file in file_list:
        if list_replace is not None:
            for key, value in list_replace.items():
                if key in file:
                    new_file = file.replace(key, value)
                    try:
                        os.rename(file, new_file)   # change inplace
                    except Exception as e:
                        print(e)
    if folder:
        folder_name_replace(path, list_replace)


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def isfile_casesensitive(path):
    if not os.path.isfile(path):
        return False   # exit early
    directory, filename = os.path.split(path)
    return filename in os.listdir(directory)


def is_image_file(filename):
    extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    return any(filename.endswith(extension) for extension in extensions)


if __name__ == "__main__":
    # # 测试通过，2019.6.28
    # txt_laber_list = get_files_list('F:/data/detection', file_type=('.txt', '.jpeg'))

    # # 测试通过，2019.3.7
    # root = osp.abspath('..')+'/datasets/'
    # files = ['elevator_20180106.txt', 'elevator_20180115.txt', 'elevator_20181230.txt', 'elevator_20181231.txt']
    # data = read_files_to_list(root, files)

    # # 测试通过，2019.3.7
    # root = osp.abspath('..')+'/datasets/'
    # files = ['elevator_20180106.txt', 'elevator_20181230.txt', 'elevator_20181231.txt']
    # files = ['elevator_20180601_convert.txt']
    # split_data(root, files)

    # # 测试通过，2019.3.7
    # src_root = '/home/arc-fsy8515/data/elevator/20190106/'
    # dst_root = '/home/arc-fsy8515/data/elevator/20190106/'
    # replace_filename_space(src_root, dst_root)

    # # 测试通过，2019.3.11
    # file = '../datasets/train/elevator_train.txt'
    # temp = check_rept(file)
    # print(file, temp)

    # root = '../datasets/'
    # files = ['elevator_20180601.txt', 'elevator_20181230.txt', 'elevator_20181231.txt', 'elevator_20190106.txt']
    # files = ['elevator_20190106_convert.txt']
    # split_data(root, files, '../datasets/elevator_20190106_convert', test_size=0.1)

    # w = 500
    # h = 400
    # labels = np.array([[580, -2, 600, 360], [-11, 565, 144, 1000]])
    # temp = [labels[:, 0:2] < 0]
    # labels[:, 0:2][labels[:, 0:2] < 0] = 0  # 左上角坐标限幅
    # labels[:, 2][labels[:, 2] > w] = w  # 右下角坐标限幅
    # labels[:, 3][labels[:, 3] > h] = h

    # images_src = 'F:/bdd/bdd100k/images/10k/val'
    # labels_src = 'F:/bdd/bdd100k/labels/100k/val'
    # dst = 'F:/bdd/bdd100k/labels/10k/val'
    # sample_label_from_images(images_src, labels_src, dst)

    # # 测试通过，2019.6.28
    # replace = {
    #     ' ': '_',
    #     '递交数据': '_submitted',
    #     '人头标注': '_head_labeling'
    # }
    # folder_name_replace('F:/data/detection', replace)

    replace = {
        ' ': '_',
        ',': '_',
        '月': '_mouth_',
        '日': '_day_',
        '递交数据': '_submitted',
        '提交数据': '_submitted',
        '人头标注': '_head_labeling',
        '视频': 'video',
        '质检完成': 'quality_inspection'
    }
    files_name_replace('/media/data/detection', folder=True, list_replace=replace)

    pass
