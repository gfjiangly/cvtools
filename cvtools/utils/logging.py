# -*- encoding:utf-8 -*-
# @Time    : 2019/9/20 22:09
# @Author  : gfjiang
# @Site    : 
# @File    : logging.py
# @Software: PyCharm
import logging


def get_logger(level=logging.INFO, name=None):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=level)
    logger = logging.getLogger(name)
    return logger


def logger_file_handler(logger,
                        filename=None,
                        mode='w',
                        level=logging.INFO):
    file_handler = logging.FileHandler(filename, mode)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    return logger
