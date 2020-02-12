CVTools
=======

 [![Documentation Status](https://readthedocs.org/projects/cvtools/badge/?version=latest)](https://cvtools.readthedocs.io/zh/latest/?badge=latest)
[![Travis CI Build](https://travis-ci.com/gfjiangly/cvtools.svg?branch=master)](https://travis-ci.com/gfjiangly/cvtools)
 [![codecov](https://codecov.io/gh/gfjiangly/cvtools/branch/master/graph/badge.svg)](https://codecov.io/gh/gfjiangly/cvtools)
[![PyPI - Python Version](https://img.shields.io/pypi/v/cvtoolss)](https://pypi.org/project/cvtoolss)
[![GitHub - License](https://img.shields.io/github/license/open-mmlab/mmcv.svg)](https://github.com/gfjiangly/cvtools/blob/master/LICENSE)

Computer Vision Tool Library


Introduction
------------

cvtools is a helpful python library for computer vision.

It provides the following functionalities.

- Dataset Conversion(voc to coco, bdd to coco, ...)
- Data Augmentation(random mirror, random sample crop, ...)
- Dataset Analysis(visualization, cluster analysis, ...)
- Image processing(crop, resize, ...)
- Model web deployment(command line usage)
- Useful utilities (iou, timer, ...)
- Universal IO APIs

See the [documentation](https://cvtools.readthedocs.io/zh/latest) for more features and usage.


Installation
------------
Try and start with
```bash
pip install cvtoolss
```
Note: There are two s at the end.

or install from source
```bash
git clone https://github.com/gfjiangly/cvtools.git
cd cvtools
pip install .  # (add "-e" if you want to develop or modify the codes)
```


TodoList
--------
- [x] Add web deployment models


License
-------
This project is released under the [Apache 2.0 license](LICENSE).
