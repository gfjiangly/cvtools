CVTools
=======

介绍
====
cvtools是主要用于计算机视觉领域的Python工具包。在实现和训练CV模型过程，一些与核心无关的常用代码被剥离出，形成此库。

它提供以下功能：

- 数据集格式转换（voc->coco，dota->coco等）
- 数据增强（如旋转、随机裁剪、颜色变换等）
- 数据标签分析（如统计类别实例数、占比、分布等）
- 模型输出结果评估
- 通用的输入输出APIs
- 一些实用函数（如可视化模型输出，计算IoU等）

安装
====

.. code::

    pip install cvtoolss


.. note::
    这里多一个s，cvtools这个名字在PyPi中已被占用。PyPi上的包可能不是最新的，建议从源码安装。

从源码安装

.. code::

    git clone https://github.com/gfjiangly/cvtools.git
    cd cvtools
    pip install -e .
