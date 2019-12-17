=========
数据增强
=========

所有实现均在 ``cvtools.data_augs`` 子包中

大图裁剪成小图
==============
输入到网络的图像尺寸应该适中。太大了，resize之后可能导致目标过小，且细节丢失，
因此针对尺寸较大的图（>1024x1024像素），应先做裁剪，后进一步做数据增强等处理。

裁剪过程由三个类完成：

- cvtools.data_augs.crop.crop_abc.CropDataset
- cvtools.data_augs.crop.crop_abc.CropMethod
- cvtools.data_augs.crop.crop_abc.Crop

三个类均是抽象类，CropDataset类定义裁剪输入格式，所有用于裁剪的数据集均需继承此类，实现抽象方法。
cvtools目前提供：
- CocoDatasetForCrop: COCO数据集格式

CropMethod类定义裁剪方法，所有自定义裁剪方法均需继承此类，cvtools目前提供：
- CropImageInOrder: 滑动窗口裁剪
- CropImageAdaptive: 自适应裁剪
- CropImageProtected: 保护裁剪

Crop类规定裁剪接口，CropLargeImages类实现了全部接口：
- crop_for_train
- crop_for_test
- save

save功能实际由CropDataset提供，假设输入数据集清楚输出数据集的格式。

目前提供的裁剪方法有：

1 滑动窗口裁剪
---------------
顾名思义，俱均匀的在大图上滑动裁剪出子图，同时标签坐标也做相应转换。可以设置滑动时的重合率。

::

    x方向滑动重合的像素数 = 图像宽*重合率
    y方向滑动重合的像素数 = 图像高*重合率



滑动时的重合率参数过大，将导致目标容易，可能加剧类别实例数的不平衡；
滑动时的重合率参数过小，可能导致实例数量丢失较多。需要根据自己数据集合理设置此参数

目前用于裁剪的数据集格式仅支持coco格式，如不是coco格式需使用cvtools.label_convert中模块转换。

支持将裁剪的数据集保存成coco兼容格式，即在 ``images`` 字段的图片信息中添加 ``crop`` 字段: [x1, y1, x2, y2]，
表示裁剪框位于原图的左上角和右下角坐标。训练时读取原图，然后利用 ``crop`` 字典信息裁出子图
(注意做缓存，第一个epoch后，均直接读取缓存的子图，加快训练过程)，标签坐标无须转换，已经在生成
 ``crop`` 字段过程转换过。


用法

.. code-block:: python

    import os.path as osp
    import cvtools.data_augs as augs

    current_path = osp.dirname(__file__)

    img_prefix = current_path + '/data/DOTA/images'
    ann_file = current_path + '/data/DOTA/dota_x1y1wh_polygon.json'

    # 用于裁剪的数据集中间表示层，继承自cvtools.data_augs.crop.crop_abc.CropDataset
    dataset = augs.CocoDatasetForCrop(img_prefix, ann_file)

    # 定义滑动窗口裁剪方法
    crop_method = augs.CropImageInOrder(crop_w=1024, crop_h=1024, overlap=0.2)

    # 将数据集和裁剪方法传入通用裁剪类CropLargeImages
    crop = augs.CropLargeImages(dataset, crop_method)
    crop.crop_for_train()
    crop.save(to_file=current_path+'/out/crop/train_dota_crop1024.json')



.. note::
    如果不想将自己的数据集转换成COCO格式，需自行实现CropDataset类所有接口即可。


此外，CropLargeImages支持对特定类别实例重采样，示例：

.. code-block:: python

    # 接上代码
    # 对实例数较少的类别重采样
    crop.crop_for_train(over_samples={'roundabout': 100, })
    crop.save(to_file=current_path+'/out/crop/train_dota_crop1024+over.json')



2 自适应裁剪
------------
这里的自适应指适应裁剪窗口大小，实际上是在滑动窗口裁剪基础上，做了一些判断，修改裁剪窗口大小。

减少窗口大小情况。目的是放大密集的小目标，使小目标有很好的检测效果

- 小目标（<32x32像素）比例超过small_prop
- 目标总数超过max_objs

使用设定窗口，滑动裁剪

- 图片宽或高超过size_th

保护裁剪

- 大实例（>96x96像素）被破坏

实践中发现，保护裁剪，可能导致增加了小目标数量而加剧实例数的不平衡。

用法

.. code-block:: python

    import os.path as osp
    import cvtools.data_augs as augs

    current_path = osp.dirname(__file__)

    img_prefix = current_path + '/data/DOTA/images'
    ann_file = current_path + '/data/DOTA/dota_x1y1wh_polygon.json'
    dataset = augs.CocoDatasetForCrop(img_prefix, ann_file)

    crop_method = augs.CropImageAdaptive(
        overlap=0.1,      # 滑窗重合率
        iof_th=0.7,       # 超出裁剪范围iof阈值
        small_prop=0.5,   # 小目标比例阈值
        max_objs=100,     # 目标总数阈值
        size_th=1024,     # 滑窗最大尺寸阈值
        strict_size=True  # 是否严格遵循size_th约束
    )

    crop = augs.CropLargeImages(dataset, crop_method)
    crop.crop_for_train()
    crop.save(to_file=current_path+'/out/crop/train_dota_ada.json')



旋转和镜像
==========

对于使用水平矩形框（HBB）检测的模型，旋转任意角度可能导致GT框变大。cvtools提供角度为90/180/270
的旋转，不影响GT框的大小。

cvtools提供沿水平轴镜像和沿竖直轴镜像。

用法见测试文件：
- https://github.com/gfjiangly/cvtools/blob/dev/tests/test_mirror.py
- https://github.com/gfjiangly/cvtools/blob/dev/tests/test_rotate.py


缩放和裁剪
==========

Crop是从一张图中取一个patch，经resize后起到放大图像局部区域作用。
Expand是扩大，其行为是制作一个比原图大的画布，然后讲原图贴进去，resize后起到缩小图像作用。

Notes:
这里说的起到放大与缩小作用，均是和原图resize到特定大小做对比。

实现这两种功能的类分别是：

- cvtools.data_augs.augmentation.RandomSampleCrop
- cvtools.data_augs.augmentation.Expand


色彩变换
========

RGB空间

- 对比度变化
- 亮度Lightness变化

HSV空间

- 色相Hue变化
- 饱和度Saturation变化
- 明度Value变化

cvtools提供以下实现：

- RandomContrast
- RandomSaturation
- RandomHue
- RandomBrightness
- RandomLightingNoise
- PhotometricDistort 组合了以上所有关于颜色的变化


Resize
=======

提供了两种resize，一种是可能导致图像变形的resize，还有一种是使用填充保持图像比例的resize，分别
由以下类实现：

- Resize
- ResizeFilled


Compose组合
===========

使用Compose类可将变换组合在一起使用。

例子：

.. code-block:: python

    import cvtools.data_augs.augmentations as augs


    class SSDAugmentation(object):
        def __init__(self, size=300, mean=(104, 117, 123)):
            self.mean = mean
            self.size = size
            self.augment = augs.Compose([
                augs.ConvertFromInts(),      # int->np.float32
                augs.ToAbsoluteCoords(),     # Absolute Coords
                augs.PhotometricDistort(),   # 色彩变换
                augs.Expand(self.mean),      # 图像扩展
                augs.RandomSampleCrop(),     # 随机裁剪
                augs.RandomMirror(),         # 随机镜像
                augs.ToPercentCoords(),      # [0, 1] Relative Coords
                augs.Resize(self.size),
                augs.SubtractMeans(self.mean)
            ])

        def __call__(self, img, boxes, labels):
            return self.augment(img, boxes, labels)

