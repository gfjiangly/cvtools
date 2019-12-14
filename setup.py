# -*- encoding:utf-8 -*-
# @Time    : 2019/7/6 18:31
# @Author  : gfjiang
# @Site    : 
# @File    : setup.py
# @Software: PyCharm
from setuptools import find_packages, setup, Extension, dist

dist.Distribution().fetch_build_eggs(['Cython', 'numpy>=1.11.1'])

import numpy  # noqa: E402
from Cython.Distutils import build_ext  # noqa: E402


install_requires = [
    'numpy>=1.11.1',
    'opencv-python',
    'pillow',
    'matplotlib',
    'tqdm',
    'pyyaml',
    'terminaltables',
    'mmcv>=0.2.13',
    # 'scikit-learn>=0.21.2',
    'shapely>=1.6.4',
    'terminaltables',
]


def get_version():
    version_file = 'cvtools/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


ext_modules = [
    Extension(
        'cvtools.cocotools._mask',
        sources=['cvtools/cocotools/maskApi.c', 'cvtools/cocotools/_mask.pyx'],
        include_dirs=[numpy.get_include(), './'],
        # originally was ['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
        extra_compile_args=[]
    )
]

setup(
    name='cvtoolss',
    version=get_version(),
    description='Computer Vision Foundation Utilities',
    long_description="""cvtools is a Python toolkit mainly used in the field of 
    computer vision. In the process of implementing and training the CV model, 
    some common code unrelated to the core was stripped out to form this library.
    
    It provides the following functions:
       
       - Data set format conversion (voc-> coco, dota-> coco, etc.)
       - Data augmentation (such as rotation, random cropping, 
            color transformation, etc.)
       - Data label analysis (such as statistics, number of instances, 
            proportion, distribution, etc.)
       - Evaluation of model output results
       - Common input and output APIs
       - Some useful functions (such as visualizing model output, 
            calculating IoU, etc.)
    """,
    keywords='computer vision',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Utilities',
    ],
    # metadata for upload to PyPI
    author='Guangfeng Jiang',
    author_email='gfjiang_xxjl@163.com',
    url='https://github.com/gfjiangly/cvtools',

    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=install_requires,
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False)

"""如果应用在开发过程中会频繁变更，每次安装还需要先将原来的版本卸掉，很麻烦。
使用”develop”开发方式安装的话，应用代码不会真的被拷贝到本地Python环境的”site-packages”目录下，
而是在”site-packages”目录里创建一个指向当前应用位置的链接。
这样如果当前位置的源码被改动，就会马上反映到”site-packages”里。"""
# python setup.py develop
# or
"""该命令会将当前的Python应用安装到当前Python环境的”site-packages”目录下，
这样其他程序就可以像导入标准库一样导入该应用的代码了。"""
# python setup.py install
