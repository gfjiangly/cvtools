# -*- encoding:utf-8 -*-
# @Time    : 2019/7/6 18:31
# @Author  : gfjiang
# @Site    : 
# @File    : setup.py
# @Software: PyCharm

from setuptools import find_packages, setup, Extension
import numpy as np


install_requires = [
    'numpy>=1.11.1', 'opencv-python', 'pillow', 'matplotlib', 'tqdm',
    'pyyaml', 'terminaltables',
    'scikit-learn>=0.21.2'
]


def get_version():
    version_file = 'cvtools/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


ext_modules = [
    Extension(
        'cvtools/pycocotools._mask',
        sources=['cvtools/pycocotools/maskApi.c', 'cvtools/pycocotools/_mask.pyx'],
        include_dirs=[np.get_include(), './'],
        extra_compile_args=[]  # originally was ['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]

setup(
    name='cvtoolss',
    version=get_version(),
    description='Computer Vision Foundation Utilities',
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
    ext_modules= ext_modules,
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
