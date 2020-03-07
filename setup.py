# -*- encoding:utf-8 -*-
# @Time    : 2019/7/6 18:31
# @Author  : gfjiang
# @Site    : 
# @File    : setup.py
# @Software: PyCharm
from setuptools import Extension, dist, find_packages, setup
dist.Distribution().fetch_build_eggs(['Cython', 'numpy>=1.11.1'])

import os
import platform
import numpy  # noqa: E402
from Cython.Build import cythonize  # noqa: E402

from cpp_extension import BuildExtension, CUDAExtension, CUDA


install_requires = [
    'numpy>=1.11.1',
    'opencv-python',
    'pillow',
    'matplotlib',
    'tqdm',
    'pyyaml',
    'terminaltables',
    'mmcv>=0.2.10',
    # 'scikit-learn>=0.21.2',
    'shapely>=1.6.4',
    'terminaltables',
    'flask',
    'requests'
]


def readme():
    # PyPi默认支持的是rst格式描述，需添加type指定md格式
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_version():
    version_file = 'cvtools/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()  # 1.6.0中被删除


def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != 'Windows':
        extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings']
        }

    extension = Extension(
        '{}.{}'.format(module, name),
        [os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=[numpy_include],
        extra_compile_args=extra_compile_args)
    extension, = cythonize(extension)
    return extension


def make_cuda_ext(name, module, sources):
    extension = CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })
    extension, = cythonize(extension)
    return extension


ext_modules = [
    make_cython_ext(
        name='_mask',
        module='cvtools.cocotools',
        sources=['maskApi.c', '_mask.pyx'],
    ),
    make_cython_ext(
        name='soft_nms_cpu',
        module='cvtools.ops.nms',
        sources=['src/soft_nms_cpu.pyx']
    ),
    make_cython_ext(
        name='_polyiou',
        module='cvtools.ops.polyiou',
        sources=['src/polyiou_wrap.cxx', 'src/polyiou.cpp']
    ),
]
if CUDA is not None:
    cuda_ext_module = [
        make_cuda_ext(
            'poly_overlaps',
            'cvtools.ops.polyiou',
            sources=['src/poly_overlaps_kernel.cu', 'src/poly_overlaps.pyx']
        ),
        make_cuda_ext(
            'poly_nms',
            'cvtools.ops.polynms',
            sources=['src/poly_nms_kernel.cu', 'src/poly_nms.pyx']
        ),
    ]
    ext_modules += cuda_ext_module

setup(
    name='cvtoolss',
    version=get_version(),
    description='Computer Vision Foundation Utilities',
    long_description=readme(),
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
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
    entry_points={'console_scripts': ['cvtools=cvtools:main']},
)

"""如果应用在开发过程中会频繁变更，每次安装还需要先将原来的版本卸掉，很麻烦。
使用”develop”开发方式安装的话，应用代码不会真的被拷贝到本地Python环境的”site-packages”目录下，
而是在”site-packages”目录里创建一个指向当前应用位置的链接。
这样如果当前位置的源码被改动，就会马上反映到”site-packages”里。"""
# python setup.py develop
# or
"""该命令会将当前的Python应用安装到当前Python环境的”site-packages”目录下，
这样其他程序就可以像导入标准库一样导入该应用的代码了。"""
# python setup.py install
