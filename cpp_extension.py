# -*- encoding:utf-8 -*-
# @Time    : 2020/2/8 14:52
# @Author  : jiang.g.f
# @File    : cpp_extension.py
# @Software: PyCharm
import setuptools
from os.path import join as pjoin
import copy
import os
import re
import sys
import numpy as np  # noqa: E402

from setuptools.command.build_ext import build_ext


IS_WINDOWS = sys.platform == 'win32'


def find_in_path(name, path):
    """Find a file in a search path"""
    # Adapted fom
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found, everything
    is based on finding 'nvcc' in the PATH.
    """

    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # otherwise, search the PATH for NVCC
        default_path = pjoin(os.sep, 'usr', 'local', 'cuda', 'bin')
        nvcc = find_in_path(
            'nvcc', os.environ['PATH'] + os.pathsep + default_path)
        if nvcc is None:
            print(
                'The nvcc binary could not be located in your $PATH. '
                'Either add it to your path, or set $CUDAHOME')
            return None
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaConfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib/x64' if IS_WINDOWS else 'lib64')}
    for k, v in cudaConfig.items():
        if not os.path.exists(v):
            print('The CUDA %s path could not be located in %s' % (k, v))
            return None
    return cudaConfig


CUDA = locate_cuda()
COMMON_NVCC_FLAGS = [
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
]


def _is_cuda_file(path):
    return os.path.splitext(path)[1] in ['.cu', '.cuh']


class BuildExtension(build_ext, object):
    """
    A custom :mod:`setuptools` build extension .

    This :class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++11``) as well as mixed
    C++/CUDA compilation (and support for CUDA files in general).

    When using :class:`BuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages (``cxx`` or ``cuda``) to a list of additional compiler flags to
    supply to the compiler. This makes it possible to supply different flags to
    the C++ and CUDA compiler during mixed compilation.
    """

    @classmethod
    def with_options(cls, **options):
        '''
        Returns an alternative constructor that extends any original keyword
        arguments to the original constructor with the given options.
        '''
        def init_with_options(*args, **kwargs):
            kwargs = kwargs.copy()
            kwargs.update(options)
            return cls(*args, **kwargs)
        return init_with_options

    def __init__(self, *args, **kwargs):
        super(BuildExtension, self).__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

    def build_extensions(self):
        # self._check_abi()
        # for extension in self.extensions:
        #     self._add_compile_flag(extension, '-DTORCH_API_INCLUDE_EXTENSION_H')
        #     self._define_torch_extension_name(extension)
        #     self._add_gnu_cpp_abi_flag(extension)

        # Register .cu and .cuh as valid source extensions.
        self.compiler.src_extensions += ['.cu', '.cuh']
        # Save the original _compile method for later.
        if self.compiler.compiler_type == 'msvc':
            self.compiler._cpp_extensions += ['.cu', '.cuh']
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile

        def unix_wrap_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_cuda_file(src):
                    nvcc = CUDA['nvcc']
                    # nvcc = _join_cuda_home('bin', 'nvcc')
                    if not isinstance(nvcc, list):
                        nvcc = [nvcc]
                    self.compiler.set_executable('compiler_so', nvcc)
                    if isinstance(cflags, dict):
                        cflags = cflags['nvcc']
                    cflags = COMMON_NVCC_FLAGS + \
                             ['--compiler-options', "'-fPIC'"] + cflags
                elif isinstance(cflags, dict):
                    cflags = cflags['cxx']
                # NVCC does not allow multiple -std to be passed, so we avoid
                # overriding the option if the user explicitly passed it.
                if not any(flag.startswith('-std=') for flag in cflags):
                    cflags.append('-std=c++11')

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable('compiler_so', original_compiler)

        def win_wrap_compile(sources,
                             output_dir=None,
                             macros=None,
                             include_dirs=None,
                             debug=0,
                             extra_preargs=None,
                             extra_postargs=None,
                             depends=None):

            self.cflags = copy.deepcopy(extra_postargs)
            extra_postargs = None

            def spawn(cmd):
                # Using regex to match src, obj and include files
                src_regex = re.compile('/T(p|c)(.*)')
                src_list = [
                    m.group(2) for m in (src_regex.match(elem) for elem in cmd)
                    if m
                ]

                obj_regex = re.compile('/Fo(.*)')
                obj_list = [
                    m.group(1) for m in (obj_regex.match(elem) for elem in cmd)
                    if m
                ]

                include_regex = re.compile(r'((\-|\/)I.*)')
                include_list = [
                    m.group(1)
                    for m in (include_regex.match(elem) for elem in cmd) if m
                ]

                if len(src_list) >= 1 and len(obj_list) >= 1:
                    src = src_list[0]
                    obj = obj_list[0]
                    if _is_cuda_file(src):
                        nvcc = CUDA['nvcc']
                        # nvcc = _join_cuda_home('bin', 'nvcc')
                        if isinstance(self.cflags, dict):
                            cflags = self.cflags['nvcc']
                        elif isinstance(self.cflags, list):
                            cflags = self.cflags
                        else:
                            cflags = []
                        cmd = [
                            nvcc, '-c', src, '-o', obj, '-Xcompiler',
                            '/wd4819', '-Xcompiler', '/MD'
                        ] + include_list + cflags
                    elif isinstance(self.cflags, dict):
                        cflags = self.cflags['cxx'] + ['/MD']
                        cmd += cflags
                    elif isinstance(self.cflags, list):
                        cflags = self.cflags + ['/MD']
                        cmd += cflags

                return original_spawn(cmd)

            try:
                self.compiler.spawn = spawn
                return original_compile(sources, output_dir, macros,
                                        include_dirs, debug, extra_preargs,
                                        extra_postargs, depends)
            finally:
                self.compiler.spawn = original_spawn

        # Monkey-patch the _compile method.
        if self.compiler.compiler_type == 'msvc':
            self.compiler.compile = win_wrap_compile
        else:
            self.compiler._compile = unix_wrap_compile

        build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        # Get the original shared library name. For Python 3, this name will be
        # suffixed with "<SOABI>.so", where <SOABI> will be something like
        # cpython-37m-x86_64-linux-gnu. On Python 2, there is no such ABI name.
        # The final extension, .so, would be .lib/.dll on Windows of course.
        ext_filename = super(BuildExtension, self).get_ext_filename(ext_name)
        # If `no_python_abi_suffix` is `True`, we omit the Python 3 ABI
        # component. This makes building shared libraries with setuptools that
        # aren't Python modules nicer.
        if self.no_python_abi_suffix and sys.version_info >= (3, 0):
            # The parts will be e.g.
            # ["my_extension", "cpython-37m-x86_64-linux-gnu", "so"].
            ext_filename_parts = ext_filename.split('.')
            # Omit the second to last element.
            without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
            ext_filename = '.'.join(without_abi)
        return ext_filename


def CppExtension(name, sources, *args, **kwargs):
    '''
    Creates a :class:`setuptools.Extension` for C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a C++ extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> from setuptools import setup
        >>> from cvtools.utils.cpp_extension import BuildExtension, CppExtension
        >>> setup(
                name='extension',
                ext_modules=[
                    CppExtension(
                        name='extension',
                        sources=['extension.cpp'],
                        extra_compile_args=['-g']),
                ],
                cmdclass={
                    'build_ext': BuildExtension
                })
    '''
    include_dirs = kwargs.get('include_dirs', [])
    # include_dirs += include_paths()
    kwargs['include_dirs'] = include_dirs

    if IS_WINDOWS:
        library_dirs = kwargs.get('library_dirs', [])
        # library_dirs += library_paths()
        kwargs['library_dirs'] = library_dirs

    kwargs['language'] = 'c++'
    return setuptools.Extension(name, sources, *args, **kwargs)


def CUDAExtension(name, sources, *args, **kwargs):
    '''
    Creates a :class:`setuptools.Extension` for CUDA/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a CUDA/C++
    extension. This includes the CUDA include path, library path and runtime
    library.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor.

    Example:
        >>> from setuptools import setup
        >>> from cvtools.utils.cpp_extension import BuildExtension, CUDAExtension
        >>> setup(
                name='cuda_extension',
                ext_modules=[
                    CUDAExtension(
                            name='cuda_extension',
                            sources=['extension.cpp', 'extension_kernel.cu'],
                            extra_compile_args={'cxx': ['-g'],
                                                'nvcc': ['-O2']})
                ],
                cmdclass={
                    'build_ext': BuildExtension
                })
    '''
    library_dirs = kwargs.get('library_dirs', [])
    library_dirs.append(CUDA['lib64'])
    kwargs['library_dirs'] = library_dirs

    runtime_library_dirs = kwargs.get('runtime_library_dirs', [])
    runtime_library_dirs.append(CUDA['lib64'])
    kwargs['runtime_library_dirs'] = runtime_library_dirs

    libraries = kwargs.get('libraries', [])
    libraries.append('cudart')
    if IS_WINDOWS:
        libraries.append('c10')
        libraries.append('c10_cuda')
    kwargs['libraries'] = libraries

    include_dirs = kwargs.get('include_dirs', [])
    include_dirs += [np.get_include(), CUDA['include']]
    kwargs['include_dirs'] = include_dirs

    kwargs['language'] = 'c++'

    return setuptools.Extension(name, sources, *args, **kwargs)
