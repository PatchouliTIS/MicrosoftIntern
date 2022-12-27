#
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import os
import sys
import sysconfig

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

from distutils.sysconfig import get_python_lib

install_requires = ["cython"]

def search_on_path(filename):
    for p in os.environ.get('PATH', '').split(os.pathsep):
        fullname = os.path.join(p, filename)
        if os.path.exists(fullname):
            return os.path.abspath(fullname)
    return None

def get_cuda_path():
    cuda_path = os.environ.get('CUDA_PATH', '')
    if os.path.exists(cuda_path):
        return cuda_path
    cuda_path = os.environ.get('CUDA_HOME', '')
    if os.path.exists(cuda_path):
        return cuda_path
    nvcc_path = search_on_path('nvcc')
    if nvcc_path is not None:
        cuda_path = os.path.normpath(
            os.path.join(os.path.dirname(nvcc_path), '..'))
        return cuda_path
    cuda_path = '/usr/local/cuda'
    if os.path.exists(cuda_path):
        return cuda_path
    return None

include_dirs = [
    os.path.dirname(sysconfig.get_path('include')),
]
library_dirs = [
    get_python_lib(),
    os.path.join(os.sys.prefix, 'lib'),
]

include_dirs.append('../include')
library_dirs.append('../lib')

cuda_path = get_cuda_path()
if cuda_path is not None:
    include_dirs.append(os.path.join(cuda_path, 'include'))
    library_dirs.append(os.path.join(cuda_path, 'lib64'))
    library_dirs.append(os.path.join(cuda_path, 'lib'))

print('# include_dirs: {}'.format(include_dirs))
print('# library_dirs: {}'.format(library_dirs))

extensions = cythonize(
    [
        Extension(
            "cuann.libcuann",
            sources=['bindings/libcuann.pyx',],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=[],
            libraries=['cuann',],
            language='c++',
            extra_compile_args=['-fopenmp',],
            extra_link_args=['-fopenmp',],
        )
    ]
)

setup(
    name='cuann',
    version='0.0.7',
    description='GPU ANN library using IVFPQ',
    author='Akira Naruse',
    author_email='anaruse@nvidia.com',
    py_modules = ['cuann.ivfpq', 'cuann.utils'],
    ext_modules = extensions,
)
