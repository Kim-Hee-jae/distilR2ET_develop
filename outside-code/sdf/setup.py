import torch
from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []


nvcc_flags = [
    '-allow-unsupported-compiler',
    '-gencode=arch=compute_89,code=sm_89',   
    '-Xcompiler', '/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH',
]

ext_modules = [
    CUDAExtension(
        'sdf.csrc',
        sources=[
            'sdf/csrc/sdf_cuda.cpp',
            'sdf/csrc/sdf_cuda_kernel.cu',
        ],
        extra_compile_args={'cxx': [], 'nvcc': nvcc_flags}
    ),
]

setup(
    description='PyTorch implementation of SDF loss',
    author='Nikos Kolotouros',
    author_email='nkolot@seas.upenn.edu',
    license='MIT License',
    version='0.0.1',
    name='sdf_pytorch',
    packages=['sdf', 'sdf.csrc'],
    install_requires=INSTALL_REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)
