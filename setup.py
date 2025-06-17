import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='raft',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension('cuda_corr',
            sources=['src/model/raft_gaussian_splat/altcorr/correlation.cpp', 'src/model/raft_gaussian_splat/altcorr/correlation_kernel.cu'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': ['-O3'],
            }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })