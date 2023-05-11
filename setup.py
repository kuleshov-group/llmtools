from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension

setup(
    name='llmtune',
    version='0.1.0',
    packages=find_packages(include=['llmtune', 'llmtune.*']),
    entry_points={
        'console_scripts': ['llmtune=llmtune.run:main']
    }
)

setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'quant_cuda', 
        [
        	'llmtune/engine/quant/cuda/quant_cuda.cpp', 
        	'llmtune/engine/quant/cuda/quant_cuda_kernel.cu'
        ]
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
