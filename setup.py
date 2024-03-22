from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension

setup(
    name='llmtools',
    version='0.1.0',
    packages=find_packages(include=['llmtools', 'llmtools.*']),
    entry_points={
        'console_scripts': ['llmtools=llmtools.run:main']
    }
)

setup(
    name='quip',
    version='0.1.0',
    packages=find_packages(where='third-party'),
    package_dir={'': 'third-party'}, 
)

setup(
    name='quant_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'quant_cuda', 
        [
        	'llmtools/engine/inference/cuda/quant_cuda.cpp', 
        	'llmtools/engine/inference/cuda/quant_cuda_kernel.cu'
        ]
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)

