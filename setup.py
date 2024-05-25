from setuptools import setup, find_packages, Extension
from torch.utils import cpp_extension


setup(
    ext_modules=[
        cpp_extension.CUDAExtension(
            'quant_cuda', 
            [
                'kernels/quant_cuda/quant_cuda.cpp', 
                'kernels/quant_cuda/quant_cuda_kernel.cu'
            ]
        ),
        cpp_extension.CUDAExtension(
            'quiptools_cuda',
            [
                'kernels/quiptools_cuda/quiptools_wrapper.cpp',
                'kernels/quiptools_cuda/quiptools.cu',
                'kernels/quiptools_cuda/quiptools_e8p_gemv.cu'
            ],
            extra_compile_args={
                'cxx': ['-g', '-lineinfo'],
                'nvcc': ['-O2', '-g', '-Xcompiler', '-rdynamic', '-lineinfo']
            }
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
