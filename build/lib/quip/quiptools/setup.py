from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='quiptools_cuda',
      ext_modules=[cpp_extension.CUDAExtension('quiptools_cuda', ['quiptools_wrapper.cpp', 'quiptools.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

