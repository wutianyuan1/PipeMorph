from setuptools import setup
from torch.utils import cpp_extension

setup(name='memcpy',
      ext_modules=[cpp_extension.CUDAExtension('memcpy', ['memcpy.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

# from setuptools import setup, Extension
# from torch.utils import cpp_extension

# setup(name='memcpy',
#       ext_modules=[cpp_extension.CppExtension('memcpy', ['memcpy.cpp'])],
#       cmdclass={'build_ext': cpp_extension.BuildExtension})