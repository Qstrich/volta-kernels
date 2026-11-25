from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(name="volta_ext", ext_modules=[CUDAExtension("volta_ext", ["triad.cu", "attn_fwd.cu"],
      extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-maxrregcount=128"]})], cmdclass={"build_ext": BuildExtension})
