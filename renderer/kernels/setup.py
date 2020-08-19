from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name="rasterize_triangles_cpp",
      ext_modules=[
            CUDAExtension(
                  "rasterize_triangles_cpp", ["rasterize_triangles.cpp", "rasterize_triangles_kernel.cu"]),
      ],
      cmdclass={"build_ext": BuildExtension}
)
