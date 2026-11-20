from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="query_adaptive_quantization_ext",
    ext_modules=[
        cpp_extension.CUDAExtension("query_adaptive_quantization_ext", ["main.cu"])
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
