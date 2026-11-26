"""Pre-builds the alpha-blend extension used by PutBack:  python setup_blend.py build_ext --inplace

Upstream compiles it lazily with -march=native, which bakes the build machine's CPU features into the binary.
"""
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

setup(
    name="ditto-blend",
    ext_modules=cythonize(
        [
            Extension(
                "core.utils.blend.blend",
                sources=["core/utils/blend/blend.pyx", "core/utils/blend/blend_impl.c"],
                include_dirs=[np.get_include(), "core/utils/blend"],
                extra_compile_args=["-O3", "-std=c99", "-ffast-math"],
            )
        ],
        language_level=3,
    ),
)
