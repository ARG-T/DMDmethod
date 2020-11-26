from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

"""
sourceの名前を一致させる!!!
"""

ext_module = Extension(
    name="c_free_energy",
    sources=["c_free_energy.pyx"],
    extra_compile_args=['/openmp'],
)

setup(
    name='c_free_energy',
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_module],
    include_dirs=[numpy.get_include()],
)
