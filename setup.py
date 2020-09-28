from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
"""
sourceの名前を一致させる!!!
"""

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
                    Extension("c_inte_test", sources=["c_inte_test.pyx"], include_dirs=[numpy.get_include()],)
                    ]
)