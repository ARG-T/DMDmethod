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
                    Extension("c_free_energy", sources=["c_free_energy.pyx"], include_dirs=[numpy.get_include()],)
                    ]
)