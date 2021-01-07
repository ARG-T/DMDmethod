from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

"""
sourceの名前を一致させる!!!
"""

ext_module = Extension(
    name="free_energy_culc",
    sources=["free_energy_culc.pyx"],
)

setup(
    name='free_energy_culc',
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_module],
    include_dirs=[numpy.get_include()],
)