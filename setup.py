from setuptools import setup
from Cython.Build import cythonize

setup(
    version="0.1.0",
    packages=["pyhexspline", "cython_functions"],
    ext_modules=cythonize("src/cython_functions/find_closed_curve.pyx"),
)
