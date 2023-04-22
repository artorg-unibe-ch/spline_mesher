from setuptools import setup
from Cython.Build import cythonize

setup(
    version="0.0.1",
    packages=["spline_mesher", "cython_functions"],
    ext_modules=cythonize("src/cython_functions/find_closed_curve.pyx"),
)
