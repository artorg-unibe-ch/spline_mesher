from setuptools import setup
from Cython.Build import cythonize

setup(
    name="spline_mesher",
    packages=["spline_mesher", "cython_functions"],
    ext_modules=cythonize("src/cython_functions/find_closed_curve.pyx"),
)
