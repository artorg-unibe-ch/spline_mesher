from setuptools import setup
from Cython.Build import cythonize

setup(
    name="find_closed_curve",
    ext_modules=cythonize("src/cython_functions/find_closed_curve.pyx"),
)

if __name__ == "__main__":
    setup()
