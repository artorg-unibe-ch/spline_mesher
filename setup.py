from setuptools import find_packages, setup
from Cython.Build import cythonize

setup(
    version="1.0.2",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=cythonize(
        "src/cython_functions/find_closed_curve.pyx",
        compiler_directives={"language_level": "3"},
    ),
)
