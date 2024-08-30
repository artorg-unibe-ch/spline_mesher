Installation
============

1. Clone the repo

   .. code:: sh

      git clone https://github.com/simoneponcioni/spline-mesher.git

2. Install dependencies

   .. code:: sh

      conda create -n meshenv python=3.9 --file requirements.txt
      conda activate meshenv
      python setup.py build_ext --inplace
      python setup.py install

3. For developers: install the package in editable mode and install
   requirements for testing

   .. code:: sh

      pip install -e .
      pip install -r requirements-dev.txt
