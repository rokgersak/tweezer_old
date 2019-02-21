from setuptools import setup, find_packages

packages = find_packages()

import tweezer

setup(name = 'tweezer',
      version = tweezer.__version__,
      description = 'Optical tweezer tools',
      packages = packages,
      #include_package_data=True
      package_data={
        # If any package contains *.dat, or *.ini include them:
        '': ['*.dat',"*.ini"]}
      )