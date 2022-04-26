# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools import find_packages

version_dict = {}
exec(open('src/regcma/constant.py').read(), version_dict)
VERSION = version_dict["VERSION"]

setup(
    name='regcma',
    version=VERSION,
    author='Yuji KOGUMA',
    description='A Python implementation of Regulated Evolution Strategies with Covariance Matrix Adaption for continuous "Black-Box" optimization problems.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.7',
    install_requires=open('requirements.txt').read().splitlines()
)
