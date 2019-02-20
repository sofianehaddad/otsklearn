#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
from setuptools import setup, find_packages
 
# Get the version from __init__.py
with open('otsklearn/__init__.py') as fid:
    for line in fid:
        if line.startswith('__version__'):
            version = line.strip().split()[-1][1:-1]
            break
 
setup(
 
    # library name
    name='otsklearn',
 
    # code version
    version=version,
    
    # list libraries to be imported
    packages=find_packages(),
 
    # Descriptions
    description="Expose OpenTURNS metamodels with SciKit-Learn API",
    long_description=open('README.rst').read(),
 
    # List of dependancies
    install_requires= ['scikit-learn>=0.17']
 
    # Enable to take into account MANIFEST.in
    # include_package_data=True,
)
