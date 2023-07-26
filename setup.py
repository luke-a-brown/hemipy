# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 17:09:40 2022

@author: Luke Brown
"""

from distutils.core import setup

setup(
    name = 'hemipy',
    packages = ['hemipy'],
    version = '0.1.2',
    description = 'A Python module for automated estimation of forest biophysical variables and uncertainties from digital hemispherical photographs',
    author = 'Luke A. Brown',
    author_email = 'l.a.brown4@salford.ac.uk',
    url = 'https://github.com/luke-a-brown/hemipy',
    install_requires = ['rawpy',
                        'numpy',
                        'scikit-image',
                        'scipy',
                        'imageio',
                        'uncertainties']
)