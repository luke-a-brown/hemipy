# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 17:09:40 2022

@author: Luke Brown
"""

from distutils.core import setup

setup(
    name = 'hemipy',
    packages = ['hemipy'],
    version = '0.1.0',
    description = 'A Python module for automated and traceable estimation of forest biophysical variables from digital hemispherical photographs',
    author = 'Luke A. Brown',
    author_email = 'l.a.brown4@salford.ac.uk',
    url = 'https://github.com/lukebrownuk/HemiPy',
    install_requires = ['rawpy',
                        'numpy',
                        'scikit-image',
                        'scipy',
                        'uncertainties']
)