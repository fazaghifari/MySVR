'''
Author: Ghifari Adam Faza <ghifariadamf@gmail.com>
This package is distributed under MIT license.
'''

from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
LONG_DESCRIPTION= """
MySVR is a Support Vector Regression (SVR) package with multi-kernel feature. Written with a simple style, 
this package is suitable for anyone who wish to learn about SVR implementation in Python.

Author: Kemas Zakaria and Ghifari Adam F
"""


metadata = dict(
    name='mysvr',
    version='1.0.0',
    description='Support Vector Regression (SVR) package.',
    long_description=LONG_DESCRIPTION,
    author='Kemas Zakaria and Ghifari Adam F',
    author_email='ghifariadamf@gmail.com',
    license='MIT',
    packages=[
        'svr',
        'svr/examples'
    ],
    install_requires=[
        'numpy',
        'scipy',
        'cvxopt',
        'sobolsampling'
    ],
    python_requires='>=3.6.*',
    zip_safe=False,
    include_package_data=True,
    url = 'https://github.com/fazaghifari/MySVR', # use the URL to the github repo
    download_url = 'https://github.com/fazaghifari/MySVR/releases',
)

setup(**metadata)
