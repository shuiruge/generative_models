#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


NAME = 'genmod'
DESCRIPTION = 'Generative models'
AUTHOR = 'shuiruge'
AUTHOR_EMAIL = 'shuiruge@hotmail.com'
URL = 'https://github.com/shuiruge/generative_models'
VERSION = '0.1.0'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license='MIT',
    url=URL,
    packages=find_packages(exclude=[
        'tests.*', 'tests',
        'examples.*', 'examples',
        'dat.*', 'dat']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3+',
    ],
    zip_safe=False,
)
