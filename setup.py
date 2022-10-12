# encoding: utf-8
"""
@author: andy
@file: setup.py
@time: 2022/2/11 上午9:34
@desc:
"""
from setuptools import setup, find_packages
from os import path

PROJECT_DIR = path.dirname(path.abspath(__file__))

INSTALL_PACKAGES = open(path.join(PROJECT_DIR, 'requirements.txt')).read().splitlines()

README = open(path.join(PROJECT_DIR, 'README.md')).read()

setup(
    name='electriceel',
    packages=find_packages(include=["electriceel"]),
    description="Model electriceel for model feature metric",
    long_description=README,
    long_description_content_type='text/markdown',
    install_requires=INSTALL_PACKAGES,
    version='0.0.1',
    url='https://git.ty.ink/zhangpengfei/electric-eel',
    author='zhangpengfei',
    author_email='1241833581@qq.com',
    keywords=['model-electriceel', 'scikit-learn', 'model', 'feature', 'metric'],
    include_package_data=True,
    python_requires='>=3'
)