"""
打包 sawtooth-edge-storage 交易处理器
"""
from __future__ import print_function

import os
import subprocess

from setuptools import setup, find_packages

conf_dir = "/etc/sawtooth"


setup(
    name='sawtooth-es',
    version='1.0',
    description='Sawtooth ES Example',
    author='Hyperledger Sawtooth',
    packages=find_packages(),
    install_requires=[
        'aiohttp',
        'colorlog',
        'protobuf',
        'sawtooth-sdk',
        'PyYAML',
    ],
    entry_points={
        'console_scripts': [
            'es = sawtooth_edge_storage.es_cli:main_wrapper',
            'es-tp-python = sawtooth_edge_storage.processor.main:main',
        ]
    })


SGA-MARL