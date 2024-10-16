"""
对引擎的打包
"""

import subprocess

from setuptools import setup, find_packages

setup(name='sawtooth-pote-consensus',
      version="0.1",
      description='Sawtooth PoTE Consensus Module',
      packages=find_packages(),
      install_requires=[
          'requests',
          'sawtooth-sdk',
      ],
      entry_points={
          'console_scripts': [
              'pote-engine = sawtooth_pote.main:main'
          ]
      })
