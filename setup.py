from setuptools import setup, find_packages

setup(
    name='somegrad',
    version='0.1.0',
    description='A lightweight, hardware-aware autograd engine and deep learning library built from scratch, inspired by micrograd and tinygrad.',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    extras_require={
        'examples': [
            'jupyter',
            'matplotlib',
        ],
        'test': [
            'pytest',
        ],
    },
    python_requires='>=3.8',
)
