from setuptools import setup, find_packages

setup(
    name='somegrad',
    version='0.1.0',
    description='A simple autograd engine and neural network library',
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
