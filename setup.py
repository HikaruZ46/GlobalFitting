from setuptools import setup, find_packages

setup(
    name='GlobalFitting',
    version='0.0.2',
    author='Hikaru Nozawa',
    description='Global fitting of Gaussian mixture model',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
    ],
    license='Apache License 2.0'
)