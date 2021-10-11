import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='peip',
    version='0.0.1',
    description='peiplib : Python codes and jupyter notebooks for '
                'examples for the 2nd edition of Parameter Estimation '
                'and Inverse Problems.',
    long_description=read('README.md'),
    url='https://gitlab.com/nimanzik/peiplib',
    author='Nima Nooshiri',
    author_email='nima.nooshiri@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering'],
    keywords='parameter-estimation inverse-problem inversion',
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.2',
        'scipy>=1.7.1',
        'matplotlib>=3.4.3'],
    packages=['peiplib'],
    package_dir={'peiplib': 'peiplib'})
