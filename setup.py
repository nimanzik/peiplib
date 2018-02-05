import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='peip',
    version='0.0.1',
    description='PEIP : Parameter Estimation and Inverse Problems '
                '(Aster et al., 2nd edition, 2011) Examples and Lib',
    long_description=read('README.md'),
    url='https://gitlab.com/nimanshr/PEIP',
    author='Nima Nooshiri',
    author_email='nima.nooshiri@gfz-potsdam.de; '
                 'nima.nooshiri@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering'],
    keywords='parameter-estimation inverse-problem inversion',
    install_requires=[
        'numpy>=1.13.1',
        'scipy>=0.19.1'],
    packages=['peiplib'],
    package_dir={'peiplib': 'peiplib'})
