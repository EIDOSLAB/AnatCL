from setuptools import setup, find_packages

setup(
    name='anatcl',
    version='0.0.1',
    url='https://github.com/carloalbertobarbano/AnatCL',
    author='Carlo Alberto Barbano',
    author_email='carlo.barbano@unito.it',
    description='AnatCL',
    packages=find_packages(),
    install_requires=['torch >= 2.2.1'],
)