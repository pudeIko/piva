from setuptools import setup, find_packages

with open('requirements.txt', 'r') as fh:
    requirements = fh.read()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='piva',
    version='2.2.0',
    author='Wojtek Pudelko',
    author_email='wojciech.pudelko@psi.ch',
    description='PIVA - Photoemission Interface for Visualization and Analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pudeIko/piva.git',
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'db = piva.main:db'],
    }
)
