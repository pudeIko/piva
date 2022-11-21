import setuptools

with open('requirements.txt', 'r') as fh:
    requirements = fh.read()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='piva',
    version='1.0.3a0',
    author='Wojtek Pudelko',
    author_email='wojciech.pudelko@psi.ch',
    description='My piva, have fun',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pudeIko/piva.git',
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'piva2d = piva.main:open2D',
            'piva3d = piva.main:open3D',
            'h5_to_pickle = piva.main:pickle_h5',
            'reshape_pickled = piva.main:reshape_pickled',
            'db = piva.main:db'
        ],
    }
)
