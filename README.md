# PIVA - Python Interactive Viewer for Arpes

![](https://raw.githubusercontent.com/pudeIko/piva/master/doc/img/piva_showcase.gif)

PIVA is a GUI application based on `PyQt5` and `pyqtgraph` toolkits, designed 
for interactive and intuitive examination of large, image-like datasets. 
Although it is generally capable of displaying any multidimensional data, a 
vast majority of the implemented functionalities are geared towards users 
performing Angle-Resolved Photoemission Spectroscopy (ARPES) experiments.

A number of standard methods and image processing algorithms (momentum-space 
conversion, energy distribution curve symmetrization and 
derivative analysis, *etc.*) are accessible from the GUI level. However,
discrepancies in ARPES results between different physical systems force 
experimenters to implement unique analysis methods for virtually every 
investigated system. For that reason, PIVA contains only a limited amount of 
deep-analysis procedures, as in every case they would have to be tailored to 
match specific needs of the user. Instead, it includes straightforward tools 
for exporting loaded datasets to the ``jupyter-lab`` notebook, where one can 
perform more sophisticated analysis, allowing to take full advantage of the 
wonders that `numpy` and `scipy` provide.


## Installation 

The installation of PIVA has been tested on macOS, Windows and Linux.

The easiest way to install the package is to use `pip`. Just type the following 
on a command line:

    pip install piva

or directly from this repo:

    pip install git+https://github.com/pudeIko/piva.git


## Documentation 

More details on the installation and full description of the package can be 
found on the [documentation](https://piva.readthedocs.io/en/latest/) site.
