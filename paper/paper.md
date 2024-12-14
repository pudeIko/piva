---
title: 'PIVA: Photoemission Interface for Visualization and Analysis'
tags:
  - Python
  - Angle-resolved photoemission spectroscopy
  - Experimental physics
  - Data visualization
  - Data analysis
authors:
  - name: Wojciech R. Pudelko
    orcid: 0000-0001-5889-4594
    affiliation: "1, 2"
  - name: Kevin Kramer
    orcid: 0000-0001-5523-6924
    affiliation: 2
  - name: Julia Küspert
    orcid: 0000-0002-2905-9992
    affiliation: 2
  - name: Johan Chang
    orcid: 0000-0002-4655-1516
    affiliation: 2
  - name: Nicholas C. Plumb
    orcid: 0000-0002-2334-8494
    affiliation: 1
affiliations:
  - name: Swiss Light Source, Paul Scherrer Institut, Forschungstrasse 111, CH-5232 Villigen PSI, Switzerland
    index: 1
  - name: Physik Institut, Universität Zürich, Winterthurerstrasse 190, CH-8057 Zürich, Switzerland
    index: 2
date: 2 December 2024
bibliography: paper.bib
---



# Summary

Angle-resolved photoemission spectroscopy (ARPES) is a powerful probe for 
elucidating the electronic structures of quantum materials. As the technique's 
throughput continues to increase, there is a pressing need for specialized 
tools that enable rapid and intuitive data inspection and analysis. The PIVA 
package introduced in this work addresses this need by providing an 
interactive interface specifically designed for the efficient visualization 
and examination of large volumes of multidimensional datasets. It integrates 
image-processing methods, measurement tools, and a comprehensive suite of 
analysis routines within an intuitive environment. Together, these 
capabilities offer a robust framework for conducting detailed ARPES 
investigations, significantly enhancing the efficiency and depth of data 
exploration.



# Statement of need

Over the past few decades, our ability to collect data has rapidly outpaced 
our capacity to process it efficiently and extract meaningful insights. This 
trend is evident not only in the broad field of data science but also across 
many areas of experimental research, where specialized tools are essential to 
manage the ever-increasing speed and scale of data acquisition. 

The challenges of analyzing large amounts of complex data certainly arise 
for physicists utilizing angle-resolved 
photoemission spectroscopy (ARPES), a technique that directly probes the 
electronic band structure of crystal materials [@sobota2021].
The overall significance of ARPES in the field of condensed matter physics 
can hardly be exaggerated, as it has provided vital insights into numerous 
important topics, like unconventional superconductivity [@damascelli2003], 
topological states of matter [@dil2019], and novel forms of magnetism 
[@krempasky2024]. These scientific discoveries would not have been possible 
without significant advancements in instrumentation and technical capabilities
[@wannberg2009; @zhou2018; @koralek2007; @strocov2014], each setting a 
milestone in data acquisition efficiency.

The increasing speed of the measurement process brings, however, unique 
challenges to data handling and analysis in ARPES investigations. Namely, 
researchers require flexible tools to quickly glean insights from the data 
during an ongoing, dynamic experiment, as well as to perform deeper data 
exploration and analysis later on. PIVA is a package specifically designed to 
meet these needs. In comparison 
to other available solutions 
[@stansbury2020; @arpes_gui_antares; @rotenberg2021], its primary objective is 
to enhance the efficiency of ARPES 
data inspection and analysis by offering interactive and intuitive tools 
tailored to manage large amounts of multidimensional datasets 
simultaneously—all without relying on proprietary environments. While the 
content of this manuscript describes PIVA's implementation and structure, 
more details and examples can be found in the project's documentation 
[@piva_doc].



# Software description
The PIVA package is an open-source, Python-based software. Its data handling 
and analysis components utilize standard numerical and scientific libraries, 
including NumPy, SciPy, Pandas, Matplotlib, and Numba [@numpy; @scipy; 
@pandas; @numba; @matplotlib]. The GUI applications are built with the PyQt5 and 
pyqtgraph toolkits [@pyqt; @pyqtgraph]. \autoref{overview} depicts 
PIVA's general structure and its individual components.

![Schematic of the PIVA package showing its components and workflow within the 
software.
  \label{overview}
](1-overview.pdf){ width=99% }

The package consists of two main components. The first part is a graphical 
user interface (GUI) application designed for quick data visualization and 
inspection during time-critical measurements. The GUI utilizes interactive 
graphics tools based on the `data-slicer` package [@kramer2021]. 
It allows one to navigate through the acquired data files, open selected data 
files in interactive applications, conveniently browse through 
multidimensional datasets, and carry out preliminary analysis. In the next 
steps, the users can take advantage of PIVA's collection of analysis methods 
and conduct the second part—detailed scrutiny of the recorded spectra. This 
can be performed within an environment of the user's choice or by using 
built-in tools to transfer the workspace to a Jupyter notebook.

## Data handling

While handling the acquired data, in the first step, raw files need to be read 
and loaded into a uniform format. Since data formats used by various 
experimental setups differ from lab to lab, dedicated file-loading scripts are 
required for each system. Within PIVA, this task is handled by the 
`data_loader` module, which imports the data and metadata into a 
uniform `Dataset` object, inheriting from the `Namespace` 
class [@namespace]. The package includes specific `Dataloader` 
classes already implemented for numerous synchrotron sources around the world, 
which can also be easily imported outside the interactive environment and used 
to run custom analysis routines. Consequently, adopting a single, standardized 
format facilitates convenient transfer of loaded datasets between other PIVA 
modules and utilities and significantly simplifies data handling when manual 
processing is required.

![**Overview of the `DataViewers`**. 
    **a-c** Appearance of the 2D, 3D and 4D `DataViewer`, respectively. 
Interactive sliders are marked with yellow lines. The main graphical 
components are highlighted with dashed lines as follows: angle-energy 
detector image (green), constant energy map (cyan), energy slider for the 
constant energy map (red), and manipulator raster scan (magenta). 
The utilities panel, displayed only in the case of the 2D viewer 
(panel **a**), is highlighted with a solid light-blue box.
  \label{viewers}
](2-viewers.pdf){ width=74% }


## Interactive tools

The main visualization applications are three `DataViewers`, designed 
to handle 2-, 3- and 4-dimensional datasets, depending on the scan mode in 
which they were acquired. Their main components and layout are presented in 
\autoref{viewers}. All three `DataViewers` consist of a set of 
plotting panels and the utilities panel. The former consists of image/curve 
plots (depending on the dimensionality of the extracted cut) and draggable 
sliders that allow one to freely browse through datasets and display slices 
along the corresponding directions. The utilities panel contains numerous 
functionalities for data exploration and visualization including, _e.g._, 
averaging over selected direction and range, normalizing along specific axes, 
and applying experimental corrections.

Additional interactive tools include the `EDCFitter` and  the `MDCFitter`, 
suitable for scrutinizing 1D energy or momentum distribution curves, 
respectively, and 
`PlotTool`, a convenient utility that allows the creation of customized 
figures by means of combining different plots.


## Analysis modules

Unlike many other experimental techniques, ARPES data and analysis can vary 
widely depending on the system under investigation and the information to be 
extracted. As a result, deep analysis necessitates meticulous and highly 
specialized data handling, which would be significantly limited when 
encapsulated within a purely graphical interface. For that reason, apart from 
interactive applications, PIVA provides an extensive library of functions and 
procedures that can easily be imported into an environment of the user's 
choice. (As a recommended option, the package includes tools for exporting 
datasets to Jupyter notebooks.) There, users can conduct detailed fitting and 
more sophisticated analysis using various methods already implemented within 
the analysis module or apply their own routines, requiring hands-on scripting.


## Modularity

PIVA contains built-in procedures for importing modules and plugins written 
by users. This includes data-loading scripts (for file formats not 
implemented within the package), GUI applications to elevate interactive data 
processing capabilities or any other utilities 
that one might want to incorporate. Once implemented, the 
extensions are automatically loaded and ready to be used at the beginning of 
each session. A comprehensive description and examples of how to implement and 
configure such extensions are provided in the documentation [@piva_doc].
Notably, while the current functionalities are primarily designed for ARPES, 
custom plugins can be developed to manage and visualize various other 
types of data, thereby expanding PIVA's applicability to a wide range of 
experimental techniques that produce image-like results.



# Conclusions and outlook

In conclusion, PIVA was developed to address bottlenecks in users' ability to 
process ARPES data. The `DataViewers` enable efficient visualization 
and preliminary analysis of a broad range of multidimensional datasets. 
Further detailed analysis can be performed using dedicated interactive 
`Fitters` or by utilizing implemented analysis methods. Furthermore, 
the package addresses an essential aspect of data format inhomogeneity arising 
from varying conventions across sources and instruments, by introducing a 
standardized `Dataset` object. Thanks to these advantages, PIVA has been 
utilized by external users and staff members of the SIS beamline at the Swiss 
Light Source and employed in numerous experiments, some of which have already 
been documented in published works 
[@wang2022; @hu2023; @soh2024; @pudelko2024].

Although PIVA's combination of features fulfills most of the everyday 
needs of ARPES experimenters, further developments are planned to expand its 
capabilities. Foreseen improvements include implementing data loading scripts 
for additional beamlines and instruments, expanding the library of 
postprocessing methods with analysis routines and machine learning algorithms 
for noise reduction, and developing new interactive applications tailored for 
other variants of ARPES, such as spin-resolved measurements.



# Acknowledgements

W.R.P. and N.C.P. acknowledge support from the Swiss National Science 
Foundation through Project Number 200021_185037.



# References
