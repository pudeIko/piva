.. _sec-file-formats:

File formats
============

This section discusses what it means for a datafile to be *recognizable* and 
thus *readable* by :mod:`piva`.

With ARPES data, the tendency seems to be that each beamline and each lab uses 
their own file format and conventions.
Therefore, in order to be able to handle these different files there really 
is no other way than to have a dedicated script or program which converts 
these into a common form.

Within :mod:`piva`, this is taken care of by the :ref:`data_loader.py 
file<sec-data-loader-file>`.
This file defines a data structure that is used consistently within :mod:`piva`.
Furthermore, it provides the :class:`~Dataloader` class as a base structure 
for data loaders from different sources (beamlines, electron analyzers, labs).
A dedicated dataloader exists for some ARPES beamlines.
If your beamline is not yet supported, you will not be able to read your data 
into :mod:`piva` unless a new data loader is written and added to that file.
Confer :ref:`this section below<sec-writing-dataloader>` about how to write a 
custom :class:`~Dataloader`.

:ref:`working_procedures.py file<sec-working_procedures-file>`

.. _sec-data-structure:

The data structure
------------------

The file format that is used by :mod:`piva` internally is a simple 
data structure implemented as an :class:`argparse.Namespace` 
object. [#namespace]_
The following table gives an overview of the data structure definition:

.. table:: `piva`'s internal data structure
   :widths: auto

   ============  ==========  ==================================================
   name          type        description
   ============  ==========  ==================================================
   data          3d array    Acquired data of dimension *(l, m, n)*. For 2d 
                             scans this still has to be a 3d array where one 
                             dimension is empty (*n* = 0).
   xscale        1d array    Units along dimension 0. Has length *l*.
   yscale        1d array    Units along dimension 1. Has length *m*.
   zscale        1d array    Units along dimension 2. Has length *n*.
   ekin          float       Kinetic energy in eV.
   kxscale       1d array    Units along the first angular dimension 
                             converted to *k-space*.
   kyscale       1d array    Units along the second angular dimension
                             converted to *k-space*.
   x             float       Manipulator x position (usually in micrometer).
   y             float       Manipulator y position (usually in micrometer).
   z             float       Manipulator z position (usually in micrometer).
   theta         float       Manipulator theta angle in degree.
   phi           float       Manipulator phi angle in degree.
   tilt          float       Manipulator tilt angle in degree.
   pressure      float       Vacuum chamber pressure in Torr.
   hv            float       Incoming photon energy in eV.
   wf            float       Work function in eV.
   Ef            float       Fermi energy in eV.
   polarization  string      Polarization of the incoming photon beam.
   PE            ?           ?
   exit_slit     string      Status of beamline exit slit.
   FE            ?           ?
   scan_type     string      Type of the *outer loop* scan, e.g. Fermi 
                             surface map or photon energy scan, etc.
   scan_dim      string      Name of the *outer loop* dimension that was 
                             scanned.
   acq_mode      string      Name of the used acquisition mode (dithered, 
                             fixed) 
   lens_mode     string      Lens mode.
   anal_slit     string      Orientation of the analyzer slit with respect to 
                             the photoemission plane defined by the incoming 
                             and outgoing beam trajectories.
   n_sweeps      int         Number of sweeps per cut.
   DT            ?           Dithering parameter?
   ============  ==========  ==================================================

Not all attributes are present or even required in order for *piva* to 
display data.
In fact, as long as the *data* attribute is present, *piva* should be able to 
display the data.
A lot of the other information, however, is needed for processing routines, 
such as angle-to- *k* -space conversion.

Attributes that are not present in a given dataset are represented by the 
value `None`.


.. _sec-writing-dataloader:

Writing a `Dataloader`
----------------------

A `piva` dataloader is nothing else than a subclass of :class:`~Dataloader` 
that takes care of three things:

1. Read in an ARPES data file
2. Extract all relevant data and metadata from said file
3. Put the so extracted data and metadata it into a 
   :class:`~argparse.Namespace` adhering to the data structure as 
   :ref:`defined above<sec-data-structure>`.

The :class:`~Dataloader` class offers an interface for this functionality.
In order to write a custom dataloader, let's call it `DataloaderExample` for 
the sake of the following example, you would start by creating a subclass of 
:class:`~Dataloader`::

    class DataloaderExample(Dataloader) :
        name = 'Example'
        date = '1947-08-15'

Be sure to give it an appropriate name and update the date whenever you make 
changes.
This is not functionally necessary, but will greatly help yourself and others.

All the work of steps 1, 2 and 3 above has to be carried out by this new 
dataloaders `load_data` method::

    class DataloaderExample(Dataloader) :
        name = 'Example'
        date = '1947-08-15'

        def load_data(self, filename) :
            # <Your code here>
            # 1) Read in data from *filename*
            # 2) Extract necessary (meta)data
            # 3) Put it into an argparse.Namespace and return it
            return D

In order for your new dataloader to be usable from `piva`, it needs to be 
registered in the `data_loader.py` file by being added to the list `all_dls`::

    # in file data_loader.py
    # somewhere deep down
    all_dls = [
        # ...
        # Lots of dataloaders
        # ...
        DataloaderExample
        ]

You are highly encouraged to share your self-written data loaders with the 
world by adding it to :mod:`piva`'s source code.
This is ideally done `directly through github 
<https://github.com/pudeIko/piva/pulls>`_ or alternatively by contacting the 
development team:

.. include:: contact.rst

.. [#namespace]

    The only thing you need to know about this, is that it accepts arbitrary 
    python objects to store as its attributes, making it function as a simple 
    *key* - *value* container, like a python dictionary.
    The advantage with respect to a dictionary is that its attributes can be 
    accessed simply through *dot notation*, i.e. `container.attribute` 
    instead of `container['attribute']`.


