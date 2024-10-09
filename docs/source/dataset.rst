.. _sec-dataset:


Data Format in :mod:`piva`
==========================

This section discusses what it means for a data file to be *recognizable* and
*readable* by :mod:`piva`.

Problem of Multiple File Formats
--------------------------------

Within the ARPES community, it is common for each beamline and lab to use
their own file formats and conventions. Consequently, handling these different
files requires a dedicated script that converts them into a common format.

In :mod:`piva`, this task is managed by the
:ref:`data_loaders module <sec-data-loaders-file>`, which implements specific
Dataloaders for files from various sources and returns a standardized
:class:`~data_loaders.Dataset` object.

The :class:`~data_loaders.Dataset` defines a data structure consistently used
within :mod:`piva` and is understandable by all other :mod:`piva` modules.

.. seealso::
    To open file formats **not** originally included in :mod:`piva`, you need
    to implement a loader that returns a :class:`~data_loaders.Dataset`. A
    detailed guide on how to do this can be found :ref:`here <sec-custom-dl>`.


.. _sec-dataset-structure:


:class:`~data_loaders.Dataset` structure
----------------------------------------

The file format used internally by :mod:`piva` is a simple structured object
inheriting from the :class:`argparse.Namespace` object. [#namespace]_

The following table provides an overview of the data structure definition:


    ===============  ===================  =====================================
    **attribute**    **type**             **description**
    ===============  ===================  =====================================
    **data** *       :class:`np.ndarray`  Acquired data of dimension
                                          *(l, m, n)*. Oriented as: `l`-scanned
                                          axis, `m`- analyzer axis, `n`-energy
                                          axis. When scan is a single cut
                                          (data are 2D), first dimension is
                                          equal to
                                          ``data[0, :, :] = np.array([0])``.
    **xscale** *     :class:`np.ndarray`  Axis along the scanned direction,
                                          has length *l*. Units depend on the
                                          scan type. When scan type is a single
                                          cut (2D), it is set to
                                          ``np.array([1])``.
    **yscale** *     :class:`np.ndarray`  Axis along the analyzer slit, has
                                          length *m*. Most likely in [deg].
    **zscale** *     :class:`np.ndarray`  Axis along the energy direction, has
                                          length *m*. Most likely in [eV].
    ekin             :class:`np.ndarray`  Energy axis in kinetic energy
                     | :class:`None`      scale (if default scale is in
                                          binding energy).
    kxscale          :class:`np.ndarray`  Momentum axis (saved after
                     | :class:`None`      conversion) along the scanned
                                          direction.
    kyscale          :class:`np.ndarray`  Momentum axis (saved after
                     | :class:`None`      conversion) along the analyzer
                                          direction.
    x                :class:`float` |     `x` position of the manipulator.
                     :class:`None`
    y                :class:`float` |     `y` position of the manipulator.
                     :class:`None`
    z                :class:`float` |     `z` position of the manipulator.
                     :class:`None`
    theta            :class:`float` |     `theta` angle of the manipulator;
                     :class:`None`        often referred as `polar`.

    phi              :class:`float` |     `phi` angle of the manipulator;
                     :class:`None`        often referred as `azimuth`.

    tilt             :class:`float` |     `tilt` angle of the manipulator.
                     :class:`None`
    temp             :class:`float` |     Temperature during the experiment.
                     :class:`None`
    pressure         :class:`float` |     Pressure during the experiment.
                     :class:`None`
    hv               :class:`float` |     Photon energy used during the
                     :class:`None`        experiment.
    wf               :class:`float` |     Work function of the analyzer.
                     :class:`None`
    Ef               :class:`float` |     Correction for the Fermi level.
                     :class:`None`
    polarization     :class:`str` |       Photon polarization.
                     :class:`None`
    PE               :class:`int` |       Pass energy of the analyzer.
                     :class:`None`
    exit_slit        :class:`float` |     Exit (vertical) slit of the
                     :class:`None`        beamline; responsible for energy
                                          resolution.
    FE               :class:`float` |     Front end of the beamline.
                     :class:`None`
    scan_type        :class:`str` |       Type of the measurement (e.g.
                     :class:`None`        `cut`, `tilt scan`, `hv scan`).

    scan_dim         :class:`list` |      If scan other than `cut`, scanned
                     :class:`None`        dimensions as list: [`start`,
                                          `stop`, `step`].
    acq_mode         :class:`str` |       Data acquisition mode.
                     :class:`None`
    lens_mode        :class:`str` |       Lens mode of the analyzer.
                     :class:`None`
    ana_slit         :class:`str` |       Slit opening of the analyzer.
                     :class:`None`
    defl_angle       :class:`float` |     Applied deflection angle.
                     :class:`None`
    n_sweeps         :class:`int` |       Number of sweeps.
                     :class:`None`
    DT               :class:`int` |       Analyzer dwell time during data
                     :class:`None`        acquisition, most likely in
                                          [miliseconds].
    data_provenance  :class:`dict`        Dataset logbook; contains
                                          information about original file
                                          and keeps track of functions
                                          called on the data.
    ===============  ===================  =====================================

Not all attributes are present or required for :mod:`piva` to display data.
Attributes that are mandatory for the functioning of the **DataViewers** are
marked with an asterisk (*). However, much of the other information is needed
for processing routines, such as angle-to-`k`-space conversion.


.. [#namespace]

    The only thing one needs to know about this, is that it accepts arbitrary
    python objects to store as its attributes, making it function as a simple 
    *key* - *value* container, like a python dictionary.
    The advantage with respect to a dictionary is that its attributes can be 
    accessed simply through *dot notation*, i.e. `container.attribute` 
    instead of `container['attribute']`.




