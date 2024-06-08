.. _sec-dataset:

Data format in :mod:`piva`
==========================


This section discusses what it means for a datafile to be *recognizable* and
thus *readable* by :mod:`piva`.


Problem of multiple file formats
--------------------------------

Within ARPES community, the tendency seems to be that each beamline and each
lab uses their own file formats and conventions.
Therefore, in order to handle these different files there is really
no other way than to have a dedicated script which converts these into a
common form.

Within :mod:`piva`, this is taken care of by the :ref:`Dataloader module
<sec-data-loaders-file>`, which implements specific Dataloaders for files from
different sources and returns a standardized :class:`~data_loaders.Dataset`
object.

:class:`~data_loaders.Dataset` defines a data structure used consistently
within :mod:`piva` and understandable by all other :mod:`piva` modules.

.. seealso::
    In order to open a file formats **not** included originally in :mod:`piva`,
    one needs to implement a loader returning :class:`~data_loaders.Dataset`.
    Detailed guide on how to do it can be found :ref:`here <sec-custom-dl>`.


.. _sec-dataset-structure:


:class:`~data_loaders.Dataset` structure
----------------------------------------

The file format that is used by :mod:`piva` internally is a simple structured
object inheriting from :class:`argparse.Namespace` object. [#namespace]_


The following table gives an overview of the data structure definition:


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
    anal_slit        :class:`str` |       Slit opening of the analyzer.
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

Not all attributes are present or even required for *piva* to display data.
Attributes that are mandatory for functioning of the **DataViewers** are marked
in above table with asterix (*).
A lot of the other information, however, is needed for processing routines, 
such as angle-to-`k`-space conversion.



.. [#namespace]

    The only thing you need to know about this, is that it accepts arbitrary 
    python objects to store as its attributes, making it function as a simple 
    *key* - *value* container, like a python dictionary.
    The advantage with respect to a dictionary is that its attributes can be 
    accessed simply through *dot notation*, i.e. `container.attribute` 
    instead of `container['attribute']`.


