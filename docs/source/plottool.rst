.. _sec-plottool:

Plot Tool
=========

**PlotTool** is a utility allowing for plotting together multiple curves.
Every 1D set of data displayed currently on **DataViewers** or **Fitters** can
be imported and further processed to compare differences between *e.g* results
obtained under different conditions.

To open **PlotTool** window one can simply use a ``CTRL + P`` shortcut or
access it from menu bar of the :ref:`DataBrowser <sec-db>`.

Its overall layout is shown below.

.. figure:: ../img/plottool-main.png
   :alt: Image not found.

=====   =======================================================================
**a**   - **Curves** dropdown menu contains list of all currently loaded
          curves. By selecting corresponding curve one can change its
          properties in other tabs.
        - **save** button allows to save current plot as an image, or entire
          session as a pickle file, which can be loaded later on.
        - **load** button allows to load previously loaded **PlotTool**
          session.
        - **close** button closes the window.
**b**   **Utilities panel**, described in more detail :ref:`below <sec-pt-up>`.
**c**   **Plot panel** displaying loaded and edited curves.
=====   =======================================================================



.. _sec-pt-up:

Plot tool's Utilities Panel
---------------------------

**Utilities panel** allows for managing datasets, editing their appearance
and many other functionalities helpful in creating easy to read plots.

Functionalities included in different tabs are described below.


Add/Remove data tab
^^^^^^^^^^^^^^^^^^^

Allows for general management over imported data.

.. figure:: ../img/plottool-up-add_remove.png
   :alt: Image not found.

===================     =======================================================
opened datasets         Select dataset to import:

                        - **source** - dropdown menu containing list of all
                          currently opened **DataViewers** and **Fitters**.
                          Option *custom* allows to add *user defined points*.
                        - **plot** - dropdown menu containing a list of all
                          curves possible to import. Available options are
                          based on selection of the **source**, *e.g* when **3D
                          DataViewer** is selected list includes: *main
                          edc, single point edc, vertical (analyzer)* and
                          *horizontal (scanned)*.
user defined points     If **source** selection is set to *custom*, one can
                        simply add curve to plot by hand. *x* and *y* points
                        must be given as a list of numbers separated by
                        ``space``. **Name** specifies a label by which curve
                        will be saved and stored in **Curves** list.
action buttons          - **add** data based on current **source** and
                          **plot** selection.
                        - **update** list of **source** dropdown menu. When
                          initializing **PlotTool** window, list contains all
                          currently opened **DataViewers** and **Fitters**; if
                          a new one is opened, **source** list needs to be
                          updated.
                        - **remove** curve based on the current selection of
                          the **Curves** list on the left side (outside of the
                          **Utilities panel**).
===================     =======================================================


Edit curves tab
^^^^^^^^^^^^^^^

Helps to adjust and personalize appearance of the imported curves. All changes
made in this tab are performed on a curve currently selected in **Curves**
list.

.. figure:: ../img/plottool-up-edit_curves.png
   :alt: Image not found.

===============     ===========================================================
graphic options     Change visual parameters (color, line width and its style)
                    of the selected curve.
offsets             Apply offsets to the dataset along *x* and *y* directions.
scaling options     Scale by a constant or normalize data.
===============     ===========================================================


Edit plot tab
^^^^^^^^^^^^^

Helps to adjust and personalize overall appearance of the plot.

.. figure:: ../img/plottool-up-edit_plot.png
   :alt: Image not found.

===============     ===========================================================
graphic options     Change visual parameters (background color, axes color and
                    ticks font size) of the plot.
axes' labels        Set labels of the axes and their font size.
===============     ===========================================================

.. note::

    While specifying labels, separating description and units with a semicolon
    automatically formats the label from "`descr`;`units`" to "`descr`
    (`units`)". Moreover, :mod:`pyqtgraph` scales an axis with an appropriate
    prefix (kilo, mili, micro, *etc.*).


Markers tab
^^^^^^^^^^^

**Drop** buttons append movable markers on the currently selected curve.

Sliding markers across the curve helps to read exact position (*x* and *y*) of
fine features on the plot and display differences between their positions
(*dx* and *dy*) along the corresponding direction.

.. figure:: ../img/plottool-up-markers.png
   :alt: Image not found.


Annotate tab
^^^^^^^^^^^^

Allows to append text annotations on the plot to include some useful
information.

.. figure:: ../img/plottool-up-annotate.png
   :alt: Image not found.

=====================   =======================================================
annotation parameters   - **name** - label of an annotation in **added** list,
                        - **text** - annotation text.
added                   Dropdown menu containing list of added annotations.
                        Editing is performed on currently selected one.
graphic options         Change visual parameters (color and font size) of the
                        text.
position                Set position of the annotation in data coordinates.
action buttons          - **add** new or **update** selected annotation,
                        - **delete** selected annotation.
=====================   =======================================================


.. note::

    Many more options are embedded in :class:`pyqtgraph.PlotWidget` object and
    can be accessed by clicking right mouse button on the panel.

