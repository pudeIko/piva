"""
Data handler and main window creator for 3D data inspection
"""
import time
from PyQt5.QtWidgets import QMainWindow
import matplotlib.pyplot as plt
import numpy as np
import warnings
from imageplot import *
from cmaps import cmaps
import arpys_wp as wp

app_style = """
QMainWindow{background-color: rgb(64,64,64);}
"""
DEFAULT_CMAP = 'viridis'
NDIM = 3
erg_ax = 2
slit_ax = 1
scan_ax = 0


class DataHandler:
    """ Object that keeps track of a set of 3D data and allows
    manipulations on it. In a Model-View-Controller framework this could be
    seen as the Model, while :class:`MainWindow <data_slicer.pit.MainWindow>`
    would be the View part.
    """

    def __init__(self, main_window):
        self.main_window = main_window
        self.binning = False
        # self.erg_ax = erg_ax

        # Initialize instance variables
        # np.array that contains the 3D data
        self.data = None
        self.axes = array([[0, 1], [0, 1], [0, 1]])
        # Indices of *data* that are displayed in the main plot
        self.displayed_axes = (0, 1)
        # Index along the z axis at which to produce a slice
        self.z = TracedVariable(0, name='z')
        # # Number of slices to integrate along z
        # integrate_z = TracedVariable(value=0, name='integrate_z')
        # How often we have rolled the axes from the original setup
        self._roll_state = 0

        # moved to get rid of warnings
        self.zmin = 0
        self.zmax = None
        self.integrated = None
        self.cut_y_data = None
        self.cut_x_data = None

    def get_data(self):
        """ Convenience `getter` method. Allows writing ``self.get_data()``
        instead of ``self.data.get_value()``.
        """
        return self.data.get_value()

    def set_data(self, data):
        """ Convenience `setter` method. Allows writing ``self.set_data(d)``
        instead of ``self.data.set_value(d)``.
        """
        self.data.set_value(data)

    def prepare_data(self, data, axes=3 * [None]):
        """ Load the specified data and prepare the corresponding z range.
        Then display the newly loaded data.

        **Parameters**

        ====  ==================================================================
        data  3d array; the data to display
        axes  len(3) list or array of 1d-arrays or None; the units along the
              x, y and z axes respectively. If any of those is *None*, pixels
              are used.
        ====  ==================================================================
        """

        self.data = TracedVariable(data, name='data')
        self.axes = array(axes, dtype="object")

        # Retain a copy of the original data and axes so that we can reset later
        # NOTE: this effectively doubles the used memory!
        # self.original_data = copy(self.data.get_value())
        # self.original_axes = copy(self.axes)

        self.prepare_axes()
        self.on_z_dim_change()

        # Connect signal handling so changes in data are immediately reflected
        self.z.sig_value_changed.connect(lambda: self.main_window.update_main_plot(emit=False))
        # self.data.sig_value_changed.connect(self.on_data_change)

        self.main_window.update_main_plot()
        self.main_window.set_axes()

    def get_main_data(self):
        """ Return the 2d array that is currently displayed in the main plot.
        """
        return self.main_window.main_plot.image_data

    def get_cut_data(self):
        """ Return the 2d array that is currently displayed in the cut plot.
        """
        return self.main_window.cut_plot.image_data

    def get_hprofile(self):
        """ Return an array containing the y values displayed in the
        horizontal profile plot (mw.y_plot).

        .. seealso::
            :func:`data_slicer.imageplot.CursorPlot.get_data`
        """
        return self.main_window.y_plot.get_data()[1]

    def get_vprofile(self):
        """ Return an array containing the x values displayed in the
        vertical profile plot (mw.x_plot).

        .. seealso::
            :func:`data_slicer.imageplot.CursorPlot.get_data`
        """
        return self.main_window.x_plot.get_data()[0]

    def get_iprofile(self):
        """ Return an array containing the y values displayed in the
        integrated intensity profile plot (mw.integrated_plot).

        .. seealso::
            :func:`data_slicer.imageplot.CursorPlot.get_data`
        """
        return self.main_window.integrated_plot.get_data()[1]

    def update_z_range(self):
        """ When new data is loaded or the axes are rolled, the limits and
        allowed values along the z dimension change.
        """
        # Determine the new ranges for z
        self.zmin = 0
        self.zmax = self.get_data().shape[2]

        self.z.set_allowed_values(range(self.zmin, self.zmax))

    def prepare_axes(self):
        """ Create a list containing the three original x-, y- and z-axes
        and replace *None* with the amount of pixels along the given axis.
        """
        shapes = self.data.get_value().shape
        # Avoid undefined axes scales and replace them with len(1) sequences
        for i, axis in enumerate(self.axes):
            if axis is None:
                self.axes[i] = arange(shapes[i])

    def on_data_change(self):
        """ Update self.main_window.image_data and replot. """
        self.update_image_data()
        self.main_window.redraw_plots()
        # Also need to recalculate the intensity plot
        self.on_z_dim_change()

    def on_z_dim_change(self):
        """ Called when either completely new data is loaded or the dimension
        from which we look at the data changed (e.g. through :func:`roll_axes
        <data_slicer.pit.PITDataHandler.roll_axes>`).
        Update the z range and the integrated intensity plot.
        """
        self.update_z_range()

        # Get a shorthand for the integrated intensity plot
        ip = self.main_window.plot_z
        # Remove the old integrated intensity curve
        try:
            old = ip.listDataItems()[0]
            ip.removeItem(old)
        except IndexError:
            pass

        # Calculate the integrated intensity and plot it
        self.calculate_integrated_intensity()
        ip.plot(wp.normalize(self.integrated))

        # Also display the actual data values in the top axis
        zscale = self.axes[erg_ax]
        zmin = zscale[0]
        zmax = zscale[-1]
        ip.set_secondary_axis(zmin, zmax)

    def calculate_integrated_intensity(self):
        self.integrated = self.get_data().sum(0).sum(0)

    def update_image_data(self):
        """ Get the right (possibly integrated) slice out of *self.data*,
        apply postprocessings and store it in *self.image_data*.
        Skip this if the z value happens to be out of range, which can happen
        if the image data changes and the z scale hasn't been updated yet.
        """
        z = self.z.get_value()
        integrate_z = self.main_window.plot_z.width
        data = self.get_data()
        try:
            self.main_window.image_data = self.make_slice(data, dim=2, index=z, integrate=integrate_z)
        except IndexError:
            pass
        self.main_window.util_panel.energy_main.setValue(z)
        self.main_window.util_panel.energy_main_value.setText('({:.4f})'.format(self.axes[erg_ax][z]))

    @staticmethod
    def make_slice(data, dim, index, integrate=0, silent=True):
        """
        Take a slice out of an N dimensional dataset *data* at *index* along
        dimension *dim*. Optionally integrate by +- *integrate* channels around
        *index*.
        If *data* has shape::

            (n0, n1, ..., n(dim-1), n(dim), n(dim+1), ..., n(N-1))

        the result will be of dimension N-1 and have shape::

            (n0, n1, ..., n(dim-1), n(dim+1), ..., n(N-1))

        or in other words::

            shape(result) = shape(data)[:dim] + shape(data)[dim+1:]

        .

        **Parameters**

        =========  =================================================================
        data       array-like; N dimensional dataset.
        dim        int, 0 <= d < N; dimension along which to slice.
        index      int, 0 <= index < data.size[d]; The index at which to create
                   the slice.
        integrate  int, ``0 <= integrate < |index|``; the number of slices above
                   and below slice *index* over which to integrate. A warning is
                   issued if the integration range would exceed the data (can be
                   turned off with *silent*).
        silent     bool; toggle warning messages.
        =========  =================================================================

        **Returns**

        ===  =======================================================================
        res  np.array; slice at *index* alond *dim* with dimensions shape[:d] +
             shape[d+1:].
        ===  =======================================================================
        """
        # Find the dimensionality and the number of slices along the specified dimension.
        shape = data.shape
        ndim = len(shape)
        try:
            n_slices = shape[dim]
        except IndexError:
            message = '*dim* ({}) needs to be smaller than the dimension of *data* ({})'.format(dim, ndim)
            raise IndexError(message)

        # Set the integration indices and adjust them if they go out of scope
        start = index - integrate
        stop = index + integrate + 1
        if start < 0:
            if not silent:
                warnings.warn('i - integrate ({}) < 0, setting start=0'.format(start))
            start = 0
        if stop > n_slices:
            if not silent:
                warning = 'i + integrate ({}) > n_slices ({}), setting stop=n_slices'.format(stop, n_slices)
                warnings.warn(warning)
            stop = n_slices

        # Roll the original data such that the specified dimension comes first
        i_original = arange(ndim)
        i_rolled = np.roll(i_original, dim)
        data = np.moveaxis(data, i_original, i_rolled)
        # Take the slice
        sliced = data[start:stop].sum(0)
        # Bring back to more intuitive form. For that we have to remove the now
        # lost dimension from the index arrays and shift all indices.
        i_original = np.concatenate((i_original[:dim], i_original[dim + 1:]))
        i_original[i_original > dim] -= 1
        i_rolled = np.roll(i_original, dim)
        return np.moveaxis(sliced, i_rolled, i_original)

    def lineplot(self, plot='main', dim=0, ax=None, n=10, offset=0.2, lw=0.5,
                 color='k', label_fmt='{:.2f}', n_ticks=5, **getlines_kwargs):
        """
        Create a matplotlib figure with *n* lines extracted out of one of the
        visible plots. The lines are normalized to their global maximum and
        shifted from each other by *offset*.
        See :func:`get_lines <data_slicer.utilities.get_lines>` for more
        options on the extraction of the lines.
        This wraps the :class:`ImagePlot <data_slicer.imageplot.ImagePlot>`'s
        lineplot method.

        **Parameters**

        ===============  =======================================================
        plot             str; either "main" or "cut", specifies from which
                         plot to extract the lines.
        dim              int; either 0 or 1, specifies in which direction to
                         take the lines.
        ax               matplotlib.axes.Axes; the axes in which to plot. If
                         *None*, create a new figure with a fresh axes.
        n                int; number of lines to extract.
        offset           float; spacing between neighboring lines.
        lw               float; linewidth of the plotted lines.
        color            any color argument understood by matplotlib; color
                         of the plotted lines.
        label_fmt        str; a format string for the ticklabels.
        n_ticks          int; number of ticks to print.
        getlines_kwargs  other kwargs are passed to :func:`get_lines
                         <data_slicer.utilities.get_lines>`
        ===============  =======================================================

        **Returns**

        ===========  ===========================================================
        lines2ds     list of Line2D objects; the drawn lines.
        xticks       list of float; locations of the 0 intensity value of
                     each line.
        xtickvalues  list of float; if *momenta* were supplied, corresponding
                     xtick values in units of *momenta*. Otherwise this is
                     just a copy of *xticks*.
        xticklabels  list of str; *xtickvalues* formatted according to
                     *label_fmt*.
        ===========  ===========================================================

        .. seealso::
            :func:`get_lines <data_slicer.utilities.get_lines>`
        """
        # Get the specified data
        if plot == 'main':
            imageplot = self.main_window.main_plot
        elif plot == 'cut':
            imageplot = self.main_window.cut_plot
        else:
            raise ValueError('*plot* should be one of ("main", "cut").')

        # Create a mpl axis object if none was given
        if ax is None:
            fig, ax = plt.subplots(1)

        return imageplot.lineplot(ax=ax, dim=dim, n=n, offset=offset, lw=lw,
                                  color=color, label_fmt=label_fmt,
                                  n_ticks=n_ticks, **getlines_kwargs)


class MainWindow3D(QMainWindow):

    def __init__(self, data_browser, data_set=None, title=None, index=0):
        super(MainWindow3D, self).__init__()
        self.title = title
        self.central_widget = QtGui.QWidget()
        self.layout = QtGui.QGridLayout()

        # moved to get rid of warnings
        self.cmap = None
        self.image_kwargs = None
        self.cmap_name = None
        self.lut = None
        self.db = data_browser
        self.index = index

        # Initialize instance variables
        # Plot transparency alpha
        self.alpha = 1
        # Plot powerlaw normalization exponent gamma
        self.gamma = 1
        # Relative colormap maximum
        self.vmax = 1

        # Need to store original transformation information for `rotate()`
        self._transform_factors = []

        # Create the 3D (main) and cut ImagePlots
        self.main_plot = ImagePlot(name='main_plot')
        # Create cut plot along x
        self.cut_x = ImagePlot(name='cut_x')
        # Create cut of cut_x
        self.plot_x = CursorPlot(name='plot_x')
        # Create cut plot along y
        self.cut_y = ImagePlot(name='cut_y', orientation='vertical')
        # Create cut of cut_y
        self.plot_y = CursorPlot(name='plot_y', orientation='horizontal')
        # Create the integrated intensity plot
        self.plot_z = CursorPlot(name='plot_z', z_plot=True)
        # Create utilities panel
        self.util_panel = UtilitiesPanel(self, name='utilities_panel')

        self.setStyleSheet(app_style)
        self.set_cmap()

        self.setGeometry(100, 100, 800, 900)
        self.setWindowTitle(self.title)
        self.sp_EDC = None

        time_dl_b = time.time()
        if data_set is None:
            print('No dataset to open.')
            self.close_mw()
        else:
            D = data_set

        self.data_handler = DataHandler(self)
        self.initUI()

        self.data_handler.prepare_data(D.data, [D.xscale, D.yscale, D.zscale])
        self.set_sliders_labels(D)

        self.util_panel.energy_main.setRange(0, len(self.data_handler.axes[erg_ax]))
        self.util_panel.energy_hor.setRange(0, len(self.data_handler.axes[erg_ax]))
        self.util_panel.energy_vert.setRange(0, len(self.data_handler.axes[erg_ax]))
        self.util_panel.momentum_hor.setRange(0, len(self.data_handler.axes[slit_ax]))
        self.util_panel.momentum_vert.setRange(0, len(self.data_handler.axes[scan_ax]))

        # create a single point EDC at crossing point of momentum sliders
        self.sp_EDC = self.plot_z.plot()
        self.set_sp_EDC_data()

        self.put_sliders_in_initial_positions()

    def initUI(self):
        self.setWindowTitle(self.title)
        # Create a "central widget" and its layout
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        # Create cut plot along x
        self.cut_x.register_momentum_slider(self.main_plot.pos[0])
        self.cut_x.crosshair.vpos.sig_value_changed.connect(self.update_cut_y)
        self.cut_x.crosshair.hpos.sig_value_changed.connect(self.update_plot_x)

        # Create cut of cut_x
        self.plot_x.register_traced_variable(self.main_plot.pos[0])

        # Create cut plot along y
        self.cut_y.register_momentum_slider(self.main_plot.pos[1])
        self.cut_y.crosshair.vpos.sig_value_changed.connect(self.update_cut_x)
        self.cut_y.crosshair.hpos.sig_value_changed.connect(self.update_plot_y)

        # Create cut of cut_y
        self.plot_y.register_traced_variable(self.main_plot.pos[1])

        # Create the integrated intensity plot
        self.plot_z.register_traced_variable(self.data_handler.z)
        self.plot_z.change_width_enabled = True

        # Connect signals to utilities panel
        self.util_panel.close_button.clicked.connect(self.close_mw)
        self.util_panel.cmaps.currentIndexChanged.connect(self.set_cmap)
        self.util_panel.invert_colors.stateChanged.connect(self.set_cmap)
        self.util_panel.gamma.valueChanged.connect(self.set_gamma)
        self.util_panel.colorscale.valueChanged.connect(self.set_colorscale)

        # binning signals
        self.util_panel.bin_x.stateChanged.connect(self.set_x_binning_lines)
        self.util_panel.bin_x_nbins.valueChanged.connect(self.set_x_binning_lines)
        self.util_panel.bin_y.stateChanged.connect(self.set_y_binning_lines)
        self.util_panel.bin_y_nbins.valueChanged.connect(self.set_y_binning_lines)
        self.util_panel.bin_z.stateChanged.connect(self.update_z_binning_lines)
        self.util_panel.bin_z_nbins.valueChanged.connect(self.update_z_binning_lines)
        self.util_panel.bin_zx.stateChanged.connect(self.set_zx_binning_line)
        self.util_panel.bin_zx_nbins.valueChanged.connect(self.set_zx_binning_line)
        self.util_panel.bin_zy.stateChanged.connect(self.set_zy_binning_line)
        self.util_panel.bin_zy_nbins.valueChanged.connect(self.set_zy_binning_line)
        self.util_panel.energy_main.valueChanged.connect(self.set_main_energy_slider)
        self.util_panel.energy_hor.valueChanged.connect(self.set_hor_energy_slider)
        self.util_panel.energy_vert.valueChanged.connect(self.set_vert_energy_slider)
        self.util_panel.momentum_hor.valueChanged.connect(self.set_hor_momentum_slider)
        self.util_panel.momentum_vert.valueChanged.connect(self.set_vert_momentum_slider)
        self.util_panel.bin_x_nbins.setValue(2)
        self.util_panel.bin_y_nbins.setValue(10)
        self.util_panel.bin_z_nbins.setValue(10)
        self.util_panel.bin_zx_nbins.setValue(5)
        self.util_panel.bin_zy_nbins.setValue(5)

        # Align all the gui elements
        self._align()
        self.show()

    def _align(self):
        """ Align all the GUI elements in the QLayout::

              0   1   2   3
            +---+---+---+---+
            |utilities panel| 0
            +---+---+---+---+
            | mdc x |       | 1
            +-------+  edc  |
            | cut x |       | 2
            +-------+-------+
            |       | c | m | 3
            | main  | y | y | 4
            +---+---+---+---+

            (Units of subdivision [sd])
        """
        # subdivision
        sd = 1
        # Get a short handle
        l = self.layout
        # addWIdget(row, column, rowSpan, columnSpan)
        # utilities bar
        # l.addWidget(self.b0, 0, 0, 1 * sd, 5 * sd)
        l.addWidget(self.util_panel, 0, 0, 1 * sd, 5 * sd)
        # Main plot
        l.addWidget(self.main_plot, 3 * sd, 0, 2 * sd, 2 * sd)
        # X cut and mdc
        l.addWidget(self.cut_x, 2 * sd, 0, 1 * sd, 2 * sd)
        l.addWidget(self.plot_x, 1 * sd, 0, 1 * sd, 2 * sd)
        # Y cut and mdc
        l.addWidget(self.cut_y, 3 * sd, 2, 2 * sd, 1 * sd)
        l.addWidget(self.plot_y, 3 * sd, 3 * sd, 2 * sd, 1 * sd)
        # EDC (integrated)
        l.addWidget(self.plot_z, 1 * sd, 2 * sd, 2 * sd, 2 * sd)

        nrows = 5 * sd
        ncols = 4 * sd
        # Need to manually set all row- and columnspans as well as min-sizes
        for i in range(nrows):
            l.setRowMinimumHeight(i, 50)
            l.setRowStretch(i, 1)
        for i in range(ncols):
            l.setColumnMinimumWidth(i, 50)
            l.setColumnStretch(i, 1)

    # def print_to_console(self, message):
    #     """ Print a *message* to the embedded ipython console. """
    #     self.console.kernel.stdout.write(str(message) + '\n')

    def update_main_plot(self, **image_kwargs):
        """ Change *self.main_plot*`s currently displayed
        :class:`image_item <data_slicer.imageplot.ImagePlot.image_item>` to
        the slice of *self.data_handler.data* corresponding to the current
        value of *self.z*.
        """
        slider_pos = self.get_sliders_positions()
        self.data_handler.update_image_data()

        if image_kwargs != {}:
            self.image_kwargs = image_kwargs

        # Add image to main_plot
        self.set_image(self.image_data, **image_kwargs)
        self.set_sliders_postions(slider_pos)
        self.update_xy_binning_lines()

    def set_axes(self):
        """ Set the x- and y-scales of the plots. The :class:`ImagePlot
        <data_slicer.imageplot.ImagePlot>` object takes care of keeping the
        scales as they are, once they are set.
        """
        xaxis = self.data_handler.axes[scan_ax]
        yaxis = self.data_handler.axes[slit_ax]
        zaxis = self.data_handler.axes[erg_ax]
        # self.main_plot.set_xscale(xaxis)
        self.main_plot.set_xscale(range(0, len(xaxis)))
        self.main_plot.set_ticks(xaxis[0], xaxis[-1], self.main_plot.main_xaxis)
        self.cut_x.set_ticks(xaxis[0], xaxis[-1], self.cut_x.main_xaxis)
        self.cut_y.set_ticks(yaxis[0], yaxis[-1], self.cut_y.main_xaxis)
        self.plot_x.set_ticks(xaxis[0], xaxis[-1], self.plot_x.main_xaxis)
        self.plot_x.set_secondary_axis(0, len(xaxis))
        # self.main_plot.set_yscale(yaxis)
        self.main_plot.set_yscale(range(0, len(yaxis)))
        self.main_plot.set_ticks(yaxis[0], yaxis[-1], self.main_plot.main_yaxis)
        self.cut_x.set_ticks(zaxis[0], zaxis[-1], self.cut_x.main_yaxis)
        self.cut_y.set_ticks(zaxis[0], zaxis[-1], self.cut_y.main_yaxis)
        self.plot_y.set_ticks(yaxis[0], yaxis[-1], self.plot_y.main_xaxis)
        self.plot_y.set_secondary_axis(0, len(yaxis))
        self.main_plot.fix_viewrange()

        # Kind of a hack to get the crosshair to the right position...
        # self.cut_x.sig_axes_changed.emit()
        # self.cut_y.sig_axes_changed.emit()

    def update_plot_x(self):
        # Get shorthands for plot
        xp = self.plot_x
        try:
            old = xp.listDataItems()[0]
            xp.removeItem(old)
        except IndexError:
            pass

        # Get the correct position indicator
        pos = self.cut_x.crosshair.hpos
        if pos.allowed_values is not None:
            i_x = int(min(pos.get_value(), pos.allowed_values.max() - 1))
        else:
            i_x = 0
        if self.util_panel.bin_zx.isChecked():
            bins = self.util_panel.bin_zx_nbins.value()
            start, stop = i_x - bins, i_x + bins
            y = wp.normalize(np.sum(self.data_handler.cut_x_data[:, start:stop], axis=1))
        else:
            y = self.data_handler.cut_x_data[:, i_x]
        x = arange(0, len(self.data_handler.axes[scan_ax]))
        xp.plot(x, y)
        self.util_panel.energy_hor.setValue(i_x)
        self.util_panel.energy_hor_value.setText('({:.4f})'.format(self.data_handler.axes[erg_ax][i_x]))
        self.update_zx_zy_binning_line()

    def update_plot_y(self):
        # Get shorthands for plot
        yp = self.plot_y
        try:
            old = yp.listDataItems()[0]
            yp.removeItem(old)
        except IndexError:
            pass

        # Get the correct position indicator
        pos = self.cut_y.crosshair.hpos
        if pos.allowed_values is not None:
            i_x = int(min(pos.get_value(), pos.allowed_values.max() - 1))
        else:
            i_x = 0
        if self.util_panel.bin_zy.isChecked():
            bins = self.util_panel.bin_zy_nbins.value()
            start, stop = i_x - bins, i_x + bins
            y = wp.normalize(np.sum(self.data_handler.cut_y_data[start:stop, :], axis=0))
        else:
            y = self.data_handler.cut_y_data[i_x, :]
        x = arange(0, len(self.data_handler.axes[slit_ax]))
        yp.plot(y, x)
        self.util_panel.energy_vert.setValue(i_x)
        self.util_panel.energy_vert_value.setText('({:.4f})'.format(self.data_handler.axes[erg_ax][i_x]))
        self.update_zx_zy_binning_line()

    def update_cut_x(self):
        """ Take a cut of *self.data_handler.data* along *self.cutline*. This
        is used to update only the cut plot without affecting the main plot.
        """
        data = self.data_handler.get_data()
        # axes = self.data_handler.displayed_axes
        # Transpose, if necessary
        pos = self.main_plot.crosshair.hpos
        if pos.allowed_values is not None:
            i_x = int(min(pos.get_value(), pos.allowed_values.max() - 1))
        else:
            i_x = 0
        try:
            if self.util_panel.bin_y.isChecked():
                bins = self.util_panel.bin_y_nbins.value()
                start, stop = i_x - bins, i_x + bins
                cut = np.sum(data[:, start:stop, :], axis=1)
            else:
                cut = data[:, i_x, :]
        except IndexError:
            return

        if self.util_panel.bin_x.isChecked():
            binning = self.util_panel.bin_x_nbins.value()
        else:
            binning = 0

        self.data_handler.cut_x_data = cut
        self.cut_x.xscale_rescaled = self.data_handler.axes[scan_ax]
        self.cut_x.yscale_rescaled = self.data_handler.axes[erg_ax]
        self.set_cut_x_image(image=cut, lut=self.lut)
        self.cut_x.crosshair.vpos.set_allowed_values(arange(0, len(self.data_handler.axes[scan_ax])), binning=binning)
        self.cut_x.crosshair.hpos.set_allowed_values(arange(0, len(self.data_handler.axes[erg_ax])), binning=binning)
        self.cut_x.set_bounds(0, len(self.data_handler.axes[scan_ax]), 0, len(self.data_handler.axes[erg_ax]))

        self.cut_x.fix_viewrange()

        # update values of momentum at utilities panel
        self.util_panel.momentum_hor.setValue(i_x)
        self.util_panel.momentum_hor_value.setText('({:.3f})'.format(self.data_handler.axes[slit_ax][i_x]))

        # update EDC at crossing point
        if self.sp_EDC is not None:
            self.set_sp_EDC_data()

        self.update_xy_binning_lines()

    def update_cut_y(self):
        """ Take a cut of *self.data_handler.data* along *self.cutline*. This
        is used to update only the cut plot without affecting the main plot.
        """
        data = self.data_handler.get_data()
        # axes = self.data_handler.displayed_axes
        # Transpose, if necessary
        pos = self.cut_x.crosshair.vpos
        if pos.allowed_values is not None:
            i_x = int(min(pos.get_value(), pos.allowed_values.max() - 1))
        else:
            i_x = 0
        try:
            if self.util_panel.bin_x.isChecked():
                bins = self.util_panel.bin_x_nbins.value()
                start, stop = i_x - bins, i_x + bins
                cut = np.sum(data[start:stop, :, :], axis=0).T
            else:
                cut = data[i_x, :, :].T
        except IndexError:
            return

        self.data_handler.cut_y_data = cut
        self.cut_y.xscale_rescaled = self.data_handler.axes[slit_ax]
        self.cut_y.yscale_rescaled = self.data_handler.axes[erg_ax]
        self.set_cut_y_image(image=cut, lut=self.lut)
        # bounds swapped to match transposed dimensions
        self.cut_y.set_bounds(0, len(self.data_handler.axes[erg_ax]), 0, len(self.data_handler.axes[slit_ax]))

        self.cut_y.fix_viewrange()

        # update values of momentum at utilities panel
        self.util_panel.momentum_vert.setValue(i_x)
        self.util_panel.momentum_vert_value.setText('({:.3f})'.format(self.data_handler.axes[scan_ax][i_x]))
        self.update_xy_binning_lines()

        # update EDC at crossing point
        if self.sp_EDC is not None:
            self.set_sp_EDC_data()

    def set_sp_EDC_data(self):
        xpos = self.main_plot.crosshair.vpos.get_value()
        ypos = self.main_plot.crosshair.hpos.get_value()
        data = self.data_handler.get_data()[xpos, ypos, :]
        self.sp_EDC.setData(wp.normalize(data), pen=self.plot_z.sp_EDC_pen)

    def redraw_plots(self, image=None):
        """ Redraw plotted data to reflect changes in data or its colors. """
        try:
            # Redraw main plot
            self.set_image(image, displayed_axes=self.data_handler.displayed_axes)
            # Redraw cut plot
            self.update_cut()
        except AttributeError:
            pass

    def set_image(self, image=None, *args, **kwargs):
        """ Wraps the underlying ImagePlot3d's set_image method.
        See :func:`~data_slicer.imageplot.ImagePlot3d.set_image`. *image* can
        be *None* i.e. in order to just update the plot with a new colormap.
        """

        # Reset the transformation
        self._transform_factors = []
        if image is None:
            image = self.image_data
        self.main_plot.set_image(image, *args, lut=self.lut, **kwargs)

    def set_cut_x_image(self, image=None, *args, **kwargs):
        """ Wraps the underlying ImagePlot3d's set_image method.
        See :func:`~data_slicer.imageplot.ImagePlot3d.set_image`. *image* can
        be *None* i.e. in order to just update the plot with a new colormap.
        """

        # Reset the transformation
        self._transform_factors = []
        if image is None:
            image = self.image_data
        self.cut_x.set_image(image, *args, **kwargs)

    def set_cut_y_image(self, image=None, *args, **kwargs):
        """ Wraps the underlying ImagePlot3d's set_image method.
        See :func:`~data_slicer.imageplot.ImagePlot3d.set_image`. *image* can
        be *None* i.e. in order to just update the plot with a new colormap.
        """

        # Reset the transformation
        self._transform_factors = []
        if image is None:
            image = self.image_data
        self.cut_y.set_image(image, *args, **kwargs)

    # color methods
    def set_cmap(self):
        """ Set the colormap to *cmap* where *cmap* is one of the names
        registered in :mod:`<data_slicer.cmaps>` which includes all matplotlib and
        kustom cmaps.
        WP: small changes made to use only my list of cmaps (see cmaps.py) and to shorten the list
        by using 'invert_colors' checkBox
        """
        try:
            cmap = self.util_panel.cmaps.currentText()
            if self.util_panel.invert_colors.isChecked() and MY_CMAPS:
                cmap = cmap + '_r'
        except AttributeError:
            cmap = DEFAULT_CMAP

        try:
            self.cmap = cmaps[cmap]
        except KeyError:
            print('Invalid colormap name. Use one of: ')
            print(cmaps.keys())
        self.cmap_name = cmap
        # Since the cmap changed it forgot our settings for alpha and gamma
        self.cmap.set_alpha(self.alpha)
        self.cmap.set_gamma()
        sliders_pos = self.get_sliders_positions()
        self.cmap_changed()
        self.set_sliders_postions(sliders_pos)
        self.update_z_binning_lines()

    def cmap_changed(self):
        """ Recalculate the lookup table and redraw the plots such that the
        changes are immediately reflected.
        """
        self.lut = self.cmap.getLookupTable()
        self.redraw_plots()

    def set_alpha(self, alpha):
        """ Set the alpha value of the currently used cmap. *alpha* can be a
        single float or an array of length ``len(self.cmap.color)``.
        """
        self.alpha = alpha
        sliders_pos = self.get_sliders_positions()
        self.cmap.set_alpha(alpha)
        self.cmap_changed()
        self.set_sliders_postions(sliders_pos)
        self.update_z_binning_lines()
        self.set_xy_binning_lines()

    def set_gamma(self):
        """ Set the exponent for the power-law norm that maps the colors to
        values. I.e. the values where the colours are defined are mapped like
        ``y=x**gamma``.
        WP: changed to work with applied QDoubleSpinBox
        """
        gamma = self.util_panel.gamma.value()
        self.gamma = gamma
        sliders_pos = self.get_sliders_positions()
        self.cmap.set_gamma(gamma)
        self.cmap_changed()
        self.update_z_binning_lines()
        self.set_sliders_postions(sliders_pos)
        self.set_xy_binning_lines()

    def set_colorscale(self):
        """ Set the relative maximum of the colormap. I.e. the colors are
        mapped to the range `min(data)` - `vmax*max(data)`.
        WP: changed to work with applied QDoubleSpinBox
        """
        vmax = self.util_panel.colorscale.value()
        self.vmax = vmax
        sliders_pos = self.get_sliders_positions()
        self.cmap.set_vmax(vmax)
        self.cmap_changed()
        self.set_sliders_postions(sliders_pos)
        self.set_xy_binning_lines()

    # sliders and binning methods
    def set_x_binning_lines(self):
        """ Update binning lines accordingly. """
        # horizontal cut
        if self.util_panel.bin_x.isChecked():
            try:
                half_width = self.util_panel.bin_x_nbins.value()
                pos = self.main_plot.pos[0].get_value()
                self.main_plot.add_binning_lines(pos, half_width, orientation='vertical')
                self.cut_x.add_binning_lines(pos, half_width, orientation='vertical')
                self.plot_x.add_binning_lines(pos, half_width)
                xmin = half_width
                xmax = len(self.data_handler.axes[0]) - half_width
                new_range = np.arange(xmin, xmax)
                self.main_plot.pos[0].set_allowed_values(new_range)
                self.cut_x.pos[0].set_allowed_values(new_range)
                self.plot_x.pos.set_allowed_values(new_range)
            except AttributeError:
                pass
        else:
            try:
                self.main_plot.remove_binning_lines(orientation='vertical')
                self.cut_x.remove_binning_lines(orientation='vertical')
                self.plot_x.remove_binning_lines()
                org_range = np.arange(0, len(self.data_handler.axes[0]))
                self.main_plot.pos[0].set_allowed_values(org_range)
                self.cut_x.pos[0].set_allowed_values(org_range)
                self.plot_x.pos.set_allowed_values(org_range)
            except AttributeError:
                pass

    def set_y_binning_lines(self):
        """ Update binning lines accordingly. """
        # vertical cut
        if self.util_panel.bin_y.isChecked():
            try:
                half_width = self.util_panel.bin_y_nbins.value()
                pos = self.main_plot.pos[1].get_value()
                self.main_plot.add_binning_lines(pos, half_width)
                self.cut_y.add_binning_lines(pos, half_width)
                self.plot_y.add_binning_lines(pos, half_width)
                ymin = half_width
                ymax = len(self.data_handler.axes[1]) - half_width
                new_range = np.arange(ymin, ymax)
                self.main_plot.pos[1].set_allowed_values(new_range)
                self.cut_y.pos[0].set_allowed_values(new_range)
                self.plot_y.pos.set_allowed_values(new_range)
            except AttributeError:
                pass
        else:
            try:
                self.main_plot.remove_binning_lines()
                self.cut_y.remove_binning_lines()
                self.plot_y.remove_binning_lines()
                org_range = np.arange(0, len(self.data_handler.axes[1]))
                self.main_plot.pos[1].set_allowed_values(org_range)
                self.cut_y.pos[0].set_allowed_values(org_range)
                self.plot_y.pos.set_allowed_values(org_range)
            except AttributeError:
                pass

    def set_zx_binning_line(self):
        # horizontal plot
        if self.util_panel.bin_zx.isChecked():
            try:
                half_width = self.util_panel.bin_zx_nbins.value()
                pos = self.cut_x.pos[1].get_value()
                self.cut_x.add_binning_lines(pos, half_width)
                zmin = half_width
                zmax = len(self.data_handler.axes[erg_ax]) - half_width
                new_range = np.arange(zmin, zmax)
                self.cut_x.pos[1].set_allowed_values(new_range)
            except AttributeError:
                pass
        else:
            try:
                self.cut_x.remove_binning_lines()
                org_range = np.arange(0, len(self.data_handler.axes[erg_ax]))
                self.cut_x.pos[1].set_allowed_values(org_range)
            except AttributeError:
                pass

    def set_zy_binning_line(self):
        # vertical plot
        if self.util_panel.bin_zy.isChecked():
            try:
                half_width = self.util_panel.bin_zy_nbins.value()
                pos = self.cut_y.pos[1].get_value()
                self.cut_y.add_binning_lines(pos, half_width, orientation='vertical')
                zmin = half_width
                zmax = len(self.data_handler.axes[erg_ax]) - half_width
                new_range = np.arange(zmin, zmax)
                self.cut_y.pos[1].set_allowed_values(new_range)
            except AttributeError:
                pass
        else:
            try:
                self.cut_y.remove_binning_lines(orientation='vertical')
                org_range = np.arange(0, len(self.data_handler.axes[erg_ax]))
                self.cut_y.pos[1].set_allowed_values(org_range)
            except AttributeError:
                pass

    def update_xy_binning_lines(self):
        if self.util_panel.bin_x.isChecked():
            try:
                pos = self.main_plot.pos[0].get_value()
                half_width = self.util_panel.bin_x_nbins.value()
                self.main_plot.add_binning_lines(pos, half_width, orientation='vertical')
                self.cut_x.add_binning_lines(pos, half_width, orientation='vertical')
                self.plot_x.add_binning_lines(pos, half_width)
            except AttributeError:
                pass
        else:
            pass

        if self.util_panel.bin_y.isChecked():
            try:
                pos = self.main_plot.pos[1].get_value()
                half_width = self.util_panel.bin_y_nbins.value()
                self.main_plot.add_binning_lines(pos, half_width)
                self.cut_y.add_binning_lines(pos, half_width)
                self.plot_y.add_binning_lines(pos, half_width)
            except AttributeError:
                pass
        else:
            pass

    def update_zx_zy_binning_line(self):
        if self.util_panel.bin_zx.isChecked():
            try:
                pos = self.cut_x.pos[1].get_value()
                half_width = self.util_panel.bin_zx_nbins.value()
                self.cut_x.add_binning_lines(pos, half_width)
            except AttributeError:
                pass
        else:
            pass

        if self.util_panel.bin_zy.isChecked():
            try:
                pos = self.cut_y.pos[1].get_value()
                half_width = self.util_panel.bin_zy_nbins.value()
                self.cut_y.add_binning_lines(pos, half_width, orientation='vertical')
            except AttributeError:
                pass
        else:
            pass

    def update_z_binning_lines(self):
        """ Update binning lines accordingly. """
        if self.util_panel.bin_z.isChecked():
            try:
                half_width = self.util_panel.bin_z_nbins.value()
                z_pos = self.data_handler.z.get_value()
                self.plot_z.add_binning_lines(z_pos, half_width)
                zmin = half_width
                zmax = len(self.data_handler.axes[2]) - half_width
                new_range = arange(zmin, zmax)
                self.plot_z.width = half_width
                self.plot_z.n_bins = half_width
                self.plot_z.pos.set_allowed_values(new_range)
                self.update_main_plot(emit=False)
            except AttributeError:
                pass
        else:
            try:
                self.plot_z.remove_binning_lines()
                self.data_handler.update_z_range()
            except AttributeError:
                pass

    def set_main_energy_slider(self):
        energy = self.util_panel.energy_main.value()
        self.data_handler.z.set_value(energy)

    def set_hor_energy_slider(self):
        energy = self.util_panel.energy_hor.value()
        self.cut_x.crosshair.hpos.set_value(energy)

    def set_vert_energy_slider(self):
        energy = self.util_panel.energy_vert.value()
        self.cut_y.crosshair.hpos.set_value(energy)

    def set_hor_momentum_slider(self):
        angle = self.util_panel.momentum_hor.value()
        self.main_plot.pos[1].set_value(angle)

    def set_vert_momentum_slider(self):
        angle = self.util_panel.momentum_vert.value()
        self.main_plot.pos[0].set_value(angle)

    def put_sliders_in_initial_positions(self):
        if self.data_handler.axes[erg_ax].min() < 0:
            mid_energy = wp.indexof(-0.01, self.data_handler.axes[erg_ax])
        else:
            mid_energy = int(len(self.data_handler.axes[erg_ax]) / 2)
        mid_hor_angle = int(len(self.data_handler.axes[scan_ax]) / 2)
        mid_vert_angle = int(len(self.data_handler.axes[slit_ax]) / 2)

        self.data_handler.z.set_value(mid_energy)
        self.cut_x.crosshair.hpos.set_value(mid_energy)
        self.cut_y.crosshair.hpos.set_value(mid_energy)
        self.main_plot.pos[0].set_value(mid_hor_angle)
        self.main_plot.pos[1].set_value(mid_vert_angle)

    def get_sliders_positions(self):
        main_hor = self.main_plot.crosshair.hpos.get_value()
        main_ver = self.main_plot.crosshair.vpos.get_value()
        cut_hor_energy = self.cut_x.crosshair.hpos.get_value()
        cut_ver_energy = self.cut_y.crosshair.hpos.get_value()
        return [main_hor, main_ver, cut_hor_energy, cut_ver_energy]

    def set_sliders_postions(self, positions):
        self.main_plot.pos[1].set_value(positions[0])
        self.main_plot.pos[0].set_value(positions[1])
        self.cut_x.pos[1].set_value(positions[2])
        self.cut_y.pos[1].set_value(positions[3])

    def set_sliders_labels(self, dataset):
        if hasattr(dataset, 'scan_type'):
            if dataset.scan_type == 'hv scan':
                self.util_panel.momentum_vert_label.setText('hv:')
                self.util_panel.momentum_hor_label.setText('kx:')
                self.util_panel.energy_hor_label.setText('hv:')
                self.util_panel.energy_vert_label.setText('kx:')
                self.util_panel.bin_x.setText('bin hv')
                self.util_panel.bin_zx.setText('bin E (hv)')
        else:
            return

    def close_mw(self):
        self.destroy()
        self.db.thread[self.index].stop()
