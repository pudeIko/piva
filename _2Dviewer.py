"""
Data handler and main window creator for 2D data inspection
"""
import time
from PyQt5.QtWidgets import QMainWindow
import numpy as np
from imageplot import *
from cmaps import cmaps
import arpys_wp as wp
from copy import deepcopy

app_style = """
QMainWindow{background-color: rgb(64,64,64);}
"""
DEFAULT_CMAP = 'viridis'
NDIM = 3


class DataHandler:
    """ Object that keeps track of a set of 3D data and allows
    manipulations on it. In a Model-View-Controller framework this could be
    seen as the Model, while :class:`MainWindow <data_slicer.pit.MainWindow>`
    would be the View part.
    """

    def __init__(self, main_window):
        self.main_window = main_window
        self.binning = False

        # Initialize instance variables
        # np.array that contains the 3D data
        self.data = None
        self.axes = np.array([[0, 1], [0, 1], [0, 1]])
        # Indices of *data* that are displayed in the main plot
        self.displayed_axes = (0, 1)
        # Index along the z axis at which to produce a slice
        # self.z = TracedVariable(0, name='z')
        # # Number of slices to integrate along z
        # integrate_z = TracedVariable(value=0, name='integrate_z')
        # How often we have rolled the axes from the original setup
        # self._roll_state = 0

        # moved to get rid of warnings
        # self.zmin = 0
        # self.zmax = None
        # self.integrated = None
        # self.cut_y_data = None
        # self.cut_x_data = None

    def get_data(self):
        """ Convenience `getter` method. Allows writing ``self.get_data()``
        instead of ``self.data.get_value()``.
        """
        return self.data.get_value()[0, :, :]

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
        self.axes = np.array(axes, dtype="object")

        self.prepare_axes()
        self.on_z_dim_change()

        self.main_window.update_main_plot()
        self.main_window.set_axes()

    def prepare_axes(self):
        """ Create a list containing the three original x-, y- and z-axes
        and replace *None* with the amount of pixels along the given axis.
        """
        shapes = self.data.get_value().shape
        # Avoid undefined axes scales and replace them with len(1) sequences
        for i, axis in enumerate(self.axes):
            if axis is None:
                self.axes[i] = np.arange(shapes[i])

    def on_data_change(self):
        """ Update self.main_window.image_data and replot. """
        self.update_image_data()
        # self.main_window.redraw_plots()
        # Also need to recalculate the intensity plot
        # self.on_z_dim_change()

    def on_z_dim_change(self):
        """ Called when either completely new data is loaded or the dimension
        from which we look at the data changed (e.g. through :func:`roll_axes
        <data_slicer.pit.PITDataHandler.roll_axes>`).
        Update the z range and the integrated intensity plot.
        """

        # Calculate the integrated intensity and plot it
        self.calculate_integrated_intensity()
        # ip.plot(self.integrated)

    def calculate_integrated_intensity(self):
        self.integrated = self.get_data().sum(0).sum(0)

    def update_image_data(self):
        """ Get the right (possibly integrated) slice out of *self.data*,
        apply postprocessings and store it in *self.image_data*.
        Skip this if the z value happens to be out of range, which can happen
        if the image data changes and the z scale hasn't been updated yet.
        """
        # z = self.z.get_value()
        # integrate_z = self.main_window.plot_z.width
        data = self.get_data()
        try:
            self.main_window.image_data = data
        except IndexError:
            print(('update_image_data(): z index {} out of range for data of length {}.'))


class MainWindow2D(QMainWindow):

    def __init__(self, data_browser, data_set=None, title=None, index=0):
        super(MainWindow2D, self).__init__()
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
        self.slider_pos = [0, 0]

        # Initialize instance variables
        # Plot transparency alpha
        self.alpha = 1
        # Plot powerlaw normalization exponent gamma
        self.gamma = 1
        # Relative colormap maximum
        self.vmax = 1

        # Need to store original transformation information for `rotate()`
        self._transform_factors = []

        self.data_handler = DataHandler(self)

        # Create the 3D (main) and cut ImagePlots
        self.main_plot = ImagePlot(name='main_plot', crosshair=True)
        # Create cut of cut_x
        self.plot_x = CursorPlot(name='plot_x')
        # Create cut of cut_y
        self.plot_y = CursorPlot(name='plot_y', orientation='horizontal')
        # Create utilities panel
        self.util_panel = UtilitiesPanel(self, name='utilities_panel', dim=2)
        self.util_panel.positions_momentum_label.setText('Sliders')
        # self.util_panel.momentum_vert_label.setText('E:')

        self.setStyleSheet(app_style)
        self.set_cmap()

        self.setGeometry(50, 50, 700, 800)
        self.setWindowTitle(self.title)
        self.initUI()

        time_dl_b = time.time()
        # Set the loaded data in PIT
        if data_set is None:
            print('Data set to open not defined.')
            self.close_mw()
        else:
            raw_data = data_set

        D = self.swap_axes_aroud(raw_data)

        # print(f'data shape = {D.data.shape}')
        # print(f'x shape = {D.xscale.shape}')
        # print(f'y shape = {D.yscale.shape}')
        # print(f'z shape = {D.zscale.shape}')
        self.data_handler.prepare_data(D.data, [D.xscale, D.yscale, D.zscale])
        # print('data shape = {}'.format(D.data.shape))

        self.util_panel.energy_vert.setRange(0, len(self.data_handler.axes[1]))
        self.util_panel.momentum_hor.setRange(0, len(self.data_handler.axes[0]))

        self.put_sliders_in_the_middle()

    def initUI(self):
        self.setWindowTitle(self.title)
        # Create a "central widget" and its layout
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        self.util_panel.bin_y_nbins.setValue(10)
        self.util_panel.bin_z_nbins.setValue(5)

        # Create plot_x
        self.plot_x.register_traced_variable(self.main_plot.pos[0])
        self.plot_x.pos.sig_value_changed.connect(self.update_plot_y)

        # Create plot_y
        self.plot_y.register_traced_variable(self.main_plot.pos[1])
        self.plot_y.pos.sig_value_changed.connect(self.update_plot_x)

        # Create utilities panel
        self.util_panel.close_button.clicked.connect(self.close_mw)
        # and connect signals
        self.util_panel.cmaps.currentIndexChanged.connect(self.set_cmap)
        self.util_panel.invert_colors.stateChanged.connect(self.set_cmap)
        self.util_panel.gamma.valueChanged.connect(self.set_gamma)
        self.util_panel.colorscale.valueChanged.connect(self.set_alpha)
        self.util_panel.bin_y.stateChanged.connect(self.update_binning_lines)
        self.util_panel.bin_y_nbins.valueChanged.connect(self.update_binning_lines)
        self.util_panel.bin_z.stateChanged.connect(self.update_binning_lines)
        self.util_panel.bin_z_nbins.valueChanged.connect(self.update_binning_lines)
        self.util_panel.energy_vert.valueChanged.connect(self.set_vert_energy_slider)
        self.util_panel.momentum_hor.valueChanged.connect(self.set_hor_momentum_slider)

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
        l.addWidget(self.util_panel, 0, 0, 2 * sd, 6 * sd)
        # EDC
        l.addWidget(self.plot_x, 2 * sd, 0, 3 * sd, 4 * sd)
        # Main plot
        l.addWidget(self.main_plot, 5 * sd, 0, 6 * sd, 4 * sd)
        # MDC
        l.addWidget(self.plot_y, 5 * sd, 4 * sd, 6 * sd, 2 * sd)

        nrows = 12 * sd
        ncols = 6 * sd
        # Need to manually set all row- and columnspans as well as min-sizes
        for i in range(nrows):
            l.setRowMinimumHeight(i, 50)
            l.setRowStretch(i, 1)
        for i in range(ncols):
            l.setColumnMinimumWidth(i, 50)
            l.setColumnStretch(i, 1)

    @staticmethod
    def swap_axes_aroud(D):
        """
        Swap axes and data dimensions to fit rest of the code
        """
        tmp = deepcopy(D)
        D.xscale = tmp.yscale
        D.yscale = tmp.zscale
        D.zscale = tmp.xscale
        D.data = np.ones((1, tmp.zscale.size, tmp.yscale.size))
        D.data[0, :, :] = np.rollaxis(tmp.data[0, :, :], 1)
        # print('viewer format')
        # print(f'data shape: {D.data.shape}')
        # print(f'x shape: {D.xscale.shape}')
        # print(f'y shape: {D.yscale.shape}')
        # print(f'z shape: {D.zscale.shape}')

        return D

    def update_main_plot(self, **image_kwargs):
        """ Change *self.main_plot*`s currently displayed
        :class:`image_item <data_slicer.imageplot.ImagePlot.image_item>` to
        the slice of *self.data_handler.data* corresponding to the current
        value of *self.z*.
        """

        self.data_handler.update_image_data()

        if image_kwargs != {}:
            self.image_kwargs = image_kwargs
        # Add image to main_plot
        self.set_image(self.image_data, **image_kwargs)

    def set_axes(self):
        """ Set the x- and y-scales of the plots. The :class:`ImagePlot
        <data_slicer.imageplot.ImagePlot>` object takes care of keeping the
        scales as they are, once they are set.
        """
        xaxis = self.data_handler.axes[1]
        yaxis = self.data_handler.axes[0]
        self.main_plot.set_xscale(range(0, len(xaxis)))
        self.main_plot.set_ticks(xaxis[0], xaxis[-1], self.main_plot.main_xaxis)
        self.plot_x.set_ticks(xaxis[0], xaxis[-1], self.plot_x.main_xaxis)
        self.plot_x.set_secondary_axis(0, len(xaxis))
        self.main_plot.set_yscale(range(0, len(yaxis)))
        self.main_plot.set_ticks(yaxis[0], yaxis[-1], self.main_plot.main_yaxis)
        self.plot_y.set_ticks(yaxis[0], yaxis[-1], self.plot_y.main_xaxis)
        self.plot_y.set_secondary_axis(0, len(yaxis))
        self.main_plot.fix_viewrange()

    def update_plot_x(self):
        # Get shorthands for plot
        xp = self.plot_x
        try:
            old = xp.listDataItems()[0]
            xp.removeItem(old)
        except IndexError:
            pass

        # Get the correct position indicator
        pos = self.main_plot.pos[1]
        binning = self.util_panel.bin_y.isChecked()
        if binning:
            width = self.util_panel.bin_y_nbins.value()
        else:
            width = 0
        if pos.allowed_values is not None:
            i_x = int(min(pos.get_value(), pos.allowed_values.max() - 1))
        else:
            i_x = 0
        data = self.data_handler.get_data()
        if width == 0:
            y = data[:, i_x]
        else:
            start = i_x - width
            stop = i_x + width
            y = np.sum(data[:, start:stop], axis=1)
            # y = wp.normalize(y)
        x = np.arange(0, len(self.data_handler.axes[1]))
        xp.plot(x, y)
        self.util_panel.momentum_hor.setValue(i_x)
        self.util_panel.momentum_hor_value.setText('({:.3f})'.format(self.data_handler.axes[0][i_x]))
        if binning:
            self.plot_y.left_line.setValue(i_x - width)
            self.plot_y.right_line.setValue(i_x + width)

    def update_plot_y(self):
        # Get shorthands for plot
        yp = self.plot_y
        try:
            old = yp.listDataItems()[0]
            yp.removeItem(old)
        except IndexError:
            pass

        # Get the correct position indicator
        pos = self.main_plot.pos[0]
        binning = self.util_panel.bin_z.isChecked()
        if binning:
            width = self.util_panel.bin_z_nbins.value()
        else:
            width = 0
        if pos.allowed_values is not None:
            i_y = int(min(pos.get_value(), pos.allowed_values.max() - 1))
        else:
            i_y = 0
        data = self.data_handler.get_data()
        if width == 0:
            y = data[i_y, :]
        else:
            start = i_y - width
            stop = i_y + width
            y = np.sum(data[start:stop, :], axis=0)
            # y = wp.normalize(y)
        # y = data[i_y, :]
        x = np.arange(0, len(self.data_handler.axes[0]))
        yp.plot(y, x)
        self.util_panel.energy_vert.setValue(i_y)
        self.util_panel.energy_vert_value.setText('({:.4f})'.format(self.data_handler.axes[1][i_y]))
        if binning:
            self.plot_x.left_line.setValue(i_y - width)
            self.plot_x.right_line.setValue(i_y + width)

    def update_xy_plots(self):
        """ Update the x and y profile plots. """
        self.update_x_plot()
        self.update_y_plot()

    def update_binning_lines(self):
        """ Update binning lines accordingly. """
        # edc plot
        if self.util_panel.bin_y.isChecked():
            try:
                half_width = self.util_panel.bin_y_nbins.value()
                pos = self.main_plot.pos[1].get_value()
                self.main_plot.add_binning_lines(pos, half_width)
                self.plot_y.add_binning_lines(pos, half_width)
                ymin = 0 + half_width
                ymax = len(self.data_handler.axes[0]) - half_width
                new_range = np.arange(ymin, ymax)
                self.main_plot.pos[1].set_allowed_values(new_range)
                self.plot_y.pos.set_allowed_values(new_range)
            except AttributeError:
                pass
        else:
            try:
                self.main_plot.remove_binning_lines()
                self.plot_y.remove_binning_lines()
                org_range = np.arange(0, len(self.data_handler.axes[0]))
                self.main_plot.pos[1].set_allowed_values(org_range)
                self.plot_y.pos.set_allowed_values(org_range)
            except AttributeError:
                pass

        # mdc plot
        if self.util_panel.bin_z.isChecked():
            try:
                half_width = self.util_panel.bin_z_nbins.value()
                pos = self.main_plot.pos[0].get_value()
                self.main_plot.add_binning_lines(pos, half_width, orientation='vertical')
                self.plot_x.add_binning_lines(pos, half_width)
                ymin = 0 + half_width
                ymax = len(self.data_handler.axes[1]) - half_width
                new_range = np.arange(ymin, ymax)
                self.main_plot.pos[0].set_allowed_values(new_range)
                self.plot_x.pos.set_allowed_values(new_range)
            except AttributeError:
                pass
        else:
            try:
                self.main_plot.remove_binning_lines(orientation='vertical')
                self.plot_x.remove_binning_lines()
                org_range = np.arange(0, len(self.data_handler.axes[0]))
                self.main_plot.pos[0].set_allowed_values(org_range)
                self.plot_x.pos.set_allowed_values(org_range)
            except AttributeError:
                pass

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
        self.update_binning_lines()

    def cmap_changed(self):
        """ Recalculate the lookup table and redraw the plots such that the
        changes are immediately reflected.
        """
        self.lut = self.cmap.getLookupTable()
        self.redraw_plots()

    def redraw_plots(self, image=None):
        """ Redraw plotted data to reflect changes in data or its colors. """
        try:
            # Redraw main plot
            self.set_image(image, displayed_axes=self.data_handler.displayed_axes)
            # Redraw cut plot
            self.update_cut()
        except AttributeError as e:
            # In some cases (namely initialization) the mainwindow is not defined yet
            pass
            # print('AttributeError: {}'.format(e))

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

    def set_alpha(self, alpha):
        """ Set the alpha value of the currently used cmap. *alpha* can be a
        single float or an array of length ``len(self.cmap.color)``.
        """
        self.alpha = alpha
        sliders_pos = self.get_sliders_positions()
        self.cmap.set_alpha(alpha)
        self.cmap_changed()
        self.set_sliders_postions(sliders_pos)
        self.update_binning_lines()

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
        self.set_sliders_postions(sliders_pos)
        self.update_binning_lines()

    def set_colorscale(self):
        """ Set the relative maximum of the colormap. I.e. the colors are
        mapped to the range `min(data)` - `vmax*max(data)`.
        WP: changed to work with applied QDoubleSpinBox
        """
        vmax = self.util_panel.colorscale.value()
        self.vmax = vmax
        self.cmap.set_vmax(vmax)
        self.cmap_changed()

    def set_vert_energy_slider(self):
        energy = self.util_panel.energy_vert.value()
        self.main_plot.pos[0].set_value(energy)
        self.update_binning_lines()

    def set_hor_momentum_slider(self):
        angle = self.util_panel.momentum_hor.value()
        self.main_plot.pos[1].set_value(angle)
        self.update_binning_lines()

    def set_sliders_postions(self, positions):
        self.main_plot.pos[1].set_value(positions[0])
        self.main_plot.pos[0].set_value(positions[1])

    def get_sliders_positions(self):
        main_hor = self.main_plot.crosshair.hpos.get_value()
        main_ver = self.main_plot.crosshair.vpos.get_value()
        return [main_hor, main_ver]

    def put_sliders_in_the_middle(self):
        mid_energy = int(len(self.data_handler.axes[1]) / 2)
        mid_angle = int(len(self.data_handler.axes[0]) / 2)

        self.main_plot.pos[0].set_value(mid_energy)
        self.main_plot.pos[1].set_value(mid_angle)

    def close_mw(self):
        self.destroy()
        self.db.thread[self.index].stop()

