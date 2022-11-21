"""
Data handler and main window creator for 2D data inspection
"""
import os
import time
from copy import deepcopy

import numpy as np
from pyqtgraph.Qt import QtWidgets
from PyQt5.QtWidgets import QMessageBox, QLineEdit

import piva.arpys_wp as wp
import piva.data_loader as dl
import piva.imageplot as ip
from data_slicer import pit
from piva.cmaps import cmaps
from piva.edc_fitter import EDCFitter
from piva.mdc_fitter import MDCFitter

app_style = """
QMainWindow{background-color: rgb(64,64,64);}
QTabWidget{background-color: rgb(64,64,64);}
"""
DEFAULT_CMAP = 'viridis'
ORIENTLINES_LINECOLOR = (164, 37, 22, 255)
erg_ax = 2
scan_ax = 0


class DataHandler2D :
    """ Object that keeps track of a set of 2D data and allows
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
        if self.main_window.util_panel.image_normalize_edcs.isChecked():
            return self.norm_data
        else:
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

        self.data = ip.TracedVariable(data, name='data')
        self.axes = np.array(axes, dtype="object")
        self.norm_data = wp.normalize(self.data.get_value()[0, :, :].T).T

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


class MainWindow2D(QtWidgets.QMainWindow):

    def __init__(self, data_browser, data_set=None, index=None, slice=False):
        super(MainWindow2D, self).__init__()
        self.title = index.split('/')[-1]
        self.fname = index
        self.central_widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout()

        # moved to get rid of warnings
        self.cmap = None
        self.image_kwargs = None
        self.cmap_name = None
        self.lut = None
        self.db = data_browser
        self.index = index
        self.slice = slice
        self.slider_pos = [0, 0]
        self.new_energy_axis = None
        self.k_axis = None
        self.smooth = False
        self.curvature = False
        self.thread = {}
        # self.thread_count = 0
        self.data_viewers = {}

        # Initialize instance variables
        # Plot transparency alpha
        self.alpha = 1
        # Plot powerlaw normalization exponent gamma
        self.gamma = 1
        # Relative colormap maximum
        self.vmax = 1

        # Need to store original transformation information for `rotate()`
        self._transform_factors = []

        self.data_set = deepcopy(data_set)
        self.org_dataset = None
        self.data_handler = DataHandler2D(self)

        # Create the 3D (main) and cut ImagePlots
        self.main_plot = ip.ImagePlot(name='main_plot', crosshair=True)
        # Create cut of cut_x
        self.plot_x = ip.CursorPlot(name='plot_x')
        # Create cut of cut_y
        self.plot_y = ip.CursorPlot(name='plot_y', orientation='vertical')
        # Create utilities panel
        self.util_panel = ip.UtilitiesPanel(self, name='utilities_panel', dim=2)
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
            print('Data set not defined.')
        else:
            raw_data = data_set

        D = self.swap_axes_aroud(raw_data)

        # print(f'data shape = {D.data.shape}')
        # print(f'x shape = {D.xscale.shape}')
        # print(f'y shape = {D.yscale.shape}')
        # print(f'z shape = {D.zscale.shape}')
        self.data_handler.prepare_data(D.data, [D.xscale, D.yscale, D.zscale])
        self.org_image_data = self.data_handler.get_data()

        self.util_panel.energy_vert.setRange(0, len(self.data_handler.axes[1]))
        self.util_panel.momentum_hor.setRange(0, len(self.data_handler.axes[0]))

        try:
            self.load_saved_corrections(data_set)
        except AttributeError:
            print('going with old settings.')
            self.load_saved_corrections_old(data_set)

        self.util_panel.set_metadata_window(raw_data)
        self.put_sliders_in_initial_positions()

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

        # Create utilities panel and connect signals
        self.util_panel.image_cmaps.currentIndexChanged.connect(self.set_cmap)
        self.util_panel.image_invert_colors.stateChanged.connect(self.set_cmap)
        self.util_panel.image_gamma.valueChanged.connect(self.set_gamma)
        self.util_panel.image_colorscale.valueChanged.connect(self.set_alpha)
        self.util_panel.image_normalize_edcs.stateChanged.connect(self.update_main_plot)
        self.util_panel.image_smooth_button.clicked.connect(self.smoooth_data)
        self.util_panel.image_curvature_button.clicked.connect(self.curvature_method)

        self.util_panel.file_mdc_fitter_button.clicked.connect(self.open_mdc_fitter)
        self.util_panel.file_edc_fitter_button.clicked.connect(self.open_edc_fitter)

        # binning utilities
        self.util_panel.bin_y.stateChanged.connect(self.update_binning_lines)
        self.util_panel.bin_y_nbins.valueChanged.connect(self.update_binning_lines)
        self.util_panel.bin_z.stateChanged.connect(self.update_binning_lines)
        self.util_panel.bin_z_nbins.valueChanged.connect(self.update_binning_lines)
        self.util_panel.energy_vert.valueChanged.connect(self.set_vert_energy_slider)
        self.util_panel.momentum_hor.valueChanged.connect(self.set_hor_momentum_slider)

        # buttons
        self.util_panel.close_button.clicked.connect(self.close)
        self.util_panel.save_button.clicked.connect(self.save_to_pickle)
        self.util_panel.pit_button.clicked.connect(self.open_pit)

        # energy and k-space concersion
        self.util_panel.axes_energy_Ef.valueChanged.connect(self.apply_energy_correction)
        self.util_panel.axes_energy_hv.valueChanged.connect(self.apply_energy_correction)
        self.util_panel.axes_energy_wf.valueChanged.connect(self.apply_energy_correction)
        self.util_panel.axes_energy_scale.currentIndexChanged.connect(self.apply_energy_correction)
        self.util_panel.axes_do_kspace_conv.clicked.connect(self.convert_to_kspace)
        self.util_panel.axes_reset_conv.clicked.connect(self.reset_kspace_conversion)

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

    def closeEvent(self, event) :
        """ Ensure that this instance is un-registered from the DataBrowser. """
        self.db.thread[self.index].quit()
        self.db.thread[self.index].wait()
        del(self.db.thread[self.index])
        del(self.db.data_viewers[self.index])

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

        try:
            data = self.image_data
        except AttributeError:
            data = self.data_handler.get_data()

        if width == 0:
            y = data[:, i_x]
        else:
            start = i_x - width
            stop = i_x + width
            y = np.sum(data[:, start:stop], axis=1)
            # y = wp.normalize(y)
        x = np.arange(0, len(self.data_handler.axes[1]))
        self.edc = y
        xp.plot(x, y)
        self.util_panel.momentum_hor.setValue(i_x)
        if self.k_axis is None:
            self.util_panel.momentum_hor_value.setText('({:.4f})'.format(self.data_handler.axes[0][i_x]))
        else:
            self.util_panel.momentum_hor_value.setText('({:.4f})'.format(self.k_axis[i_x]))

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
        try:
            data = self.image_data
        except AttributeError:
            data = self.data_handler.get_data()
        if width == 0:
            y = data[i_y, :]
        else:
            start = i_y - width
            stop = i_y + width
            y = np.sum(data[start:stop, :], axis=0)
            # y = wp.normalize(y)
        x = np.arange(0, len(self.data_handler.axes[0]))
        self.mdc = y
        yp.plot(y, x)
        self.util_panel.energy_vert.setValue(i_y)

        if self.new_energy_axis is None:
            self.util_panel.energy_vert_value.setText('({:.4f})'.format(self.data_handler.axes[1][i_y]))
        else:
            self.util_panel.energy_vert_value.setText('({:.4f})'.format(self.new_energy_axis[i_y]))

        if binning:
            self.plot_x.left_line.setValue(i_y - width)
            self.plot_x.right_line.setValue(i_y + width)

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
                org_range = np.arange(0, len(self.data_handler.axes[1]))
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
            cmap = self.util_panel.image_cmaps.currentText()
            if self.util_panel.image_invert_colors.isChecked() and ip.MY_CMAPS:
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
            # self.update_cut()
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
        gamma = self.util_panel.image_gamma.value()
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
        vmax = self.util_panel.image_colorscale.value()
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

    def put_sliders_in_initial_positions(self):
        if self.new_energy_axis is None:
            e_ax = self.data_handler.axes[1]
        else:
            e_ax = self.new_energy_axis
        if (e_ax.min() < 0) and (e_ax.max() > 0):
            mid_energy = wp.indexof(-0.005, e_ax)
        else:
            mid_energy = int(len(e_ax) / 2)
            self.util_panel.axes_energy_scale.setCurrentIndex(1)

        if self.k_axis is None:
            mom_ax = self.data_handler.axes[0]
        else:
            mom_ax = self.k_axis
        if (mom_ax.min() < 0) and (mom_ax.max() > 0):
            mid_angle = wp.indexof(0, mom_ax)
        else:
            mid_angle = int(len(mom_ax) / 2)

        self.main_plot.pos[0].set_value(mid_energy)
        self.main_plot.pos[1].set_value(mid_angle)

    def apply_energy_correction(self):
        Ef = self.util_panel.axes_energy_Ef.value()
        hv = self.util_panel.axes_energy_hv.value()
        wf = self.util_panel.axes_energy_wf.value()

        scale = self.util_panel.axes_energy_scale.currentText()
        if scale == 'binding':
            hv = 0
            wf = 0

        new_energy_axis = self.data_handler.axes[1] + Ef - hv + wf
        self.new_energy_axis = new_energy_axis
        new_range = [new_energy_axis[0], new_energy_axis[-1]]
        self.main_plot.plotItem.getAxis(self.main_plot.main_xaxis).setRange(*new_range)
        self.plot_x.plotItem.getAxis(self.plot_x.main_xaxis).setRange(*new_range)

        # update energy labels
        erg_idx = self.main_plot.crosshair.vpos.get_value()
        self.util_panel.momentum_hor_value.setText('({:.4f})'.format(self.new_energy_axis[erg_idx]))

    def convert_to_kspace(self):
        scan_ax = np.array([0])
        anal_axis = self.data_handler.axes[0]
        d_scan_ax = self.util_panel.axes_angle_off.value()
        d_anal_ax = self.data_handler.axes[0][self.util_panel.axes_gamma_x.value()]
        orientation = self.util_panel.axes_slit_orient.currentText()
        a = self.util_panel.axes_conv_lc.value()
        energy = self.new_energy_axis
        hv = self.util_panel.axes_energy_hv.value()
        wf = self.util_panel.axes_energy_wf.value()

        if hv == 0 or wf == 0:
            warning_box = QMessageBox()
            warning_box.setIcon(QMessageBox.Information)
            warning_box.setWindowTitle('Wrong conversion values.')
            if hv == 0 and wf == 0:
                msg = 'Photon energy and work fonction values not given.'
            elif hv == 0:
                msg = 'Photon energy value not given.'
            elif wf == 0:
                msg = 'Work fonction value not given.'
            warning_box.setText(msg)
            warning_box.setStandardButtons(QMessageBox.Ok)
            if warning_box.exec() == QMessageBox.Ok:
                return

        nma, erg = wp.angle2kscape(scan_ax, anal_axis, d_scan_ax=d_scan_ax, d_anal_ax=d_anal_ax,
                                   orientation=orientation, a=a, energy=energy, hv=hv, work_func=wf)
        self.k_axis = nma[-1]
        new_range = [nma[-1][0], nma[-1][-1]]
        self.main_plot.plotItem.getAxis(self.main_plot.main_yaxis).setRange(*new_range)
        self.plot_y.plotItem.getAxis(self.plot_y.main_xaxis).setRange(*new_range)
        self.util_panel.momentum_hor_value.setText('({:.4f})'.format(self.k_axis[self.main_plot.pos[1].get_value()]))

    def reset_kspace_conversion(self):
        self.k_axis = None
        org_range = [self.data_handler.axes[0][0], self.data_handler.axes[0][-1]]
        self.main_plot.plotItem.getAxis(self.main_plot.main_yaxis).setRange(*org_range)
        self.plot_y.plotItem.getAxis(self.plot_y.main_xaxis).setRange(*org_range)
        self.util_panel.momentum_hor_value.setText('({:.4f})'.format(
            self.data_handler.axes[0][self.main_plot.pos[1].get_value()]))

    def smoooth_data(self):
        self.smooth = not self.smooth
        if self.smooth:
            data = self.data_handler.get_data()
            nb = self.util_panel.image_smooth_n.value()
            rl = self.util_panel.image_smooth_rl.value()
            self.image_data = wp.smooth_2d(data, n_box=nb, recursion_level=rl)
            self.set_image()
            self.util_panel.image_smooth_button.setText('Reset')
        else:
            self.image_data = deepcopy(self.org_image_data)
            self.set_image()
            self.curvature = False
            self.util_panel.image_smooth_button.setText('Smooth')
            self.util_panel.image_curvature_button.setText('Do curvature')

    def curvature_method(self):
        self.curvature = not self.curvature
        if self.curvature:
            a = self.util_panel.image_curvature_a.value()
            if self.k_axis is None:
                dx = wp.get_step(self.data_handler.axes[0])
            else:
                dx = wp.get_step(self.k_axis)
            if self.new_energy_axis is None:
                dy = wp.get_step(self.data_handler.axes[1])
            else:
                dy = wp.get_step(self.new_energy_axis)
            self.smooth_data = deepcopy(self.image_data)
            method = self.util_panel.image_curvature_method.currentText()
            if method == '2D':
                self.image_data = wp.curvature_2d(self.image_data, dx, dy, a0=a)
            elif method == '1D (EDC)':
                for idx in range(self.image_data.shape[1]):
                    self.image_data[:, idx] = wp.curvature_1d(self.image_data[:, idx], a0=a)
            elif method == '1D (MDC)':
                for idx in range(self.image_data.shape[0]):
                    self.image_data[idx, :] = wp.curvature_1d(self.image_data[idx, :], a0=a)
            self.set_image()
            self.util_panel.image_curvature_button.setText('Reset')
        else:
            if self.smooth:
                self.image_data = self.smooth_data
            else:
                self.image_data = deepcopy(self.org_image_data)
            self.set_image()
            self.util_panel.image_curvature_button.setText('Do curvature')

    def open_mdc_fitter(self):

        thread_lbl = self.index + '_mdc_viewer'
        self.mdc_thread_lbl = thread_lbl

        if thread_lbl in self.data_viewers:
            already_opened_box = QMessageBox()
            already_opened_box.setIcon(QMessageBox.Information)
            already_opened_box.setText('MDC viewer already opened.')
            already_opened_box.setStandardButtons(QMessageBox.Ok)
            if already_opened_box.exec() == QMessageBox.Ok:
                return

        self.thread[thread_lbl] = ip.ThreadClass(index=thread_lbl)
        self.thread[thread_lbl].start()
        title = self.title + ' - mdc fitter'
        if self.new_energy_axis is not None:
            erg_ax = self.new_energy_axis
        else:
            erg_ax = self.data_set.zscale
        if self.k_axis is not None:
            k_ax = self.k_axis
        else:
            k_ax = self.data_set.yscale
        axes = [k_ax, erg_ax]
        try:
            self.data_viewers[thread_lbl] = MDCFitter(self, self.data_set, axes, title, index=thread_lbl)
        except Exception:
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Information)
            error_box.setText('Couldn\'t load data,  something went wrong.')
            error_box.setStandardButtons(QMessageBox.Ok)
            if error_box.exec() == QMessageBox.Ok:
                return

    def open_edc_fitter(self):

        thread_idx = self.index + '_edc_viewer'
        self.edc_thread_lbl = thread_idx

        if thread_idx in self.data_viewers:
            already_opened_box = QMessageBox()
            already_opened_box.setIcon(QMessageBox.Information)
            already_opened_box.setText('EDC viewer already opened.')
            already_opened_box.setStandardButtons(QMessageBox.Ok)
            if already_opened_box.exec() == QMessageBox.Ok:
                return

        self.thread[thread_idx] = ip.ThreadClass(index=thread_idx)
        self.thread[thread_idx].start()
        title = self.title + ' - edc fitter'
        if self.new_energy_axis is not None:
            erg_ax = self.new_energy_axis
        else:
            erg_ax = self.data_set.zscale
        if self.k_axis is not None:
            k_ax = self.k_axis
        else:
            k_ax = self.data_set.yscale
        axes = [k_ax, erg_ax]
        try:
            self.data_viewers[thread_idx] = EDCFitter(self, self.data_set, axes, title, index=thread_idx)
        except Exception:
            error_box = QMessageBox()
            error_box.setIcon(QMessageBox.Information)
            error_box.setText('Couldn\'t load data,  something went wrong.')
            error_box.setStandardButtons(QMessageBox.Ok)
            if error_box.exec() == QMessageBox.Ok:
                return

    def save_to_pickle(self):
        # TODO change 'k_axis' to 'k' for all cuts: add 'change attrs name'
        # energy axis
        dataset = self.data_set
        savedir = self.fname[:-len(self.title)]
        up = self.util_panel
        file_selection = True
        # Prepare a filename with the .p suffix
        init_fname = '.'.join(self.title.split('.')[:-1] + ['p'])

        while file_selection:
            fname, fname_return_value = \
                    QtWidgets.QInputDialog.getText(self, '', 'File name:', 
                                                   QtWidgets.QLineEdit.Normal,
                                                   init_fname)
            if not fname_return_value:
                return

            # check if there is no fname colosions
            if fname in os.listdir(savedir):
                fname_colision_box = QMessageBox()
                fname_colision_box.setIcon(QMessageBox.Question)
                fname_colision_box.setWindowTitle('File name already used.')
                fname_colision_box.setText('File {} already exists.\nDo you want to overwrite it?'.format(fname))
                fname_colision_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                if fname_colision_box.exec() == QMessageBox.Ok:
                    file_selection = False
                else:
                    init_fname = fname
            else:
                file_selection = False

        conditions = [up.axes_energy_Ef.value() != 0, up.axes_energy_hv.value() != 0, up.axes_energy_wf.value() != 0,
                      up.axes_angle_off.value() != 0, up.axes_gamma_x.value() != 0]

        if np.any(conditions):
            save_cor_box = QMessageBox()
            save_cor_box.setIcon(QMessageBox.Question)
            save_cor_box.setWindowTitle('Save data')
            save_cor_box.setText("Do you want to save applied corrections?")
            save_cor_box.setStandardButtons(QMessageBox.No | QMessageBox.Ok | QMessageBox.Cancel)

            box_return_value = save_cor_box.exec()
            if box_return_value == QMessageBox.Ok:
                if up.axes_energy_Ef.value() != 0:
                    dataset.Ef = up.axes_energy_Ef.value()
                if up.axes_energy_hv.value() != 0:
                    dataset.hv = up.axes_energy_hv.value()
                if up.axes_energy_wf.value() != 0:
                    dataset.wf = up.axes_energy_wf.value()
                # if up.axes_angle_off.value() != 0:
                #     attrs['angle_off'] = up.axes_angle_off.value()
                if not (self.k_axis is None):
                    dataset.kyscale = self.k_axis
            elif box_return_value == QMessageBox.No:
                pass
            elif box_return_value == QMessageBox.Cancel:
                return
        else:
            pass

        dl.dump(dataset, (savedir + fname), force=True)

    def load_saved_corrections(self, data_set):

        if hasattr(data_set, 'saved'):
            raise AttributeError
        if not (data_set.Ef is None):
            self.util_panel.axes_energy_Ef.setValue(data_set.Ef)
        if not (data_set.hv is None):
            self.util_panel.axes_energy_hv.setValue(data_set.hv)
        if not (data_set.wf is None):
            self.util_panel.axes_energy_wf.setValue(data_set.wf)
        if not (data_set.kyscale is None):
            self.k_axis = data_set.kyscale
            new_range = [self.k_axis[0], self.k_axis[-1]]
            self.main_plot.plotItem.getAxis(self.main_plot.main_yaxis).setRange(*new_range)
            self.plot_y.plotItem.getAxis(self.plot_y.main_xaxis).setRange(*new_range)
            self.util_panel.momentum_hor_value.setText('({:.4f})'.format(
                self.k_axis[self.main_plot.pos[1].get_value()]))

    def load_saved_corrections_old(self, data_set):

        if hasattr(data_set, 'saved'):
            saved = data_set.saved
            if 'Ef' in saved.keys():
                self.util_panel.axes_energy_Ef.setValue(saved['Ef'])
            if 'hv' in saved.keys():
                self.util_panel.axes_energy_hv.setValue(saved['hv'])
            if 'wf' in saved.keys():
                self.util_panel.axes_energy_wf.setValue(saved['wf'])
            if 'angle_off' in saved.keys():
                self.util_panel.axes_angle_off.setValue(saved['angle_off'])
            if 'k' in saved.keys():
                self.k_axis = saved['k']
                new_range = [self.k_axis[0], self.k_axis[-1]]
                self.main_plot.plotItem.getAxis(self.main_plot.main_yaxis).setRange(*new_range)
                self.plot_y.plotItem.getAxis(self.plot_y.main_xaxis).setRange(*new_range)
                self.util_panel.momentum_hor_value.setText('({:.4f})'.format(
                    self.k_axis[self.main_plot.pos[1].get_value()]))
            if 'k_axis' in saved.keys():
                self.k_axis = saved['k_axis']
                new_range = [self.k_axis[0], self.k_axis[-1]]
                self.main_plot.plotItem.getAxis(self.main_plot.main_yaxis).setRange(*new_range)
                self.plot_y.plotItem.getAxis(self.plot_y.main_xaxis).setRange(*new_range)
                self.util_panel.momentum_hor_value.setText('({:.4f})'.format(
                    self.k_axis[self.main_plot.pos[1].get_value()]))
        else:
            pass

        if not (data_set.Ef is None):
            self.util_panel.axes_energy_Ef.setValue(float(data_set.Ef))
        if not (data_set.hv is None):
            self.util_panel.axes_energy_hv.setValue(float(data_set.hv))
        if not (data_set.wf is None):
            self.util_panel.axes_energy_wf.setValue(float(data_set.wf))

    def open_pit(self) :
        """ Open the data in an instance of 
        :class:`data_slicer.pit.MainWindow`, which has the benefit of 
        providing a free-slicing ROI.
        """
        mw = pit.MainWindow()
        # Move the empty axis back
        data = np.moveaxis(self.data_set.data, 0, -1)
        mw.data_handler.set_data(data, axes=self.data_handler.axes)
        mw.set_cmap(self.cmap_name)

