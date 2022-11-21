""" General classes for data visualization objects
matplotlib pcolormesh equivalent in pyqtgraph (more or less) 
"""
import os
import subprocess

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QLabel, QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QLineEdit, \
    QMainWindow, QDialogButtonBox, QMessageBox, QScrollArea
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.graphicsItems.ImageItem import ImageItem

import piva.arpys_wp as wp
import piva.data_loader as dl
from piva.cmaps import cmaps, my_cmaps

BASE_LINECOLOR = (255, 255, 0, 255)
BINLINES_LINECOLOR = (168, 168, 104, 255)
ORIENTLINES_LINECOLOR = (164, 37, 22, 255)
HOVER_COLOR = (195, 155, 0, 255)
BGR_COLOR = (64, 64, 64)
util_panel_style = """
QFrame{margin:5px; border:1px solid rgb(150,150,150);}
QLabel{color: rgb(246, 246, 246); border:1px solid rgb(64, 64, 64);}
QCheckBox{color: rgb(246, 246, 246);}
"""
SIGNALS = 5
MY_CMAPS = True
DEFAULT_CMAP = 'coolwarm'

bold_font = QtGui.QFont()
bold_font.setBold(True)


class Crosshair:
    """ Crosshair made up of two InfiniteLines. """
    def __init__(self, image_plot, pos=(0, 0), mainplot=True, orientation='horizontal'):
        self.image_plot = image_plot
        self.orientation = orientation

        # Store the positions in TracedVariables
        self.hpos = TracedVariable(pos[1], name='hpos')
        self.vpos = TracedVariable(pos[0], name='vpos')

        # Initialize the InfiniteLines
        if orientation == 'horizontal':
            self.hline = pg.InfiniteLine(pos[1], movable=True, angle=0)
            self.vline = pg.InfiniteLine(pos[0], movable=True, angle=90)
        elif orientation == 'vertical':
            self.hline = pg.InfiniteLine(pos[1], movable=True, angle=90)
            self.vline = pg.InfiniteLine(pos[0], movable=True, angle=0)

        # Set the color
        self.set_color(BASE_LINECOLOR, HOVER_COLOR)

        # Register some callbacks depending on plot type
        if mainplot:
            self.hpos.sig_value_changed.connect(self.update_position_h)
            self.vpos.sig_value_changed.connect(self.update_position_v)

            self.hline.sigDragged.connect(self.on_dragged_h)
            self.vline.sigDragged.connect(self.on_dragged_v)
        else:
            self.hpos.sig_value_changed.connect(self.update_position_h)
            self.hline.sigDragged.connect(self.on_dragged_h)

    def add_to(self, widget):
        """ Add this crosshair to a Qt widget. """
        for line in [self.hline, self.vline]:
            line.setZValue(1)
            widget.addItem(line)

    def remove_from(self, widget):
        """ Remove this crosshair from a pyqtgraph widget. """
        for line in [self.hline, self.vline]:
            widget.removeItem(line)

    def set_color(self, linecolor=BASE_LINECOLOR, hover_color=HOVER_COLOR):
        """ Set the color and hover color of both InfiniteLines that make up
        the crosshair. The arguments can be any pyqtgraph compatible color
        specifiers.
        """
        for line in [self.hline, self.vline]:
            line.setPen(linecolor)
            line.setHoverPen(hover_color)

    def set_movable(self, movable=True):
        """ Set whether or not this crosshair can be dragged by the mouse. """
        for line in [self.hline, self.vline]:
            line.setMovable = movable

    def move_to(self, pos):
        """
        **Parameters**

        ===  ===================================================================
        pos  2-tuple; x and y coordinates of the desired location of the
             crosshair in data coordinates.
        ===  ===================================================================
        """
        self.hpos.set_value(pos[1])
        self.vpos.set_value(pos[0])

    def update_position_h(self):
        """ Callback for the :signal:`sig_value_changed
        <data_slicer.utilities.TracedVariable.sig_value_changed>`. Whenever the
        value of this TracedVariable is updated (possibly from outside this
        Crosshair object), put the crosshair to the correct position.
        """
        self.hline.setValue(self.hpos.get_value())

    def update_position_v(self):
        """ Confer update_position_v. """
        self.vline.setValue(self.vpos.get_value())

    def on_dragged_h(self):
        """ Callback for dragging of InfiniteLines. Their visual position
        should be reflected in the TracedVariables self.hpos and self.vpos.
        """
        self.hpos.set_value(self.hline.value())
        # if it's an energy plot, and binning option is active, update also binning boundaries
        if self.image_plot.binning:
            pos = self.hline.value()
            self.image_plot.left_line.setValue(pos - self.image_plot.width)
            self.image_plot.right_line.setValue(pos + self.image_plot.width)

    def on_dragged_v(self):
        """ Callback for dragging of InfiniteLines. Their visual position
        should be reflected in the TracedVariables self.hpos and self.vpos.
        """
        self.vpos.set_value(self.vline.value())

    def set_bounds(self, xmin, xmax, ymin, ymax):
        """ Set the area in which the infinitelines can be dragged. """
        if self.orientation == 'horizontal':
            self.hline.setBounds([ymin, ymax])
            self.vline.setBounds([xmin, xmax])
        else:
            self.vline.setBounds([ymin, ymax])
            self.hline.setBounds([xmin, xmax])

    # def set_for_main_plot(self, pos):
    #     """
    #     Set crosshair for a main plot
    #     :param pos:         list; positions of horizontal and vertical crosshairs
    #     """
    #     # Store the positions in TracedVariables
    #     self.hpos = TracedVariable(pos[1], name='hpos')
    #     self.vpos = TracedVariable(pos[0], name='vpos')
    #
    #     # Register some callbacks
    #     self.hpos.sig_value_changed.connect(self.update_position_h)
    #     self.vpos.sig_value_changed.connect(self.update_position_v)
    #
    #     self.hline.sigDragged.connect(self.on_dragged_h)
    #     self.vline.sigDragged.connect(self.on_dragged_v)


class ImagePlot(pg.PlotWidget):
    """
    A PlotWidget which mostly contains a single 2D image (intensity 
    distribution) or a 3D array (distribution of RGB values) as well as all 
    the nice pyqtgraph axes panning/rescaling/zooming functionality.

    In addition, this allows one to use custom axes scales as opposed to 
    being limited to pixel coordinates.

    **Signals**

    =================  =========================================================
    sig_image_changed  emitted whenever the image is updated
    sig_axes_changed   emitted when the axes are updated
    sig_clicked        emitted when user clicks inside the imageplot
    =================  =========================================================
    """
    sig_image_changed = QtCore.Signal()
    sig_axes_changed = QtCore.Signal()
    sig_clicked = QtCore.Signal(object)

    def __init__(self, image=None, parent=None, background=BGR_COLOR, name=None, mainplot=True,
                 orientation='horizontal', **kwargs):
        """ Allows setting of the image upon initialization. 
        
        **Parameters**

        ==========  ============================================================
        image       np.ndarray or pyqtgraph.ImageItem instance; the image to be
                    displayed.
        parent      QtWidget instance; parent widget of this widget.
        background  str; confer PyQt documentation
        name        str; allows giving a name for debug purposes
        ==========  ============================================================
        """
        # Initialize instance variables
        # np.array, raw image data
        self.image_data = None
        # pg.ImageItem of *image_data*
        self.image_item = None
        self.image_kwargs = {}
        self.xlim = None
        self.xlim_rescaled = None
        self.ylim = None
        self.ylim_rescaled = None
        self.xscale = None
        self.xscale_rescaled = None
        self.yscale = None
        self.yscale_rescaled = None
        self.transform_factors = []
        self.crosshair_cursor_visible = False
        self.binning = False
        self.orientation = orientation
        self.pmeshitem = False
        self.transposed = TracedVariable(False, name='transposed')

        super().__init__(parent=parent, background=background, **kwargs)

        self.name = name

        self.orientate()

        if image is not None:
            self.set_image(image)

        # Initiliaze a crosshair and add it to this widget
        self.crosshair = Crosshair(self, mainplot=mainplot, orientation=orientation)
        self.crosshair.add_to(self)

        self.pos = (self.crosshair.vpos, self.crosshair.hpos)

        # Initialize range to [0, 1]x[0, 1]
        self.set_bounds(0, 1, 0, 1)

        # Disable mouse scrolling, panning and zooming for both axes
        self.setMouseEnabled(False, False)

        # Connect a slot (callback) to dragging and clicking events
        self.sig_axes_changed.connect(
            lambda: self.set_bounds(*[x for lst in self.get_limits() for x in lst]))

        self.sig_image_changed.connect(self.update_allowed_values)

    # methods added to make crosshairs work
    def update_allowed_values(self):
        """ Update the allowed values silently.
        This assumes that the displayed image is in pixel coordinates and
        sets the allowed values to the available pixels.
        """
        [[xmin, xmax], [ymin, ymax]] = self.get_limits()
        if self.orientation == 'horizontal':
            self.pos[0].set_allowed_values(np.arange(xmin, xmax + 1, 1))
            self.pos[1].set_allowed_values(np.arange(ymin, ymax + 1, 1))
        else:
            self.pos[1].set_allowed_values(np.arange(xmin, xmax + 1, 1))
            self.pos[0].set_allowed_values(np.arange(ymin, ymax + 1, 1))

    def set_bounds(self, xmin, xmax, ymin, ymax):
        """ Set both, the displayed area of the axis as well as the the range
        in which the crosshair can be dragged to the intervals [xmin, xmax]
        and [ymin, ymax].
        """
        self.setXRange(xmin, xmax, padding=0.0)
        self.setYRange(ymin, ymax, padding=0.0)

        self.crosshair.set_bounds(xmin, xmax, ymin, ymax)

    def orientate(self):
        """
        Configure plot's layout depending on an orientation
        """
        if self.orientation == 'horizontal':
            # Show top and tight axes by default, but without ticklabels
            self.showAxis('top')
            self.showAxis('right')
            self.getAxis('top').setStyle(showValues=False)
            self.getAxis('right').setStyle(showValues=False)
            self.main_xaxis = 'bottom'
            self.main_xaxis_grid = (255, 1)
            self.main_yaxis = 'left'
            self.main_yaxis_grid = (2, 0)

            # moved here to get rid of warnings:
            self.right_axis = 'top'
            self.secondary_axis = 'right'
            self.secondary_axis_grid = (2, 2)
            self.angle = 0
            self.slider_axis_index = 1
        elif self.orientation == 'vertical':
            # Show top and tight axes by default, but without ticklabels
            self.showAxis('right')
            self.showAxis('top')
            self.getAxis('right').setStyle(showValues=False)
            self.getAxis('top').setStyle(showValues=False)
            self.main_xaxis = 'left'
            self.main_xaxis_grid = (255, 1)
            self.main_yaxis = 'bottom'
            self.main_yaxis_grid = (2, 0)

            # moved here to get rid of warnings:
            self.right_axis = 'right'
            self.secondary_axis = 'top'
            self.secondary_axis_grid = (2, 2)
            self.angle = 90
            self.slider_axis_index = 1

    def remove_image(self):
        """ Removes the current image using the parent's :meth:`removeItem 
        pyqtgraph.PlotWidget.removeItem` function. 
        """
        if self.image_item is not None:
            self.removeItem(self.image_item)
        self.image_item = None

    def set_image(self, image, pmesh=False, pmesh_x=None, pmesh_y=None, emit=True, *args, **kwargs):
        """ Expects either np.arrays or pg.ImageItems as input and sets them 
        correctly to this PlotWidget's Image with `addItem`. Also makes sure 
        there is only one Image by deleting the previous image.

        Emits :signal:`sig_image_changed`

        **Parameters**

        ========  ==============================================================
        image     np.ndarray or pyqtgraph.ImageItem instance; the image to be
                  displayed.
        emit      bool; whether or not to emit :signal:`sig_image_changed`
        (kw)args  positional and keyword arguments that are passed on to 
                  :class:`pyqtgraph.ImageItem`
        ========  ==============================================================
        """
        # Convert array to ImageItem
        if isinstance(image, np.ndarray):
            if 0 not in image.shape:
                if pmesh:
                    try:
                        print(pmesh_x.shape, pmesh_y.shape, image.shape)
                        image_item = pg.PColorMeshItem(pmesh_x, pmesh_y, image)
                        # image_item = PColorMeshItem(image, **kwargs)
                    except AttributeError:
                        print('unable to use pcolormesh object')
                        image_item = ImageItem(image, *args, **kwargs)
                else:
                    image_item = ImageItem(image, *args, **kwargs)
            else:
                return
        else:
            image_item = image

        # Transpose if necessary
        if self.transposed.get_value():
            image_item = ImageItem(image_item.image.T, *args, **kwargs)

        # Replace the image
        self.remove_image()
        self.image_item = image_item
        self.image_data = image
        self.addItem(image_item)
        # Reset limits if necessary
        if self.xscale is not None and self.yscale is not None:
            axes_shape = (len(self.xscale), len(self.yscale))
            if axes_shape != self.image_data.shape:
                self.xlim = None
                self.ylim = None
        self._set_axes_scales(emit=emit)

        if emit:
            self.sig_image_changed.emit()

    def set_xscale(self, xscale, update=False):
        """ Set the xscale of the plot. *xscale* is an array of the length 
        ``len(self.image_item.shape[0])``.
        """
        if self.orientation == 'vertical':
            self._set_yscale(xscale, update)
        else:
            self._set_xscale(xscale, update)

    def set_yscale(self, yscale, update=False):
        """ Set the yscale of the plot. *yscale* is an array of the length 
        ``len(self.image_item.image.shape[1])``.
        """
        if self.orientation == 'vertical':
            self._set_xscale(yscale, update)
        else:
            self._set_yscale(yscale, update)

    def _set_xscale(self, xscale, update=False, force=False):
        """ Set the scale of the horizontal axis of the plot. *force* can be 
        used to bypass the length checking.
        """
        # Sanity check
        if self.orientation == 'horizontal':
            if not force and self.image_item is not None and (len(xscale) != self.image_data.shape[0]):
                raise TypeError('Shape of xscale does not match data dimensions.')
        else:
            if not force and self.image_item is not None and (len(xscale) != self.image_data.shape[1]):
                raise TypeError('Shape of xscale does not match data dimensions.')

        self.xscale = xscale
        # 'Autoscale' the image to the xscale
        self.xlim = (xscale[0], xscale[-1])

        if update:
            self._set_axes_scales(emit=True)

    def _set_yscale(self, yscale, update=False, force=False):
        """ Set the scale of the vertical axis of the plot. *force* can be 
        used to bypass the length checking.
        """
        # Sanity check
        if self.orientation == 'horizontal':
            if not force and self.image_item is not None and (len(yscale) != self.image_data.shape[1]):
                raise TypeError('Shape of yscale does not match data dimensions.')
        else:
            if not force and self.image_item is not None and (len(yscale) != self.image_data.shape[0]):
                raise TypeError('Shape of yscale does not match data dimensions.')

        self.yscale = yscale
        # 'Autoscale' the image to the xscale
        self.ylim = (yscale[0], yscale[-1])

        if update:
            self._set_axes_scales(emit=True)

    def transpose(self):
        """ Transpose the image, i.e. swap the x- and y-axes. """
        self.transposed.set_value(not self.transposed.get_value())
        # Swap the scales
        new_xscale = self.yscale
        new_yscale = self.xscale
        self._set_xscale(new_xscale, force=True)
        self._set_yscale(new_yscale, force=True)
        # Update the image
        if not self.transposed.get_value():
            # Take care of the back-transposition here
            self.set_image(self.image_item.image.T, lut=self.image_item.lut)
        else:
            self.set_image(self.image_item, lut=self.image_item.lut)

    def set_xlabel(self, label):
        """ Shorthand for setting this plot's x axis label. """
        axis = self.getAxis('bottom')
        axis.setLabel(label)

    def set_ticks(self, min_val, max_val, axis):
        """
        Set customized to reflect the dimensions of the physical data
        :param min_val:     float; first axis' value
        :param max_val:     float; last axis' value
        :param axis:        str; axis of which ticks should be put
        """
        plotItem = self.plotItem

        # Remove the old top-axis
        plotItem.layout.removeItem(plotItem.getAxis(axis))
        # Create the new axis and set its range
        new_axis = pg.AxisItem(orientation=axis)
        new_axis.setRange(min_val, max_val)
        # Attach it internally to the plotItem and its layout (The arguments
        # `*(1, 1)` or `*(2, 2)` refers to the axis' position in the GridLayout)
        plotItem.axes[axis]['item'] = new_axis
        if axis == 'bottom':
            plotItem.layout.addItem(new_axis, *self.main_xaxis_grid)
        else:
            plotItem.layout.addItem(new_axis, *self.main_yaxis_grid)

    def set_ylabel(self, label):
        """ Shorthand for setting this plot's y axis label. """
        axis = self.getAxis('left')
        axis.setLabel(label)

    def _set_axes_scales(self, emit=False):
        """ Transform the image such that it matches the desired x and y 
        scales.
        """
        # Get image dimensions and requested origin (x0,y0) and top right corner (x1, y1)
        nx, ny = self.image_data.shape
        [[x0, x1], [y0, y1]] = self.get_limits()
        # Calculate the scaling factors
        sx = (x1 - x0) / nx
        sy = (y1 - y0) / ny
        # Ensure nonzero
        sx = 1 if sx == 0 else sx
        sy = 1 if sy == 0 else sy
        # Define a transformation matrix that scales and translates the image 
        # such that it appears at the coordinates that match our x and y axes.
        transform = QtGui.QTransform()
        transform.scale(sx, sy)
        # Carry out the translation in scaled coordinates
        transform.translate(x0/sx, y0/sy)
        # Finally, apply the transformation to the imageItem
        self.image_item.setTransform(transform)
        self._update_transform_factors()

        if emit:
            self.sig_axes_changed.emit()

    def set_secondary_axis(self, min_val, max_val):
        """ Create (or replace) a second x-axis on the top which ranges from
        `min_val` to `max_val`.
        This is the right axis in case of the horizontal orientation.
        """
        # Get a handle on the underlying plotItem
        plotItem = self.plotItem

        # Remove the old top-axis
        plotItem.layout.removeItem(plotItem.getAxis(self.secondary_axis))
        # Create the new axis and set its range
        new_axis = pg.AxisItem(orientation=self.secondary_axis)
        new_axis.setRange(min_val, max_val)
        # Attach it internally to the plotItem and its layout (The arguments
        # `*(1, 1)` or `*(2, 2)` refers to the axis' position in the GridLayout)
        plotItem.axes[self.secondary_axis]['item'] = new_axis
        plotItem.layout.addItem(new_axis, *self.secondary_axis_grid)

    def get_limits(self):
        """
        Get limits of the image data.
        :return:    list; [[x_min, x_max], [y_min, y_max]]
        """
        # Default to current viewrange but try to get more accurate values if possible
        if self.image_item is not None:
            x, y = self.image_data.shape
        else:
            x, y = 1, 1

        # Set the limits to image pixels if they are not defined
        if self.xlim is None or self.ylim is None:
            self.set_xscale(np.arange(0, x))
            self.set_yscale(np.arange(0, y))

        if self.orientation == 'horizontal':
            x_min, x_max = self.xlim
            y_min, y_max = self.ylim
        else:
            x_min, x_max = self.ylim
            y_min, y_max = self.xlim

        return [[x_min, x_max], [y_min, y_max]]

    def fix_viewrange(self):
        """ Prevent zooming out by fixing the limits of the ViewBox. """
        [[x_min, x_max], [y_min, y_max]] = self.get_limits()
        self.setLimits(xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max, maxXRange=x_max-x_min, maxYRange=y_max-y_min)

    def _update_transform_factors(self):
        """ Create a copy of the parameters that are necessary to reproduce 
        the current transform. This is necessary e.g. for the calculation of 
        the transform in :meth:`rotate 
        <data_slicer.imageplot.ImagePlot.rotate>`.
        """
        transform = self.image_item.transform()
        dx = transform.dx()
        dy = transform.dy()
        sx = transform.m11()
        sy = transform.m22()
        wx = self.image_item.width()
        wy = self.image_item.height()
        self.transform_factors = [dx, dy, sx, sy, wx, wy]

    def add_binning_lines(self, pos, width, orientation='horizontal'):
        """
        Add unmovable lines to an ImagePlot around a specified crosshair.
        The lines indicate integration area fro a respective cut
        :param pos:             int; position of the crosshair
        :param width:           int; number of left and right steps from the crosshair
                                    position
        :param orientation:     str; orientation of the crosshair
        """
        # delete binning lines if exist
        if orientation == 'horizontal':
            try:
                self.removeItem(self.left_hor_line)
                self.removeItem(self.right_hor_line)
            except AttributeError:
                pass
            # add new binning lines
            self.binning_hor = True
            self.hor_width = width
            self.left_hor_line = pg.InfiniteLine(pos - width, movable=False, angle=0)
            self.right_hor_line = pg.InfiniteLine(pos + width, movable=False, angle=0)
            self.left_hor_line.setPen(color=BINLINES_LINECOLOR, width=1)
            self.right_hor_line.setPen(color=BINLINES_LINECOLOR, width=1)
            self.addItem(self.left_hor_line)
            self.addItem(self.right_hor_line)
        elif orientation == 'vertical':
            try:
                self.removeItem(self.left_ver_line)
                self.removeItem(self.right_ver_line)
            except AttributeError:
                pass
            # add new binning lines
            self.binning_ver = True
            self.ver_width = width
            self.left_ver_line = pg.InfiniteLine(pos - width, movable=False, angle=90)
            self.right_ver_line = pg.InfiniteLine(pos + width, movable=False, angle=90)
            self.left_ver_line.setPen(color=BINLINES_LINECOLOR, width=1)
            self.right_ver_line.setPen(color=BINLINES_LINECOLOR, width=1)
            self.addItem(self.left_ver_line)
            self.addItem(self.right_ver_line)

    def remove_binning_lines(self, orientation='horizontal'):
        """
        Remove unmovable binning lines from an ImagePlot.
        :param orientation:
        :return:
        """
        if orientation == 'horizontal':
            self.binning_hor = False
            self.hor_width = 0
            self.removeItem(self.left_hor_line)
            self.removeItem(self.right_hor_line)
        elif orientation == 'vertical':
            self.binning_ver = False
            self.ver_width = 0
            self.removeItem(self.left_ver_line)
            self.removeItem(self.right_ver_line)

    def register_momentum_slider(self, traced_variable):
        """ Set self.pos to the given TracedVariable instance and connect the
        relevant slots to the signals. This can be used to share a
        TracedVariable among widgets.
        """
        self.crosshair.vpos = traced_variable
        self.crosshair.vpos.sig_value_changed.connect(self.set_position)
        self.crosshair.vpos.sig_allowed_values_changed.connect(self.on_allowed_values_change)

    def set_position(self):
        """ Callback for the :signal:`sig_value_changed
        <data_slicer.utilities.TracedVariable.sig_value_changed>`. Whenever the
        value of this TracedVariable is updated (possibly from outside this
        Scalebar object), put the slider to the correct position.
        """
        if self.orientation == 'horizontal':
            new_pos = self.crosshair.vpos.get_value()
            self.crosshair.vline.setValue(new_pos)
        elif self.orientation == 'vertical':
            new_pos = self.crosshair.vpos.get_value()
            self.crosshair.hline.setValue(new_pos)

    def on_allowed_values_change(self):
        """ Callback for the :signal:`sig_allowed_values_changed
        <data_slicer.utilities.TracedVariable.sig_allowed_values_changed>`.
        With a change of the allowed values in the TracedVariable, we should
        update our bounds accordingly.
        The number of allowed values can also give us a hint for a reasonable
        maximal width for the slider.
        """
        # If the allowed values were reset, just exit
        if self.crosshair.vpos.allowed_values is None:
            return

        lower = self.crosshair.vpos.min_allowed# - self.width
        upper = self.crosshair.vpos.max_allowed# + self.width
        self.set_momentum_slider_bounds(lower, upper)

    def set_momentum_slider_bounds(self, xmin, xmax):
        """
        Set bounds of the vertical crosshair.
        """
        self.crosshair.vline.setBounds([xmin, xmax])


class CursorPlot(pg.PlotWidget):
    """ Implements a simple, draggable scalebar represented by a line
    (:class:`pyqtgraph.InfiniteLine) on an axis
    (:class:`pyqtgraph.PlotWidget).
    The current position of the slider is tracked with the
    :class:`TracedVariable <data_slicer.utilities.TracedVariable>` self.pos
    and its width with the `TracedVariable` self.slider_width.
    """
    hover_color = HOVER_COLOR

    def __init__(self, parent=None, background=BGR_COLOR, name=None,
                 orientation='horizontal', slider_width=1, z_plot=False, **kwargs):
        """ Initialize the slider and set up the visual tweaks to make a
        PlotWidget look more like a scalebar.

        **Parameters**

        ===========  ============================================================
        parent       QtWidget instance; parent widget of this widget
        background   str; confer PyQt documentation
        name         str; allows giving a name for debug purposes
        orientation  str, `horizontal` or `vertical`; orientation of the cursor
        ===========  ============================================================
        """
        super().__init__(parent=parent, background=background, **kwargs)

        # moved here to get rid of warnings:
        self.right_axis = 'top'
        self.left_axis = 'bottom'
        self.secondary_axis = 'right'
        self.main_xaxis = 'left'
        self.main_xaxis_grid = (2, 0)
        self.secondary_axis_grid = (2, 2)
        self.angle = 0
        self.slider_axis_index = 1
        self.pos = None
        self.left_line = None
        self.right_line = None
        self.width = 0
        self.wheel_frames = None
        self.cursor_color = None
        self.pen_width = None
        self.sp_EDC_pen = pg.mkPen('r')

        # Whether to allow changing the slider width with arrow keys
        self.change_width_enabled = False

        if orientation not in ['horizontal', 'vertical']:
            raise ValueError('Only `horizontal` and `vertical` orientations are allowed.')
        self.orientation = orientation
        self.orientate()
        self.binning = False
        self.z_plot = z_plot

        if name is not None:
            self.name = name
        else:
            self.name = 'Unnamed'

        # Hide the pyqtgraph auto-rescale button
        self.getPlotItem().buttonsHidden = True

        # Display the right (or top) axis without ticklabels
        self.showAxis(self.right_axis)
        self.getAxis(self.right_axis).setStyle(showValues=False)

        # The position of the slider is stored with a TracedVariable
        initial_pos = 0
        pos = TracedVariable(initial_pos, name='pos')
        self.register_traced_variable(pos)

        # Set up the slider
        self.slider_width = TracedVariable(slider_width, name='{}.slider_width'.format(self.name))
        self.slider = pg.InfiniteLine(initial_pos, movable=True, angle=self.angle)
        self.set_slider_pen(color=(255, 255, 0, 255), width=slider_width)

        # Add a marker. Args are (style, position (from 0-1), size #NOTE
        # seems broken
        self.addItem(self.slider)

        # Disable mouse scrolling, panning and zooming for both axes
        self.setMouseEnabled(False, False)

        # Initialize range to [0, 1]
        self.set_bounds(initial_pos, initial_pos + 1)

        # Connect a slot (callback) to dragging and clicking events
        self.slider.sigDragged.connect(self.on_position_change)

    def get_data(self):
        """ Get the currently displayed data as a tuple of arrays, one
        containing the x values and the other the y values.

        **Returns**

        =  =====================================================================
        x  array containing the x values.
        y  array containing the y values.
        =  =====================================================================
        """
        pdi = self.listDataItems()[0]
        return pdi.getData()

    def orientate(self):
        """ Define all aspects that are dependent on the orientation. """
        if self.orientation == 'horizontal':
            self.right_axis = 'right'
            self.left_axis = 'left'
            self.secondary_axis = 'top'
            self.main_xaxis = 'bottom'
            self.main_xaxis_grid = (255, 1)
            self.secondary_axis_grid = (1, 1)
            self.angle = 90
            self.slider_axis_index = 1
        elif self.orientation == 'vertical':
            self.right_axis = 'top'
            self.left_axis = 'bottom'
            self.secondary_axis = 'right'
            self.main_xaxis = 'left'
            self.main_xaxis_grid = (2, 0)
            self.secondary_axis_grid = (2, 2)
            self.angle = 0
            self.slider_axis_index = 1

    def register_traced_variable(self, traced_variable):
        """ Set self.pos to the given TracedVariable instance and connect the
        relevant slots to the signals. This can be used to share a
        TracedVariable among widgets.
        """
        self.pos = traced_variable
        self.pos.sig_value_changed.connect(self.set_position)
        self.pos.sig_allowed_values_changed.connect(self.on_allowed_values_change)

    def on_position_change(self):
        """ Callback for the :signal:`sigDragged
        <pyqtgraph.InfiniteLine.sigDragged>`. Set the value of the
        TracedVariable instance self.pos to the current slider position.
        """
        current_pos = self.slider.value()
        # NOTE pos.set_value emits signal sig_value_changed which may lead to
        # duplicate processing of the position change.
        self.pos.set_value(current_pos)

        # if it's an energy plot, and binning option is active, update also binning boundaries
        if self.binning and self.z_plot:
            z_pos = self.slider.value()
            self.left_line.setValue(z_pos - self.width)
            self.right_line.setValue(z_pos + self.width)

    def on_allowed_values_change(self):
        """ Callback for the :signal:`sig_allowed_values_changed
        <data_slicer.utilities.TracedVariable.sig_allowed_values_changed>`.
        With a change of the allowed values in the TracedVariable, we should
        update our bounds accordingly.
        The number of allowed values can also give us a hint for a reasonable
        maximal width for the slider.
        """
        # If the allowed values were reset, just exit
        if self.pos.allowed_values is None:
            return

        lower = self.pos.min_allowed - self.width
        upper = self.pos.max_allowed + self.width
        self.set_bounds(lower, upper)

    def set_position(self):
        """ Callback for the :signal:`sig_value_changed
        <data_slicer.utilities.TracedVariable.sig_value_changed>`. Whenever the
        value of this TracedVariable is updated (possibly from outside this
        Scalebar object), put the slider to the appropriate position.
        """
        new_pos = self.pos.get_value()
        self.slider.setValue(new_pos)

    def add_binning_lines(self, pos, width):
        """ Callback for the :signal:`stateChanged and valueChanged. Called whenever the
        binning checkBox is set to True or number of bins changes.
        """
        # delete binning lines if exist
        try:
            self.removeItem(self.left_line)
            self.removeItem(self.right_line)
        except AttributeError:
            pass
        # add new binning lines
        self.binning = True
        self.width = width
        self.left_line = pg.InfiniteLine(pos-width, movable=False, angle=self.angle)
        self.right_line = pg.InfiniteLine(pos+width, movable=False, angle=self.angle)
        self.left_line.setPen(color=BINLINES_LINECOLOR, width=1)
        self.right_line.setPen(color=BINLINES_LINECOLOR, width=1)
        self.addItem(self.left_line)
        self.addItem(self.right_line)

    def remove_binning_lines(self):
        """ Callback for the :signal:`stateChanged. Called whenever the
        binning checkBox is set to False.
        """
        self.binning = False
        self.width = 0
        self.removeItem(self.left_line)
        self.removeItem(self.right_line)

    def set_bounds(self, lower, upper):
        """ Set both, the displayed area of the axis as well as the the range
        in which the slider (InfiniteLine) can be dragged to the interval
        [lower, upper].
        """
        if self.orientation == 'horizontal':
            self.setXRange(lower, upper, padding=0.0)
        else:
            self.setYRange(lower, upper, padding=0.0)
        self.slider.setBounds([lower, upper])

    def set_secondary_axis(self, min_val, max_val):
        """ Create (or replace) a second x-axis on the top which ranges from
        `min_val` to `max_val`.
        This is the right axis in case of the horizontal orientation.
        """
        # Get a handle on the underlying plotItem
        plotItem = self.plotItem

        # Remove the old top-axis
        plotItem.layout.removeItem(plotItem.getAxis(self.secondary_axis))
        # Create the new axis and set its range
        new_axis = pg.AxisItem(orientation=self.secondary_axis)
        new_axis.setRange(min_val, max_val)
        # Attach it internally to the plotItem and its layout (The arguments
        # `*(1, 1)` or `*(2, 2)` refers to the axis' position in the GridLayout)
        plotItem.axes[self.secondary_axis]['item'] = new_axis
        plotItem.layout.addItem(new_axis, *self.secondary_axis_grid)

    def set_ticks(self, min_val, max_val, axis):
        plotItem = self.plotItem

        # Remove the old top-axis
        plotItem.layout.removeItem(plotItem.getAxis(axis))
        # Create the new axis and set its range
        new_axis = pg.AxisItem(orientation=axis)
        new_axis.setRange(min_val, max_val)
        # Attach it internally to the plotItem and its layout (The arguments
        # `*(1, 1)` or `*(2, 2)` refers to the axis' position in the GridLayout)
        plotItem.axes[axis]['item'] = new_axis
        plotItem.layout.addItem(new_axis, *self.main_xaxis_grid)

    def set_slider_pen(self, color=None, width=None, hover_color=None):
        """ Define the color and thickness of the slider (InfiniteLine
        object :class:`pyqtgraph.InfiniteLine`) and store these attribute
        in `self.slider_width` and `self.cursor_color`).
        """
        # Default to the current values if none are given
        if color is None:
            color = self.cursor_color
        else:
            self.cursor_color = color

        if width is None:
            width = self.pen_width
        else:
            self.pen_width = width

        if hover_color is None:
            hover_color = self.hover_color
        else:
            self.hover_color = hover_color

        self.slider.setPen(color=color, width=width)
        # Keep the hoverPen-size consistent
        self.slider.setHoverPen(color=hover_color, width=width)


class UtilitiesPanel(QWidget):
    """ Utilities panel on the top. Mostly just creating and aligning the stuff, signals and callbacks are
    handled in 'MainWindow' """
    # TODO Fermi edge fitting
    # TODO k-space conversion
    # TODO ROI
    # TODO multiple plotting tool
    # TODO logbook!
    # TODO curvature along specific directions
    # TODO smoothing along specific directions
    # TODO colorscale

    def __init__(self, main_window, name=None, dim=3):

        super().__init__()

        self.mw = main_window
        self.layout = QtWidgets.QGridLayout()
        self.tabs = QtWidgets.QTabWidget()
        self.tabs_visible = True
        self.dim = dim

        self.close_button = QPushButton('close')
        self.save_button = QPushButton('save')
        self.hide_button = QPushButton('hide tabs')
        self.pit_button = QPushButton('open in PIT')

        self.buttons = QWidget()
        self.buttons_layout = QtWidgets.QGridLayout()
        self.buttons_layout.addWidget(self.close_button,    1, 0)
        self.buttons_layout.addWidget(self.save_button,     2, 0)
        self.buttons_layout.addWidget(self.hide_button,     3, 0)
        self.buttons_layout.addWidget(self.pit_button,      4, 0)
        self.buttons.setLayout(self.buttons_layout)

        if name is not None:
            self.name = name
        else:
            self.name = 'Unnamed'

        self.initUI()

    def initUI(self):

        self.setStyleSheet(util_panel_style)
        momentum_labels_width = 80
        energy_labels_width = 80
        self.tabs_rows_span = 4
        self.tabs_cols_span = 9

        self.align()

        if self.dim == 2:
            self.energy_vert_value.setFixedWidth(energy_labels_width)
            self.momentum_hor_value.setFixedWidth(momentum_labels_width)
        elif self.dim == 3:
            self.energy_main_value.setFixedWidth(energy_labels_width)
            self.energy_hor_value.setFixedWidth(energy_labels_width)
            self.energy_vert_value.setFixedWidth(energy_labels_width)
            self.momentum_hor_value.setFixedWidth(momentum_labels_width)
            self.momentum_vert_value.setFixedWidth(momentum_labels_width)

        self.layout.addWidget(self.tabs,            0, 0, self.tabs_rows_span, self.tabs_cols_span)
        self.layout.addWidget(self.buttons,         0, self.tabs_cols_span + 1)
        self.setLayout(self.layout)

        # file options
        self.file_show_md_button.clicked.connect(self.show_metadata_window)
        self.file_add_md_button.clicked.connect(self.add_metadata)
        self.file_remove_md_button.clicked.connect(self.remove_metadata)
        self.file_sum_datasets_sum_button.clicked.connect(self.sum_datasets)
        self.file_sum_datasets_reset_button.clicked.connect(self.reset_summation)
        self.file_jn_button.clicked.connect(self.open_jupyter_notebook)

        # connect callbacks
        self.hide_button.clicked.connect(self.hidde_tabs)

        self.setup_cmaps()
        self.setup_gamma()
        self.setup_colorscale()
        self.setup_bin_z()

    def align(self):

        self.set_sliders_tab()
        self.set_image_tab()
        self.set_axes_tab()
        if self.dim == 3:
            self.set_orientate_tab()
        self.set_file_tab()

    def hidde_tabs(self):
        self.tabs_visible = not self.tabs_visible
        self.tabs.setVisible(self.tabs_visible)
        if self.tabs_visible:
            self.hide_button.setText('hide tabs')
        else:
            self.hide_button.setText('show tabs')

    def set_image_tab(self):

        max_w = 80
        # create elements
        self.image_tab = QWidget()
        itl = QtWidgets.QGridLayout()
        self.image_colors_label = QLabel('Colors')
        self.image_colors_label.setFont(bold_font)
        self.image_cmaps_label = QLabel('cmaps:')
        self.image_cmaps = QComboBox()
        self.image_invert_colors = QCheckBox('invert colors')
        self.image_gamma_label = QLabel('gamma:')
        self.image_gamma = QDoubleSpinBox()
        self.image_gamma.setRange(0.05, 10)
        self.image_colorscale_label = QLabel('color scale:')
        self.image_colorscale = QDoubleSpinBox()

        self.image_pmesh = QCheckBox('pmesh')

        self.image_other_lbl = QLabel('Normalize')
        self.image_other_lbl.setFont(bold_font)
        self.image_normalize_edcs = QCheckBox('normalize by each EDC')

        self.image_BZ_contour_lbl = QLabel('BZ contour')
        self.image_BZ_contour_lbl.setFont(bold_font)
        self.image_show_BZ = QCheckBox('show')
        self.image_symmetry_label = QLabel('symmetry:')
        self.image_symmetry = QSpinBox()
        self.image_symmetry.setRange(4, 6)
        self.image_rotate_BZ_label = QLabel('rotate:')
        self.image_rotate_BZ = QDoubleSpinBox()
        self.image_rotate_BZ.setRange(-90, 90)
        self.image_rotate_BZ.setSingleStep(0.5)

        self.image_2dv_lbl = QLabel('Open in 2D viewer')
        self.image_2dv_lbl.setFont(bold_font)
        self.image_2dv_cut_selector_lbl = QLabel('select cut')
        self.image_2dv_cut_selector = QComboBox()
        self.image_2dv_cut_selector.addItems(['vertical', 'horizontal'])
        self.image_2dv_button = QPushButton('Open')

        self.image_smooth_lbl = QLabel('Smooth')
        self.image_smooth_lbl.setFont(bold_font)
        self.image_smooth_n_lbl = QLabel('box size:')
        self.image_smooth_n = QSpinBox()
        self.image_smooth_n.setValue(3)
        self.image_smooth_n.setRange(3, 50)
        self.image_smooth_n.setMaximumWidth(max_w)
        self.image_smooth_rl_lbl = QLabel('recursion:')
        self.image_smooth_rl = QSpinBox()
        self.image_smooth_rl.setValue(3)
        self.image_smooth_rl.setRange(1, 20)
        self.image_smooth_button = QPushButton('Smooth')

        self.image_curvature_lbl = QLabel('Curvature')
        self.image_curvature_lbl.setFont(bold_font)
        self.image_curvature_method_lbl = QLabel('method:')
        self.image_curvature_method = QComboBox()
        curvature_methods = ['2D', '1D (EDC)', '1D (MDC)']
        self.image_curvature_method.addItems(curvature_methods)
        self.image_curvature_a_lbl = QLabel('a:')
        self.image_curvature_a = QDoubleSpinBox()
        self.image_curvature_a.setRange(0, 1e10)
        self.image_curvature_a.setSingleStep(0.0001)
        self.image_curvature_a.setValue(100.)
        self.image_curvature_a.setMaximumWidth(max_w)
        self.image_curvature_button = QPushButton('Do curvature')

        sd = 1
        # addWidget(widget, row, column, rowSpan, columnSpan)
        row = 0
        itl.addWidget(self.image_colors_label,          row * sd, 0)
        itl.addWidget(self.image_cmaps_label,           row * sd, 1)
        itl.addWidget(self.image_cmaps,                 row * sd, 2)
        itl.addWidget(self.image_invert_colors,         row * sd, 3)
        itl.addWidget(self.image_pmesh,                 row * sd, 4)

        row = 1
        itl.addWidget(self.image_gamma_label,           row * sd, 1)
        itl.addWidget(self.image_gamma,                 row * sd, 2)
        itl.addWidget(self.image_colorscale_label,      row * sd, 3)
        itl.addWidget(self.image_colorscale,            row * sd, 4)

        row = 2
        itl.addWidget(self.image_other_lbl,             row * sd, 0)
        itl.addWidget(self.image_normalize_edcs,        row * sd, 1, 1, 2)

        if self.dim == 2:
            row = 3
            itl.addWidget(self.image_smooth_lbl,        row * sd, 0)
            itl.addWidget(self.image_smooth_n_lbl,      row * sd, 1)
            itl.addWidget(self.image_smooth_n,          row * sd, 2)
            itl.addWidget(self.image_smooth_rl_lbl,     row * sd, 3)
            itl.addWidget(self.image_smooth_rl,         row * sd, 4)
            itl.addWidget(self.image_smooth_button,     row * sd, 5, 1, 2)

            row = 4
            itl.addWidget(self.image_curvature_lbl,         row * sd, 0)
            itl.addWidget(self.image_curvature_method_lbl,  row * sd, 1)
            itl.addWidget(self.image_curvature_method,      row * sd, 2)
            itl.addWidget(self.image_curvature_a_lbl,       row * sd, 3)
            itl.addWidget(self.image_curvature_a,           row * sd, 4)
            itl.addWidget(self.image_curvature_button,      row * sd, 5, 1, 2)

        if self.dim == 3:
            row = 3
            itl.addWidget(self.image_BZ_contour_lbl,    row * sd, 0)
            itl.addWidget(self.image_symmetry_label,    row * sd, 1)
            itl.addWidget(self.image_symmetry,          row * sd, 2)
            itl.addWidget(self.image_rotate_BZ_label,   row * sd, 3)
            itl.addWidget(self.image_rotate_BZ,         row * sd, 4)
            itl.addWidget(self.image_show_BZ,           row * sd, 5)

            row = 4
            itl.addWidget(self.image_2dv_lbl,               row, 0, 1, 2)
            itl.addWidget(self.image_2dv_cut_selector_lbl,  row, 2)
            itl.addWidget(self.image_2dv_cut_selector,      row, 3)
            itl.addWidget(self.image_2dv_button,            row, 4)

            # dummy item
            dummy_lbl = QLabel('')
            itl.addWidget(dummy_lbl, 5, 0, 1, 7)

        self.image_tab.layout = itl
        self.image_tab.setLayout(itl)
        self.tabs.addTab(self.image_tab, 'Image')

    def set_sliders_tab(self):

        self.sliders_tab = QWidget()
        vtl = QtWidgets.QGridLayout()
        max_lbl_w = 40
        bin_box_w = 50
        coords_box_w = 70

        if self.dim == 2:
            # binning option
            self.bins_label = QLabel('Integrate')
            self.bins_label.setFont(bold_font)
            self.bin_y = QCheckBox('bin EDCs')
            self.bin_y_nbins = QSpinBox()
            self.bin_z = QCheckBox('bin MDCs')
            self.bin_z_nbins = QSpinBox()

            # cross' hairs positions
            self.positions_momentum_label = QLabel('Momentum sliders')
            self.positions_momentum_label.setFont(bold_font)
            self.energy_vert_label = QLabel('E:')
            self.energy_vert = QSpinBox()
            self.energy_vert_value = QLabel('eV')
            self.momentum_hor_label = QLabel('kx:')
            self.momentum_hor = QSpinBox()
            self.momentum_hor_value = QLabel('deg')

            sd = 1
            # addWidget(widget, row, column, rowSpan, columnSpan)
            col = 0
            vtl.addWidget(self.bins_label,                0 * sd, col, 1, 3)
            vtl.addWidget(self.bin_y,                     1 * sd, col * sd)
            vtl.addWidget(self.bin_y_nbins,               1 * sd, (col+1) * sd)
            vtl.addWidget(self.bin_z,                     2 * sd, col * sd)
            vtl.addWidget(self.bin_z_nbins,               2 * sd, (col+1) * sd)

            col = 3
            vtl.addWidget(self.positions_momentum_label,  0 * sd, col, 1, 3)
            vtl.addWidget(self.energy_vert_label,         1 * sd, col)
            vtl.addWidget(self.energy_vert,               1 * sd, (col+1) * sd)
            vtl.addWidget(self.energy_vert_value,         1 * sd, (col+2) * sd)
            vtl.addWidget(self.momentum_hor_label,        2 * sd, col)
            vtl.addWidget(self.momentum_hor,              2 * sd, (col+1) * sd)
            vtl.addWidget(self.momentum_hor_value,        2 * sd, (col+2) * sd)

            # dummy lbl
            dummy_lbl = QLabel('')
            vtl.addWidget(dummy_lbl, 0, 6, 5, 2)

        elif self.dim == 3:
            # binning option
            self.bin_z = QCheckBox('bin E')
            self.bin_z_nbins = QSpinBox()
            self.bin_z_nbins.setMaximumWidth(bin_box_w)
            self.bin_x = QCheckBox('bin kx')
            self.bin_x_nbins = QSpinBox()
            self.bin_x_nbins.setMaximumWidth(bin_box_w)
            self.bin_y = QCheckBox('bin ky')
            self.bin_y_nbins = QSpinBox()
            self.bin_y_nbins.setMaximumWidth(bin_box_w)
            self.bin_zx = QCheckBox('bin E (kx)')
            self.bin_zx_nbins = QSpinBox()
            self.bin_zx_nbins.setMaximumWidth(bin_box_w)
            self.bin_zy = QCheckBox('bin E (ky)')
            self.bin_zy_nbins = QSpinBox()
            self.bin_zy_nbins.setMaximumWidth(bin_box_w)

            # cross' hairs positions
            self.positions_energies_label = QLabel('Energy sliders')
            self.positions_energies_label.setFont(bold_font)
            self.energy_main_label = QLabel('main:')
            self.energy_main_label.setMaximumWidth(max_lbl_w)
            self.energy_main = QSpinBox()
            self.energy_main.setMaximumWidth(coords_box_w)
            self.energy_main_value = QLabel('eV')
            self.energy_hor_label = QLabel('kx:')
            self.energy_hor_label.setMaximumWidth(max_lbl_w)
            self.energy_hor = QSpinBox()
            self.energy_hor.setMaximumWidth(coords_box_w)
            self.energy_hor_value = QLabel('eV')
            self.energy_vert_label = QLabel('ky:')
            self.energy_vert_label.setMaximumWidth(max_lbl_w)
            self.energy_vert = QSpinBox()
            self.energy_vert.setMaximumWidth(coords_box_w)
            self.energy_vert_value = QLabel('eV')

            self.positions_momentum_label = QLabel('Momentum sliders')
            self.positions_momentum_label.setFont(bold_font)
            self.momentum_hor_label = QLabel('ky:')
            self.momentum_hor_label.setMaximumWidth(max_lbl_w)
            self.momentum_hor = QSpinBox()
            self.momentum_hor.setMaximumWidth(coords_box_w)
            self.momentum_hor_value = QLabel('deg')
            self.momentum_vert_label = QLabel('kx:')
            self.momentum_vert_label.setMaximumWidth(max_lbl_w)
            self.momentum_vert = QSpinBox()
            self.momentum_vert.setMaximumWidth(coords_box_w)
            self.momentum_vert_value = QLabel('deg')

            sd = 1
            # addWidget(widget, row, column, rowSpan, columnSpan)
            col = 0
            vtl.addWidget(self.positions_energies_label, 0 * sd, col * sd, 1, 3)
            vtl.addWidget(self.energy_main_label, 1 * sd, col * sd)
            vtl.addWidget(self.energy_main, 1 * sd, (col + 1) * sd)
            vtl.addWidget(self.energy_main_value, 1 * sd, (col + 2) * sd)
            vtl.addWidget(self.energy_hor_label, 2 * sd, col * sd)
            vtl.addWidget(self.energy_hor, 2 * sd, (col + 1) * sd)
            vtl.addWidget(self.energy_hor_value, 2 * sd, (col + 2) * sd)
            vtl.addWidget(self.energy_vert_label, 3 * sd, col * sd)
            vtl.addWidget(self.energy_vert, 3 * sd, (col + 1) * sd)
            vtl.addWidget(self.energy_vert_value, 3 * sd, (col + 2) * sd)

            col = 3
            vtl.addWidget(self.positions_momentum_label, 0 * sd, col * sd, 1, 3)
            vtl.addWidget(self.momentum_vert_label, 1 * sd, col * sd)
            vtl.addWidget(self.momentum_vert, 1 * sd, (col + 1) * sd)
            vtl.addWidget(self.momentum_vert_value, 1 * sd, (col + 2) * sd)
            vtl.addWidget(self.momentum_hor_label, 2 * sd, col * sd)
            vtl.addWidget(self.momentum_hor, 2 * sd, (col + 1) * sd)
            vtl.addWidget(self.momentum_hor_value, 2 * sd, (col + 2) * sd)

            col = 6
            vtl.addWidget(self.bin_z, 0 * sd, col * sd)
            vtl.addWidget(self.bin_z_nbins, 0 * sd, (col + 1) * sd)
            vtl.addWidget(self.bin_x, 1 * sd, col * sd)
            vtl.addWidget(self.bin_x_nbins, 1 * sd, (col + 1) * sd)
            vtl.addWidget(self.bin_y, 2 * sd, col * sd)
            vtl.addWidget(self.bin_y_nbins, 2 * sd, (col + 1) * sd)
            vtl.addWidget(self.bin_zx, 3 * sd, col * sd)
            vtl.addWidget(self.bin_zx_nbins, 3 * sd, (col + 1) * sd)
            vtl.addWidget(self.bin_zy, 4 * sd, col * sd)
            vtl.addWidget(self.bin_zy_nbins, 4 * sd, (col + 1) * sd)

        self.sliders_tab.layout = vtl
        self.sliders_tab.setLayout(vtl)
        self.tabs.addTab(self.sliders_tab, 'Volume')

    def set_axes_tab(self):
        self.axes_tab = QWidget()
        atl = QtWidgets.QGridLayout()
        box_max_w = 100
        lbl_max_h = 30

        self.axes_energy_main_lbl = QLabel('Energy correction')
        self.axes_energy_main_lbl.setFont(bold_font)
        self.axes_energy_main_lbl.setMaximumHeight(lbl_max_h)
        self.axes_energy_Ef_lbl = QLabel('Ef (eV):')
        # self.axes_energy_Ef_lbl.setMaximumWidth(max_lbl_w)
        self.axes_energy_Ef = QDoubleSpinBox()
        self.axes_energy_Ef.setMaximumWidth(box_max_w)
        self.axes_energy_Ef.setRange(-5000., 5000)
        self.axes_energy_Ef.setDecimals(6)
        # self.axes_energy_Ef.setMinimumWidth(100)
        self.axes_energy_Ef.setSingleStep(0.001)

        self.axes_energy_hv_lbl = QLabel('h\u03BD (eV):')
        # self.axes_energy_hv_lbl.setMaximumWidth(max_w)
        self.axes_energy_hv = QDoubleSpinBox()
        self.axes_energy_hv.setMaximumWidth(box_max_w)
        self.axes_energy_hv.setRange(-2000., 2000)
        self.axes_energy_hv.setDecimals(4)
        self.axes_energy_hv.setSingleStep(0.001)

        self.axes_energy_wf_lbl = QLabel('wf (eV):')
        # self.axes_energy_wf_lbl.setMaximumWidth(max_w)
        self.axes_energy_wf = QDoubleSpinBox()
        self.axes_energy_wf.setMaximumWidth(box_max_w)
        self.axes_energy_wf.setRange(0, 5)
        self.axes_energy_wf.setDecimals(4)
        self.axes_energy_wf.setSingleStep(0.001)

        self.axes_energy_scale_lbl = QLabel('scale:')
        self.axes_energy_scale = QComboBox()
        self.axes_energy_scale.addItems(['binding', 'kinetic'])

        self.axes_momentum_main_lbl = QLabel('k-space conversion')
        self.axes_momentum_main_lbl.setFont(bold_font)
        self.axes_momentum_main_lbl.setMaximumHeight(lbl_max_h)
        self.axes_gamma_x_lbl = QLabel('\u0393 x0:')
        self.axes_gamma_x = QSpinBox()
        self.axes_gamma_x.setRange(0, 5000)

        self.axes_transform_kz = QCheckBox('Transform to kz')

        # self.axes_conv_hv_lbl = QLabel('h\u03BD (eV):')
        # self.axes_conv_hv = QDoubleSpinBox()
        # self.axes_conv_hv.setMaximumWidth(box_max_w)
        # self.axes_conv_hv.setRange(-2000., 2000.)
        # self.axes_conv_hv.setDecimals(3)
        # self.axes_conv_hv.setSingleStep(0.001)
        #
        # self.axes_conv_wf_lbl = QLabel('wf (eV):')
        # self.axes_conv_wf = QDoubleSpinBox()
        # self.axes_conv_wf.setMaximumWidth(box_max_w)
        # self.axes_conv_wf.setRange(0, 5)
        # self.axes_conv_wf.setDecimals(3)
        # self.axes_conv_wf.setSingleStep(0.001)

        self.axes_conv_lc_lbl = QLabel('a (\u212B):')
        self.axes_conv_lc = QDoubleSpinBox()
        self.axes_conv_lc.setMaximumWidth(box_max_w)
        self.axes_conv_lc.setRange(0, 10)
        self.axes_conv_lc.setDecimals(4)
        self.axes_conv_lc.setSingleStep(0.001)
        self.axes_conv_lc.setValue(3.1416)

        self.axes_conv_lc_op_lbl = QLabel('c (\u212B):')
        self.axes_conv_lc_op = QDoubleSpinBox()
        self.axes_conv_lc_op.setMaximumWidth(box_max_w)
        self.axes_conv_lc_op.setRange(0, 100)
        self.axes_conv_lc_op.setDecimals(4)
        self.axes_conv_lc_op.setSingleStep(0.001)
        self.axes_conv_lc_op.setValue(3.1416)

        self.axes_slit_orient_lbl = QLabel('Slit:')
        self.axes_slit_orient = QComboBox()
        self.axes_slit_orient.addItems(['horizontal', 'vertical', 'deflection'])
        self.axes_copy_values = QPushButton('Copy from \'Orientate\'')
        self.axes_do_kspace_conv = QPushButton('Convert')
        self.axes_reset_conv = QPushButton('Reset')

        if self.dim == 2:

            self.axes_angle_off_lbl = QLabel('angle offset:')
            self.axes_angle_off = QDoubleSpinBox()
            self.axes_angle_off.setMaximumWidth(box_max_w)
            self.axes_angle_off.setDecimals(4)
            self.axes_angle_off.setSingleStep(0.0001)

            sd = 1
            # addWidget(widget, row, column, rowSpan, columnSpan)
            row = 0
            atl.addWidget(self.axes_energy_main_lbl,    row * sd, 0 * sd, 1, 2)
            atl.addWidget(self.axes_energy_scale_lbl,   row * sd, 4 * sd)
            atl.addWidget(self.axes_energy_scale,       row * sd, 5 * sd)
            atl.addWidget(self.axes_energy_Ef_lbl,      (row + 1) * sd, 0 * sd)
            atl.addWidget(self.axes_energy_Ef,          (row + 1) * sd, 1 * sd)
            atl.addWidget(self.axes_energy_hv_lbl,      (row + 1) * sd, 2 * sd)
            atl.addWidget(self.axes_energy_hv,          (row + 1) * sd, 3 * sd)
            atl.addWidget(self.axes_energy_wf_lbl,      (row + 1) * sd, 4 * sd)
            atl.addWidget(self.axes_energy_wf,          (row + 1) * sd, 5 * sd)

            row = 2
            atl.addWidget(self.axes_momentum_main_lbl,  row * sd, 0 * sd, 1, 2)
            atl.addWidget(self.axes_gamma_x_lbl,        (row + 1) * sd, 0 * sd)
            atl.addWidget(self.axes_gamma_x,            (row + 1) * sd, 1 * sd)
            atl.addWidget(self.axes_angle_off_lbl,      (row + 1) * sd, 2 * sd)
            atl.addWidget(self.axes_angle_off,          (row + 1) * sd, 3 * sd)
            atl.addWidget(self.axes_conv_lc_lbl,        (row + 1) * sd, 4 * sd)
            atl.addWidget(self.axes_conv_lc,            (row + 1) * sd, 5 * sd)
            # atl.addWidget(self.axes_conv_hv_lbl,        (row + 1) * sd, 4 * sd)
            # atl.addWidget(self.axes_conv_hv,            (row + 1) * sd, 5 * sd)

            row = 4
            # atl.addWidget(self.axes_conv_wf_lbl,        row * sd, 0 * sd)
            # atl.addWidget(self.axes_conv_wf,            row * sd, 1 * sd)
            atl.addWidget(self.axes_slit_orient_lbl,    row * sd, 0 * sd)
            atl.addWidget(self.axes_slit_orient,        row * sd, 1 * sd)
            atl.addWidget(self.axes_do_kspace_conv,     row * sd, 2 * sd, 1, 2)
            atl.addWidget(self.axes_reset_conv,         row * sd, 4 * sd, 1, 2)

            # # dummy item
            # self.axes_massage_lbl = QLabel('')
            # atl.addWidget(self.axes_massage_lbl, 6, 0, 1, 9)

        elif self.dim == 3:

            self.axes_gamma_y_lbl = QLabel('\u0393 y0')
            self.axes_gamma_y = QSpinBox()
            self.axes_gamma_y.setRange(0, 5000)

            sd = 1
            # addWidget(widget, row, column, rowSpan, columnSpan)
            row = 0
            atl.addWidget(self.axes_energy_main_lbl,    row * sd, 0 * sd, 1, 2)
            atl.addWidget(self.axes_energy_scale_lbl,   row * sd, 4 * sd)
            atl.addWidget(self.axes_energy_scale,       row * sd, 5 * sd)
            atl.addWidget(self.axes_energy_Ef_lbl,      (row + 1) * sd, 0 * sd)
            atl.addWidget(self.axes_energy_Ef,          (row + 1) * sd, 1 * sd)
            atl.addWidget(self.axes_energy_hv_lbl,      (row + 1) * sd, 2 * sd)
            atl.addWidget(self.axes_energy_hv,          (row + 1) * sd, 3 * sd)
            atl.addWidget(self.axes_energy_wf_lbl,      (row + 1) * sd, 4 * sd)
            atl.addWidget(self.axes_energy_wf,          (row + 1) * sd, 5 * sd)

            row = 2
            atl.addWidget(self.axes_momentum_main_lbl,  row * sd, 0 * sd, 1, 2)
            atl.addWidget(self.axes_gamma_x_lbl,        (row + 1) * sd, 0 * sd)
            atl.addWidget(self.axes_gamma_x,            (row + 1) * sd, 1 * sd)
            atl.addWidget(self.axes_gamma_y_lbl,        (row + 1) * sd, 2 * sd)
            atl.addWidget(self.axes_gamma_y,            (row + 1) * sd, 3 * sd)
            atl.addWidget(self.axes_transform_kz,       (row + 1) * sd, 4 * sd, 1, 2)

            row = 4
            atl.addWidget(self.axes_conv_lc_lbl,        row * sd, 0 * sd)
            atl.addWidget(self.axes_conv_lc,            row * sd, 1 * sd)
            atl.addWidget(self.axes_conv_lc_op_lbl,     row * sd, 2 * sd)
            atl.addWidget(self.axes_conv_lc_op,         row * sd, 3 * sd)
            atl.addWidget(self.axes_slit_orient_lbl,    row * sd, 4 * sd)
            atl.addWidget(self.axes_slit_orient,        row * sd, 5 * sd)

            row = 5
            atl.addWidget(self.axes_copy_values,        row * sd, 0 * sd, 1, 2)
            atl.addWidget(self.axes_do_kspace_conv,     row * sd, 2 * sd, 1, 2)
            atl.addWidget(self.axes_reset_conv,         row * sd, 4 * sd, 1, 2)

            # # dummy item
            # self.axes_massage_lbl = QLabel('')
            # atl.addWidget(self.axes_massage_lbl, 5, 0, 1, 9)

        self.axes_tab.layout = atl
        self.axes_tab.setLayout(atl)
        self.tabs.addTab(self.axes_tab, 'Axes')

    def set_orientate_tab(self):

        self.orientate_tab = QWidget()
        otl = QtWidgets.QGridLayout()

        self.orientate_init_cooradinates_lbl = QLabel('Give initial coordinates')
        self.orientate_init_cooradinates_lbl.setFont(bold_font)
        self.orientate_init_x_lbl = QLabel('scanned axis:')
        self.orientate_init_x = QSpinBox()
        self.orientate_init_x.setRange(0, 1000)
        self.orientate_init_y_lbl = QLabel('slit axis:')
        self.orientate_init_y = QSpinBox()
        self.orientate_init_y.setRange(0, 1000)

        self.orientate_find_gamma = QPushButton('Find \t \u0393')
        self.orientate_copy_coords = QPushButton('Copy from \'Volume\'')

        self.orientate_find_gamma_message = QLineEdit('NOTE: algorithm will process the main plot image.')
        self.orientate_find_gamma_message.setReadOnly(True)

        self.orientate_lines_lbl = QLabel('Show rotatable lines')
        self.orientate_lines_lbl.setFont(bold_font)
        self.orientate_hor_line = QCheckBox('horizontal line')
        self.orientate_hor_line
        self.orientate_ver_line = QCheckBox('vertical line')
        self.orientate_angle_lbl = QLabel('rotation angle (deg):')
        self.orientate_angle = QDoubleSpinBox()
        self.orientate_angle.setRange(-180, 180)
        self.orientate_angle.setSingleStep(0.5)

        self.orientate_info_button = QPushButton('info')

        sd = 1
        # addWidget(widget, row, column, rowSpan, columnSpan)
        row = 0
        otl.addWidget(self.orientate_init_cooradinates_lbl,   row * sd, 0 * sd, 1, 2)
        otl.addWidget(self.orientate_init_x_lbl,              (row + 1) * sd, 0 * sd)
        otl.addWidget(self.orientate_init_x,                  (row + 1) * sd, 1 * sd)
        otl.addWidget(self.orientate_init_y_lbl,              (row + 1) * sd, 2 * sd)
        otl.addWidget(self.orientate_init_y,                  (row + 1) * sd, 3 * sd)

        row = 2
        otl.addWidget(self.orientate_find_gamma,              row * sd, 0 * sd, 1, 2)
        otl.addWidget(self.orientate_copy_coords,             row * sd, 2 * sd, 1, 2)
        otl.addWidget(self.orientate_find_gamma_message,      (row + 1) * sd, 0 * sd, 1, 4)

        col = 4
        otl.addWidget(self.orientate_lines_lbl,               0 * sd, col * sd, 1, 2)
        otl.addWidget(self.orientate_hor_line,                1 * sd, col * sd)
        otl.addWidget(self.orientate_ver_line,                1 * sd, (col + 1) * sd)
        otl.addWidget(self.orientate_angle_lbl,               2 * sd, col * sd)
        otl.addWidget(self.orientate_angle,                   2 * sd, (col + 1) * sd)
        otl.addWidget(self.orientate_info_button,                    3 * sd, (col + 1) * sd)

        # dummy lbl
        dummy_lbl = QLabel('')
        otl.addWidget(dummy_lbl, 4, 0, 2, 8)

        self.orientate_tab.layout = otl
        self.orientate_tab.setLayout(otl)
        self.tabs.addTab(self.orientate_tab, 'Orientate')

        self.set_orientation_info_window()

    def set_file_tab(self):

        self.file_tab = QWidget()
        ftl = QtWidgets.QGridLayout()

        self.file_add_md_lbl = QLabel('Edit entries')
        self.file_add_md_lbl.setFont(bold_font)
        self.file_md_name_lbl = QLabel('name:')
        self.file_md_name = QLineEdit()
        self.file_md_value_lbl = QLabel('value:')
        self.file_md_value = QLineEdit()
        self.file_add_md_button = QPushButton('add/update')
        self.file_remove_md_button = QPushButton('remove')

        self.file_show_md_button = QPushButton('show metadata')

        self.file_sum_datasets_lbl = QLabel('Sum data sets')
        self.file_sum_datasets_lbl.setFont(bold_font)
        self.file_sum_datasets_fname_lbl = QLabel('file name:')
        self.file_sum_datasets_fname = QLineEdit('Only *.h5 files')
        self.file_sum_datasets_sum_button = QPushButton('sum')
        self.file_sum_datasets_reset_button = QPushButton('reset')

        self.file_jn_main_lbl = QLabel('Jupyter')
        self.file_jn_main_lbl.setFont(bold_font)
        self.file_jn_fname_lbl = QLabel('file name:')
        self.file_jn_fname = QLineEdit(self.mw.title.split('.')[0])
        self.file_jn_button = QPushButton('open in jn')

        self.file_mdc_fitter_lbl = QLabel('MDC fitter')
        self.file_mdc_fitter_lbl.setFont(bold_font)
        self.file_mdc_fitter_button = QPushButton('Open')

        self.file_edc_fitter_lbl = QLabel('EDC fitter')
        self.file_edc_fitter_lbl.setFont(bold_font)
        self.file_edc_fitter_button = QPushButton('Open')

        sd = 1
        # addWidget(widget, row, column, rowSpan, columnSpan)
        row = 0
        ftl.addWidget(self.file_add_md_lbl,                     row * sd, 0 * sd, 1, 2)
        ftl.addWidget(self.file_show_md_button,                 row * sd, 8 * sd, 1, 2)

        row = 1
        ftl.addWidget(self.file_md_name_lbl,                    row * sd, 0 * sd)
        ftl.addWidget(self.file_md_name,                        row * sd, 1 * sd, 1, 3)
        ftl.addWidget(self.file_md_value_lbl,                   row * sd, 4 * sd)
        ftl.addWidget(self.file_md_value,                       row * sd, 5 * sd, 1, 3)
        ftl.addWidget(self.file_add_md_button,                  row * sd, 8 * sd)
        ftl.addWidget(self.file_remove_md_button,               row * sd, 9 * sd)

        row = 2
        ftl.addWidget(self.file_sum_datasets_lbl,               row * sd, 0 * sd, 1, 2)
        # ftl.addWidget(self.file_sum_datasets_fname_lbl,         row * sd, 2 * sd)
        ftl.addWidget(self.file_sum_datasets_fname,             row * sd, 2 * sd, 1, 6)
        ftl.addWidget(self.file_sum_datasets_sum_button,        row * sd, 8 * sd)
        ftl.addWidget(self.file_sum_datasets_reset_button,      row * sd, 9 * sd)

        row = 3
        ftl.addWidget(self.file_jn_main_lbl,                    row * sd, 0 * sd, 1, 2)
        # ftl.addWidget(self.file_jn_fname_lbl,                   row * sd, 2 * sd)
        ftl.addWidget(self.file_jn_fname,                       row * sd, 2 * sd, 1, 6)
        ftl.addWidget(self.file_jn_button,                      row * sd, 8 * sd)

        if self.dim == 2:
            row = 4
            ftl.addWidget(self.file_mdc_fitter_lbl,             row * sd, 0, 1, 2)
            ftl.addWidget(self.file_mdc_fitter_button,          row * sd, 2)
            ftl.addWidget(self.file_edc_fitter_lbl,             row * sd, 4, 1, 2)
            ftl.addWidget(self.file_edc_fitter_button,          row * sd, 6)

        # dummy lbl
        # dummy_lbl = QLabel('')
        # ftl.addWidget(dummy_lbl, 4, 0, 1, 9)

        self.file_tab.layout = ftl
        self.file_tab.setLayout(ftl)
        self.tabs.addTab(self.file_tab, 'File')

    def setup_cmaps(self):

        cm = self.image_cmaps
        if MY_CMAPS:
            cm.addItems(my_cmaps)
        else:
            for cmap in cmaps.keys():
                cm.addItem(cmap)
        cm.setCurrentText(DEFAULT_CMAP)

    def setup_gamma(self):

        g = self.image_gamma
        g.setRange(0, 10)
        g.setValue(1)
        g.setSingleStep(0.05)

    def setup_colorscale(self):

        cs = self.image_colorscale
        cs.setRange(0, 2)
        cs.setValue(1)
        cs.setSingleStep(0.1)

    def setup_bin_z(self):

        bz = self.bin_z_nbins
        bz.setRange(0, 100)
        bz.setValue(0)

    def set_orientation_info_window(self):
        self.orient_info_window = QWidget()
        oiw = QtWidgets.QGridLayout()

        self.oi_window_lbl = QLabel('piva -> beamline coordinates translator')
        self.oi_window_lbl.setFont(bold_font)
        self.oi_beamline_lbl = QLabel('Beamline')
        self.oi_beamline_lbl.setFont(bold_font)
        self.oi_azimuth_lbl = QLabel('Azimuth (clockwise)')
        self.oi_azimuth_lbl.setFont(bold_font)
        self.oi_analyzer_lbl = QLabel('Analyzer (-> +)')
        self.oi_analyzer_lbl.setFont(bold_font)
        self.oi_scanned_lbl = QLabel('Scanned (-> +)')
        self.oi_scanned_lbl.setFont(bold_font)

        entries = [['SIS (SLS, SIStem)',    'phi -> -',     'theta -> +',   'tilt -> -'],
                   ['SIS (SLS, SES)',       'phi -> +',     'theta -> -',   'tilt -> -'],
                   ['Bloch (MaxIV)',        'azimuth -> +', 'tilt -> -',    'polar -> -'],
                   ['CASSIOPEE (SOLEIL)',   '-',            '-',            '-'],
                   ['I05 (Diamond)',        '-',            '-',            '-'],
                   ['UARPES (SOLARIS)',     '-',            '-',            '-'],
                   ['APE (Elettra)',        '-',            '-',            '-'],
                   ['ADDRES (SLS)',         '-',            '-',            '-'],
                   ['-',                    '-',            '-',            '-'],
                   ['-',                    '-',            '-',            '-']]
        labels = {}

        sd = 1
        row = 0
        oiw.addWidget(self.oi_beamline_lbl,     row * sd, 0 * sd)
        oiw.addWidget(self.oi_azimuth_lbl,      row * sd, 1 * sd)
        oiw.addWidget(self.oi_analyzer_lbl,     row * sd, 2 * sd)
        oiw.addWidget(self.oi_scanned_lbl,      row * sd, 3 * sd)

        for entry in entries:
            row += 1
            labels[str(row)] = {}
            labels[str(row)]['beamline'] = QLabel(entry[0])
            labels[str(row)]['azimuth'] = QLabel(entry[1])
            labels[str(row)]['azimuth'].setAlignment(QtCore.Qt.AlignCenter)
            labels[str(row)]['analyzer'] = QLabel(entry[2])
            labels[str(row)]['analyzer'].setAlignment(QtCore.Qt.AlignCenter)
            labels[str(row)]['scanned'] = QLabel(entry[3])
            labels[str(row)]['scanned'].setAlignment(QtCore.Qt.AlignCenter)

            oiw.addWidget(labels[str(row)]['beamline'],    row * sd, 0 * sd)
            oiw.addWidget(labels[str(row)]['azimuth'],     row * sd, 1 * sd)
            oiw.addWidget(labels[str(row)]['analyzer'],    row * sd, 2 * sd)
            oiw.addWidget(labels[str(row)]['scanned'],     row * sd, 3 * sd)

        self.orient_info_window.layout = oiw
        self.orient_info_window.setLayout(oiw)

    def set_metadata_window(self, dataset):

        self.md_window = QWidget()
        mdw = QtWidgets.QGridLayout()

        attribute_name_lbl = QLabel('Attribute')
        attribute_name_lbl.setFont(bold_font)
        attribute_value_lbl = QLabel('Value')
        attribute_value_lbl.setFont(bold_font)
        attribute_value_lbl.setAlignment(QtCore.Qt.AlignCenter)
        attribute_saved_lbl = QLabel('user saved')
        attribute_saved_lbl.setFont(bold_font)
        attribute_saved_lbl.setAlignment(QtCore.Qt.AlignCenter)

        dataset = vars(dataset)
        entries = {}

        sd = 1
        row = 0
        mdw.addWidget(attribute_name_lbl,   row * sd, 0 * sd)
        mdw.addWidget(attribute_value_lbl,  row * sd, 1 * sd)

        row = 1
        for key in dataset.keys():
            if key == 'ekin' or key == 'saved':
                continue
            elif key == 'data':
                s = dataset[key].shape
                value = '(' + str(s[0]) + ',  ' + str(s[1]) + ',  ' + str(s[2]) + ')'
                entries[str(row)] = {}
                entries[str(row)]['name'] = QLabel(key)
                entries[str(row)]['value'] = QLabel(str(value))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            elif key == 'xscale':
                value = '({:.2f}  :  {:.2f})'.format(dataset[key][0], dataset[key][-1])
                entries[str(row)] = {}
                entries[str(row)]['name'] = QLabel(key)
                entries[str(row)]['value'] = QLabel(str(value))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            elif key == 'yscale':
                value = '({:.4f}  :  {:.4f})'.format(dataset[key][0], dataset[key][-1])
                entries[str(row)] = {}
                entries[str(row)]['name'] = QLabel(key)
                entries[str(row)]['value'] = QLabel(str(value))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            elif key == 'zscale':
                value = '({:.4f}  :  {:.4f})'.format(dataset[key][0], dataset[key][-1])
                entries[str(row)] = {}
                entries[str(row)]['name'] = QLabel(key)
                entries[str(row)]['value'] = QLabel(str(value))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            elif key == 'kxscale':
                if not (dataset[key] is None):
                    value = '({:.3f}  :  {:.3f})'.format(dataset[key][0], dataset[key][-1])
                    entries[str(row)] = {}
                    entries[str(row)]['name'] = QLabel(key)
                    entries[str(row)]['value'] = QLabel(str(value))
                    entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
                else:
                    entries[str(row)] = {}
                    entries[str(row)]['name'] = QLabel(key)
                    entries[str(row)]['value'] = QLabel(str(dataset[key]))
                    entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            elif key == 'kyscale':
                if not (dataset[key] is None):
                    value = '({:.3f}  :  {:.3f})'.format(dataset[key][0], dataset[key][-1])
                    entries[str(row)] = {}
                    entries[str(row)]['name'] = QLabel(key)
                    entries[str(row)]['value'] = QLabel(str(value))
                    entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
                else:
                    entries[str(row)] = {}
                    entries[str(row)]['name'] = QLabel(key)
                    entries[str(row)]['value'] = QLabel(str(dataset[key]))
                    entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            elif key == 'pressure':
                entries[str(row)] = {}
                entries[str(row)]['name'] = QLabel(key)
                entries[str(row)]['value'] = QLabel('{:.4e}'.format((dataset[key])))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)
            else:
                entries[str(row)] = {}
                entries[str(row)]['name'] = QLabel(key)
                if isinstance(dataset[key], float):
                    entries[str(row)]['value'] = QLabel('{:.4f}'.format((dataset[key])))
                else:
                    entries[str(row)]['value'] = QLabel(str(dataset[key]))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)

            mdw.addWidget(entries[str(row)]['name'],    row * sd, 0 * sd)
            mdw.addWidget(entries[str(row)]['value'],   row * sd, 1 * sd)
            row += 1

        if 'saved' in dataset.keys():
            mdw.addWidget(attribute_saved_lbl,   row * sd, 0 * sd, 1, 2)
            for key in dataset['saved'].keys():
                row += 1
                entries[str(row)] = {}
                entries[str(row)]['name'] = QLabel(key)
                if key == 'kx' or key == 'ky' or key == 'k':
                    value = '({:.2f}  :  {:.2f})'.format(dataset['saved'][key][0], dataset['saved'][key][-1])
                    entries[str(row)]['value'] = QLabel(str(value))
                else:
                    entries[str(row)]['value'] = QLabel(str(dataset['saved'][key]))
                entries[str(row)]['value'].setAlignment(QtCore.Qt.AlignCenter)

                mdw.addWidget(entries[str(row)]['name'],    row * sd, 0 * sd)
                mdw.addWidget(entries[str(row)]['value'],   row * sd, 1 * sd)

        self.md_window.layout = mdw
        self.md_window.setLayout(mdw)

    def show_metadata_window(self):

        self.set_metadata_window(self.mw.data_set)
        title = self.mw.title + ' - metadata'
        self.info_box = InfoWindow(self.md_window, title)
        self.info_box.setMinimumWidth(350)
        self.info_box.show()

    def add_metadata(self):

        name = self.file_md_name.text()
        value = self.file_md_value.text()
        try:
            value = float(value)
        except ValueError:
            pass

        if name == '':
            empty_name_box = QMessageBox()
            empty_name_box.setIcon(QMessageBox.Information)
            empty_name_box.setText('Attribute\'s name not given.')
            empty_name_box.setStandardButtons(QMessageBox.Ok)
            if empty_name_box.exec() == QMessageBox.Ok:
                return

        message = 'Sure to add attribute \'{}\' with value <{}> (type: {}) to the file?'.format(
            name, value, type(value))
        sanity_check_box = QMessageBox()
        sanity_check_box.setIcon(QMessageBox.Question)
        sanity_check_box.setText(message)
        sanity_check_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        if sanity_check_box.exec() == QMessageBox.Ok:
            if hasattr(self.mw.data_set, name):
                attr_conflict_box = QMessageBox()
                attr_conflict_box.setIcon(QMessageBox.Question)
                attr_conflict_box.setText(f'Data set already has attribute \'{name}\'.  Overwrite?')
                attr_conflict_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                if attr_conflict_box.exec() == QMessageBox.Ok:
                    setattr(self.mw.data_set, name, value)
            else:
                dl.update_namespace(self.mw.data_set, [name, value])
        else:
            return

    def remove_metadata(self):

        name = self.file_md_name.text()

        if not hasattr(self.mw.data_set, name):
            no_attr_box = QMessageBox()
            no_attr_box.setIcon(QMessageBox.Information)
            no_attr_box.setText(f'Attribute \'{name}\' not found.')
            no_attr_box.setStandardButtons(QMessageBox.Ok)
            if no_attr_box.exec() == QMessageBox.Ok:
                return

        message = 'Sure to remove attribute \'{}\' from the data set?'.format(name)
        sanity_check_box = QMessageBox()
        sanity_check_box.setIcon(QMessageBox.Question)
        sanity_check_box.setText(message)
        sanity_check_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        if sanity_check_box.exec() == QMessageBox.Ok:
            delattr(self.mw.data_set, name)
        else:
            return

    def sum_datasets(self):

        if self.dim == 3:
            no_map_box = QMessageBox()
            no_map_box.setIcon(QMessageBox.Information)
            no_map_box.setText('Summing feature works only on cuts.')
            no_map_box.setStandardButtons(QMessageBox.Ok)
            if no_map_box.exec() == QMessageBox.Ok:
                return

        file_path = self.mw.fname[:-len(self.mw.title)] + self.file_sum_datasets_fname.text()
        org_dataset = dl.load_data(self.mw.fname)

        try:
            new_dataset = dl.load_data(file_path)
        except FileNotFoundError:
            no_file_box = QMessageBox()
            no_file_box.setIcon(QMessageBox.Information)
            no_file_box.setText('File not found.')
            no_file_box.setStandardButtons(QMessageBox.Ok)
            if no_file_box.exec() == QMessageBox.Ok:
                return

        try:
            check_result = self.check_conflicts([org_dataset, new_dataset])
        except AttributeError:
            not_h5_file_box = QMessageBox()
            not_h5_file_box.setIcon(QMessageBox.Information)
            not_h5_file_box.setText('Cut is not an SIStem *h5 file.')
            not_h5_file_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            if not_h5_file_box.exec() == QMessageBox.Cancel:
                return
            else:
                pass

        if check_result == 0:
            data_mismatch_box = QMessageBox()
            data_mismatch_box.setIcon(QMessageBox.Information)
            data_mismatch_box.setText('Data sets\' shapes don\'t match.\nConnot proceed.')
            data_mismatch_box.setStandardButtons(QMessageBox.Ok)
            if data_mismatch_box.exec() == QMessageBox.Ok:
                return

        check_result_box = QMessageBox()
        check_result_box.setMinimumWidth(600)
        check_result_box.setMaximumWidth(1000)
        check_result_box.setIcon(QMessageBox.Information)
        check_result_box.setText(check_result)
        check_result_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        if check_result_box.exec() == QMessageBox.Ok:
            self.mw.org_dataset = org_dataset
            self.mw.data_set.data += new_dataset.data
            self.mw.data_set.n_sweeps += new_dataset.n_sweeps
            d = np.swapaxes(self.mw.data_set.data, 1, 2)
            self.mw.data_handler.set_data(d)
            self.mw.update_main_plot()
        else:
            return

    def reset_summation(self):

        if self.mw.org_dataset is None:
            no_summing_yet_box = QMessageBox()
            no_summing_yet_box.setIcon(QMessageBox.Information)
            no_summing_yet_box.setText('No summing done yet.')
            no_summing_yet_box.setStandardButtons(QMessageBox.Ok)
            if no_summing_yet_box.exec() == QMessageBox.Ok:
                return

        reset_summation_box = QMessageBox()
        reset_summation_box.setMinimumWidth(600)
        reset_summation_box.setMaximumWidth(1000)
        reset_summation_box.setIcon(QMessageBox.Question)
        reset_summation_box.setText('Want to reset summation?')
        reset_summation_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        if reset_summation_box.exec() == QMessageBox.Ok:
            self.mw.data_set.data = self.mw.org_dataset.data
            self.mw.data_set.n_sweeps = self.mw.org_dataset.n_sweeps
            d = np.swapaxes(self.mw.data_set.data, 1, 2)
            self.mw.data_handler.set_data(d)
            self.mw.update_main_plot()
        else:
            return

    def open_jupyter_notebook(self):
        file_path = self.mw.fname[:-len(self.mw.title)] + self.file_jn_fname.text() + '.ipynb'
        template_path = os.path.dirname(os.path.abspath(__file__)) + '/'
        template_fname = 'template.ipynb'
        self.edit_file((template_path + template_fname), file_path)

        # Open jupyter notebook as a subprocess
        openJupyter = "jupyter notebook"
        subprocess.Popen(openJupyter, shell=True, cwd=self.mw.fname[:-len(self.mw.title)])

    def edit_file(self, template, new_file_name):
        os.system('touch ' + new_file_name)

        templ_file = open(template, 'r')
        templ_lines = templ_file.readlines()
        templ_file.close()

        new_lines = []

        # writing to file
        for line in templ_lines:
            if 'path = ' in line:
                line = '    "path = \'{}\'\\n",'.format(self.mw.fname[:-len(self.mw.title)])
            if 'fname = ' in line:
                line = '    "fname = \'{}\'\\n",'.format(self.mw.title)
            if 'slit_idx, e_idx =' in line:
                if self.dim == 2:
                    line = '    "slit_idx, e_idx = {}, {}\\n",'.format(
                        self.momentum_hor.value(), self.energy_vert.value())
                elif self.dim == 3:
                    line = '    "scan_idx, slit_idx, e_idx = {}, {}, {}\\n",'.format(
                        self.momentum_vert.value(), self.momentum_hor.value(), self.energy_vert.value())
            new_lines.append(line)

        new_file = open(new_file_name, 'w')
        new_file.writelines(new_lines)
        new_file.close()

    @staticmethod
    def check_conflicts(datasets):

        labels = ['fname', 'data', 'T', 'hv', 'polarization', 'PE', 'FE', 'exit', 'x', 'y', 'z', 'theta', 'phi', 'tilt',
                  'lens_mode', 'acq_mode', 'e_start', 'e_stop', 'e_step']
        to_check = [[] for _ in range(len(labels))]

        to_check[0] = ['original', 'new']
        for ds in datasets:
            to_check[1].append(ds.data.shape)
            to_check[2].append(ds.temp)
            to_check[3].append(ds.hv)
            to_check[4].append(ds.polarization)
            to_check[5].append(ds.PE)
            to_check[6].append(ds.FE)
            to_check[7].append(ds.exit_slit)
            to_check[8].append(ds.x)
            to_check[9].append(ds.y)
            to_check[10].append(ds.z)
            to_check[11].append(ds.theta)
            to_check[12].append(ds.phi)
            to_check[13].append(ds.tilt)
            to_check[14].append(ds.lens_mode)
            to_check[15].append(ds.acq_mode)
            to_check[16].append(ds.zscale[0])
            to_check[17].append(ds.zscale[-1])
            to_check[18].append(wp.get_step(ds.zscale))

        # check if imporatnt stuff match
        check_result = []
        for idx, lbl in enumerate(labels):
            if lbl == 'fname' or lbl == 'data':
                check_result.append(True)
            # temperature
            elif lbl == 'T':
                err = 1
                par = np.array(to_check[idx])
                to_compare = np.ones(par.size) * par[0]
                check_result.append(np.allclose(par, to_compare, atol=err))
            # photon energy
            elif lbl == 'hv':
                err = 0.1
                par = np.array(to_check[idx])
                to_compare = np.ones(par.size) * par[0]
                check_result.append(np.allclose(par, to_compare, atol=err))
            # e_min of analyzer
            elif lbl == 'e_start':
                err = to_check[-1][0]
                par = np.array(to_check[idx])
                to_compare = np.ones(par.size) * par[0]
                check_result.append(np.allclose(par, to_compare, atol=err))
            # e_max of analyzer
            elif lbl == 'e_stop':
                err = to_check[-1][0]
                par = np.array(to_check[idx])
                to_compare = np.ones(par.size) * par[0]
                check_result.append(np.allclose(par, to_compare, atol=err))
            elif lbl == 'e_step':
                err = to_check[-1][0]
                par = np.array(to_check[idx])
                to_compare = np.ones(par.size) * par[0]
                check_result.append(np.allclose(par, to_compare, atol=err * 0.1))
            else:
                check_result.append(to_check[idx][0] == to_check[idx][1])

        if not (to_check[1][0] == to_check[1][1]):
            return 0

        if np.array(check_result).all():
            message = 'Everything match! We\'re good to go!'
        else:
            message = 'Some stuff doesn\'t match...\n\n'
            dont_match = np.where(np.array(check_result) == False)
            for idx in dont_match[0]:
                try:
                    message += '{} \t\t {:.3f}\t  {:.3f} \n'.format(str(labels[idx]), to_check[idx][0], to_check[idx][1])
                except TypeError:
                    message += '{} \t\t {}\t  {} \n'.format(
                        str(labels[idx]), str(to_check[idx][0]), str(to_check[idx][1]))
                except ValueError:
                    message += '{} \t\t {}\t  {} \n'.format(
                        str(labels[idx]), str(to_check[idx][0]), str(to_check[idx][1]))
            message += '\nSure to proceed?'

        return message


class InfoWindow(QMainWindow):

    def __init__(self, info_widget, title):
        super(InfoWindow, self).__init__()

        self.scroll_area = QScrollArea()
        self.central_widget = QWidget()
        self.info_window_layout = QtWidgets.QGridLayout()
        self.central_widget.setLayout(self.info_window_layout)

        QBtn = QDialogButtonBox.Ok
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.clicked.connect(self.close)

        self.info_window_layout.addWidget(info_widget)
        self.info_window_layout.addWidget(self.buttonBox)
        self.scroll_area.setWidget(self.central_widget)
        self.setCentralWidget(self.scroll_area)
        self.setWindowTitle(title)


class TracedVariable(QtCore.QObject):

    """ A pyqt implementaion of tkinter's/Tcl's traced variables using Qt's
    signaling mechanism.
    Basically this is just a wrapper around any python object which ensures
    that pyQt signals are emitted whenever the object is accessed or changed.

    In order to use pyqt's signals, this has to be a subclass of
    :class:`QObject <pyqtgraph.Qt.QtCore.QObject>`.

    **Attributes**

    ==========================  ================================================
    _value                      the python object represented by this
                                TracedVariable instance. Should never be
                                accessed directly but only through the getter
                                and setter methods.
    sig_value_changed           :class:`Signal <pyqtgraph.Qt.QtCore.Signal>`;
                                the signal that is emitted whenever
                                ``self._value`` is changed.
    sig_value_read              :class:`Signal <pyqtgraph.Qt.QtCore.Signal>`;
                                the signal that is emitted whenever
                                ``self._value`` is read.
    sig_allowed_values_changed  :class:`Signal <pyqtgraph.Qt.QtCore.Signal>`;
                                the signal that is emitted whenever
                                ``self.allowed_values`` are set or unset.
    allowed_values              :class:`array <numpy.ndarray>`; a sorted
                                list of all values that self._value can
                                assume. If set, all tries to set the value
                                will automatically set it to the closest
                                allowed one.
    ==========================  ================================================
    """
    sig_value_changed = QtCore.Signal()
    sig_value_read = QtCore.Signal()
    sig_allowed_values_changed = QtCore.Signal()

    def __init__(self, value=None, name=None):
        # Initialize instance variables
        self.allowed_values = None
        self.min_allowed = None
        self.max_allowed = None

        # Have to call superclass init for signals to work
        super().__init__()

        self._value = value
        if name is not None:
            self.name = name
        else:
            self.name = 'Unnamed'

    def __repr__(self):
        return '<TracedVariable({}, {})>'.format(self.name, self._value)

    def set_value(self, value=None):
        """ Emit sig_value_changed and set the internal self._value. """
        # Choose the closest allowed value
        if self.allowed_values is not None:
            value = self.find_closest_allowed(value)
        self._value = value
        self.sig_value_changed.emit()

    def get_value(self):
        """ Emit sig_value_changed and return the internal self._value.

        .. warning::
            the signal is emitted here before the caller actually receives
            the return value. This could lead to unexpected behaviour.
        """
        self.sig_value_read.emit()
        return self._value

    def on_change(self, callback):
        """ Convenience wrapper for :class:`Signal
        <pyqtgraph.Qt.QtCore.Signal>`'s 'connect'.
        """
        self.sig_value_changed.connect(callback)

    def on_read(self, callback):
        """ Convenience wrapper for :class:`Signal
        <pyqtgraph.Qt.QtCore.Signal>`'s 'connect'.
        """
        self.sig_value_read.connect(callback)

    def set_allowed_values(self, values=None, binning=0):
        """ Define a set/range/list of values that are allowed for this
        Variable. Once set, all future calls to set_value will automatically
        try to pick the most reasonable of the allowed values to assign.

        Emits :signal:`sig_allowed_values_changed`

        **Parameters**

        ======  =================================================================
        values  iterable; The complete list of allowed (numerical) values. This
                is converted to a sorted np.array internally. If values is
                `None`, all restrictions on allowed values will be lifted and
                all values are allowed.
        ======  =================================================================
        """
        if values is None:
            # Reset the allowed values, i.e. all values are allowed
            self.allowed_values = None
            self.min_allowed = None
            self.max_allowed = None
        else:
            # Convert to sorted numpy array
            try:
                if binning == 0:
                    values = np.array(values)
                else:
                    values = np.array(values)[binning:-binning]
            except TypeError:
                message = 'Could not convert allowed values to np.array.'
                raise TypeError(message)

            # Sort the array for easier indexing later on
            values.sort()
            self.allowed_values = values

            # Store the max and min allowed values (necessary?)
            self.min_allowed = values[0]
            self.max_allowed = values[-1]

        # Update the current value to within the allowed range
        self.set_value(self._value)
        self.sig_allowed_values_changed.emit()

    def find_closest_allowed(self, value):
        """ Return the value of the element in self.allowed_values (if set)
        that is closest to `value`.
        """
        if self.allowed_values is None:
            return value
        else:
            ind = np.abs(self.allowed_values - value).argmin()
            return self.allowed_values[ind]


class ThreadClass(QtCore.QThread):
    any_signal = QtCore.pyqtSignal(int)

    def __init__(self, parent=None, index=0):
        super(ThreadClass, self).__init__(parent)
        self.index = index
        self.is_running = True

    def stop(self):
        self.quit()

