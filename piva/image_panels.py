from __future__ import annotations
import numpy as np
import pyqtgraph as pg
from typing import Any, Iterable
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph.graphicsItems.ImageItem import ImageItem
from data_slicer.imageplot import TracedVariable

BASE_LINECOLOR = (255, 255, 0, 255)
BINLINES_LINECOLOR = (168, 168, 104, 255)
ORIENTLINES_LINECOLOR = (164, 37, 22, 255)
HOVER_COLOR = (195, 155, 0, 255)
BGR_COLOR = (64, 64, 64)
util_panel_style = """
QFrame{margin:5px; border:1px solid rgb(150,150,150);}
QLabel{color: rgb(246, 246, 246); border:1px solid rgb(64, 64, 64);}
QCheckBox{color: rgb(246, 246, 246);}
QTabWidget{background-color: rgb(64, 64, 64);}
"""
SIGNALS = 5
MY_CMAPS = True
DEFAULT_CMAP = 'coolwarm'

bold_font = QtGui.QFont()
bold_font.setBold(True)


class Sliders:
    """
    Object defining draggable lines allowing user to choose exact position of
    a slice.
    """

    def __init__(self, image_plot: ImagePlot,
                 pos: Iterable = (0, 0), mainplot: bool = True,
                 orientation: str = 'horizontal') -> None:
        """
        Initialize two draggable sliders.

        :param image_plot: :class:`ImagePlot` on which sliders are displayed
        :param pos: initial position of the sliders
        :param mainplot: if :py:obj:`True` register signals for both sliders.
                         Different behavior is expected for :class:`ImagePlot`\
                         s being horizontal and vertical cuts
        :param orientation: relative orientation of the image, when `vertical`
                            the actual orientation of sliders is flipped.
                            Default is `horizontal`.

                            .. note::
                                Actually, it's easier to define sliders
                                in coordinate system of the data, where
                                horizontal always corresponds to slicing along
                                momentum.
        """

        self.image_plot = image_plot
        self.orientation = orientation

        # Store the positions in TracedVariables
        # self.hpos = TracedVariable(pos[1], name='hpos')
        # self.vpos = TracedVariable(pos[0], name='vpos')
        self.hpos = CustomTracedVariable(pos[1], name='hpos')
        self.vpos = CustomTracedVariable(pos[0], name='vpos')

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

    def add_to(self, widget: Any) -> None:
        """
        Add these sliders to a :mod:`pyqtgraph` widget.

        :param widget: widget to which sliders are added
        """

        for line in [self.hline, self.vline]:
            line.setZValue(1)
            widget.addItem(line)

    def remove_from(self, widget: Any) -> None:
        """
        Remove these sliders from a :mod:`pyqtgraph` widget.

        :param widget: widget to which sliders are removed
        """

        for line in [self.hline, self.vline]:
            widget.removeItem(line)

    def set_color(self, linecolor: Any = BASE_LINECOLOR,
                  hover_color: Any = HOVER_COLOR) -> None:
        """
        Set the color and hover color of both sliders that make up. The
        arguments can be any :mod:`pyqtgraph` compatible color specifiers.

        :param linecolor: can be any :mod:`pyqtgraph` compatible color
                          specifiers
        :param hover_color: can be any :mod:`pyqtgraph` compatible color
                            specifiers
        """

        for line in [self.hline, self.vline]:
            line.setPen(linecolor)
            line.setHoverPen(hover_color)

    def set_movable(self, movable: bool = True) -> None:
        """
        Define whether these sliders can be dragged by the mouse or not.

        :param movable: if :py:obj:`True`, make sliders movable
        """

        for line in [self.hline, self.vline]:
            line.setMovable = movable

    def update_position_h(self) -> None:
        """
        Update position of the horizontal slider.
        """

        self.hline.setValue(self.hpos.get_value())

    def update_position_v(self) -> None:
        """
        Update position of the vertical slider.
        """

        self.vline.setValue(self.vpos.get_value())

    def on_dragged_h(self):
        """
        Callback for dragging horizontal slider changing value of the
        connected :class:`CustomTracedVariable`.
        """

        self.hpos.set_value(self.hline.value())
        # if it's an energy plot, and binning option is active,
        # update also binning boundaries
        if self.image_plot.binning:
            pos = self.hline.value()
            self.image_plot.left_line.setValue(pos - self.image_plot.width)
            self.image_plot.right_line.setValue(pos + self.image_plot.width)

    def on_dragged_v(self) -> None:
        """
        Callback for dragging vertical slider changing value of the connected
        :class:`CustomTracedVariable`.
        """

        self.vpos.set_value(self.vline.value())

    def set_bounds(self, xmin: int, xmax: int, ymin: int, ymax: int) -> None:
        """
        Set boundaries within which sliders can be dragged.

        :param xmin: lowest value along horizontal direction
        :param xmax: highest value along horizontal direction
        :param ymin: lowest value along vertical direction
        :param ymax: highest value along vertical direction
        """

        if self.orientation == 'horizontal':
            self.hline.setBounds([ymin, ymax])
            self.vline.setBounds([xmin, xmax])
        else:
            self.vline.setBounds([ymin, ymax])
            self.hline.setBounds([xmin, xmax])


class ImagePlot(pg.PlotWidget):
    """
    Object displaying 2D color-scaled data using different colormaps. It treats
    data as a regular rectangular images, which allows for time efficient
    slicing and updating displayed cuts.
    Inherits from :class:`pyqtgraph.PlotWidget` which gives access to
    additional *plotty* features like displaying custom scales instead of just
    pixels.
    """

    sig_image_changed = QtCore.Signal()
    sig_axes_changed = QtCore.Signal()
    sig_clicked = QtCore.Signal(object)

    def __init__(self, image: np.ndarray = None, background: Any = BGR_COLOR,
                 name: str = None, mainplot: bool = True,
                 orientation: str = 'horizontal', **kwargs: dict) -> None:
        """
        Initialize color-scaled plot.

        :param image: 2D array with data
        :param background: color of the background, can be any pyqtgraph
                           compatible color specifier
        :param mainplot: specifies behavior of the draggable sliders.
                         See :class:`Sliders` for more details
        :param orientation: orientation of the image. Default is '`horizontal`'
        :param kwargs: kwargs passed to parent :class:`pg.PlotWidget`
        """

        super().__init__(background=background, **kwargs)

        # Initialize instance variables
        # np.array, raw image data
        self.image_data = None
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

        if name is not None:
            self.name = name
        else:
            self.name = 'Unnamed'
        self.orientate()

        if image is not None:
            self.set_image(image)

        # Initiliaze a sliders and add it to this widget
        self.sliders = Sliders(self, mainplot=mainplot,
                               orientation=orientation)
        self.sliders.add_to(self)

        self.pos = (self.sliders.vpos, self.sliders.hpos)

        # Initialize range to [0, 1]x[0, 1]
        self.set_bounds(0, 1, 0, 1)

        # Disable mouse scrolling, panning and zooming for both axes
        self.setMouseEnabled(False, False)

        # Connect a slot (callback) to dragging and clicking events
        self.sig_axes_changed.connect(
            lambda: self.set_bounds(*[x for lst in
                                      self.get_limits() for x in lst]))

        self.sig_image_changed.connect(self.update_allowed_values)

    # methods added to make crosshairs work
    def update_allowed_values(self) -> None:
        """
        Update the allowed values of the sliders. This assumes that the
        displayed image is in pixel coordinates and sets the allowed values
        to the available pixels.
        """

        [[xmin, xmax], [ymin, ymax]] = self.get_limits()
        if self.orientation == 'horizontal':
            self.pos[0].set_allowed_values(np.arange(xmin, xmax + 1, 1))
            self.pos[1].set_allowed_values(np.arange(ymin, ymax + 1, 1))
        else:
            self.pos[1].set_allowed_values(np.arange(xmin, xmax + 1, 1))
            self.pos[0].set_allowed_values(np.arange(ymin, ymax + 1, 1))

    def set_bounds(self, xmin: int, xmax: int, ymin: int, ymax: int) -> None:
        """
        Set both, the displayed area of the axes and the range in which
        sliders can be dragged.

        :param xmin: lowest value along horizontal direction
        :param xmax: highest value along horizontal direction
        :param ymin: lowest value along vertical direction
        :param ymax: highest value along vertical direction
        """

        self.setXRange(xmin, xmax, padding=0.0)
        self.setYRange(ymin, ymax, padding=0.0)

        self.sliders.set_bounds(xmin, xmax, ymin, ymax)

    def orientate(self) -> None:
        """
        Configure plot's layout depending on its orientation.
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

    def remove_image(self) -> None:
        """
        Remove currently displayed image.
        """

        if self.image_item is not None:
            self.removeItem(self.image_item)
        self.image_item = None

    def set_image(self, image: np.ndarray, emit: bool = True,
                  *args: dict, **kwargs: dict) -> None:
        """
        Set/update displayed image. Also make sure the old one has been
        removed.

        :param image: 2D array with data
        :param emit: if :py:obj:`True`, emmit signal that image has been
                     changed
        :param args: additional arguments for a parent
                     :class:`~pyqtgraph.graphicsItems.ImageItem.ImageItem`
        :param kwargs: additional keyword arguments for a parent
                       :class:`~pyqtgraph.graphicsItems.ImageItem.ImageItem`
        """

        # Convert array to ImageItem
        if isinstance(image, np.ndarray):
            if 0 not in image.shape:
                image_item = ImageItem(image, *args, **kwargs)
            else:
                return
        else:
            image_item = image

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

    def set_xscale(self, scale: np.ndarray) -> None:
        """
        Set custom values of the ``xscale``.

        :param scale: 1D array with axis values
        """

        if self.orientation == 'vertical':
            self.yscale = scale
            self.ylim = (scale[0], scale[-1])
        else:
            self.xscale = scale
            self.xlim = (scale[0], scale[-1])

    def set_yscale(self, scale: np.ndarray) -> None:
        """
        Set custom values of the ``yscale``.

        :param scale: 1D array with axis values
        """

        if self.orientation == 'vertical':
            self.xscale = scale
            self.xlim = (scale[0], scale[-1])
        else:
            self.yscale = scale
            self.ylim = (scale[0], scale[-1])

    def set_ticks(self, min_val: float, max_val: float, axis: str) -> None:
        """
        Set customized axis' ticks to reflect the dimensions of the physical
        data.

        :param min_val: lowest axis' value
        :param max_val: highest axis' value
        :param axis: concerned axis, can be [`bottom`, `left`, *etc*]
        """

        plotItem = self.plotItem

        # Remove the old top-axis
        plotItem.layout.removeItem(plotItem.getAxis(axis))
        # Create the new axis and set its range
        new_axis = pg.AxisItem(orientation=axis)
        new_axis.setRange(min_val, max_val)
        # Attach it internally to the plotItem and its layout (The arguments
        plotItem.axes[axis]['item'] = new_axis
        if axis == 'bottom':
            plotItem.layout.addItem(new_axis, *self.main_xaxis_grid)
        else:
            plotItem.layout.addItem(new_axis, *self.main_yaxis_grid)

    def _set_axes_scales(self, emit: bool = False) -> None:
        """
        Transform the image so that it matches the desired *x* and *y* scales.

        :param emit: if :py:obj:`True` emmit signal that axes has changed
        """

        # Get image dimensions and requested origin (x0,y0) and
        # top right corner (x1, y1)
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
        # self._update_transform_factors()

        if emit:
            self.sig_axes_changed.emit()

    def set_secondary_axis(self, min_val: float, max_val: float) -> None:
        """
        Set/update a second `x`-axis on the top.

        :param min_val: lowest axis' value
        :param max_val: highest axis' value
        """

        # Get a handle on the underlying plotItem
        plotItem = self.plotItem

        # Remove the old top-axis
        plotItem.layout.removeItem(plotItem.getAxis(self.secondary_axis))
        # Create the new axis and set its range
        new_axis = pg.AxisItem(orientation=self.secondary_axis)
        new_axis.setRange(min_val, max_val)
        # Attach it internally to the plotItem and its layout (The arguments
        plotItem.axes[self.secondary_axis]['item'] = new_axis
        plotItem.layout.addItem(new_axis, *self.secondary_axis_grid)

    def get_limits(self) -> list:
        """
        Get limits of the image data.

        :return: list of lists in a format: [[`x_min`, `x_max`],
                 [`y_min`, `y_max`]]
        """

        # Default to current viewrange but try to get more accurate values
        # if possible
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

    def fix_viewrange(self) -> None:
        """
        Prevent zooming out by fixing the limits of the :class:`QViewBox`.
        """

        [[x_min, x_max], [y_min, y_max]] = self.get_limits()
        self.setLimits(xMin=x_min, xMax=x_max, yMin=y_min, yMax=y_max,
                       maxXRange=x_max-x_min, maxYRange=y_max-y_min)

    def add_binning_lines(self, pos: int, width: int,
                          orientation: str = 'horizontal') -> None:
        """
        Add not-movable lines around the specified sliders. The lines indicate
        integration area for a corresponding cut.

        :param pos: position of the slider
        :param width: number of left and right steps from the slider position
        :param orientation: orientation of the slider
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
            self.left_hor_line = pg.InfiniteLine(pos - width,
                                                 movable=False, angle=0)
            self.right_hor_line = pg.InfiniteLine(pos + width,
                                                  movable=False, angle=0)
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
            self.left_ver_line = pg.InfiniteLine(pos - width,
                                                 movable=False, angle=90)
            self.right_ver_line = pg.InfiniteLine(pos + width,
                                                  movable=False, angle=90)
            self.left_ver_line.setPen(color=BINLINES_LINECOLOR, width=1)
            self.right_ver_line.setPen(color=BINLINES_LINECOLOR, width=1)
            self.addItem(self.left_ver_line)
            self.addItem(self.right_ver_line)

    def remove_binning_lines(self, orientation: str = 'horizontal') -> None:
        """
        Remove binning lines from the :class:`ImagePlot`.

        :param orientation: orientation of the slider
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

    def register_momentum_slider(self,
                                 traced_variable: TracedVariable) -> None:
        """
        Register vertical slider draggable along momentum in the **cut** plots.

        :param traced_variable: connected traced variable
        """

        self.sliders.vpos = traced_variable
        self.sliders.vpos.sig_value_changed.connect(self.set_position)
        self.sliders.vpos.sig_allowed_values_changed.connect(
            self.on_allowed_values_change)

    def set_position(self) -> None:
        """
        Set the position of the momentum sliders whenever the value of
        connected :class:`CustomTracedVariable` has changed.
        """

        if self.orientation == 'horizontal':
            new_pos = self.sliders.vpos.get_value()
            self.sliders.vline.setValue(new_pos)
        elif self.orientation == 'vertical':
            new_pos = self.sliders.vpos.get_value()
            self.sliders.hline.setValue(new_pos)

    def on_allowed_values_change(self) -> None:
        """
        Set new momentum slider bounds after changing allowed values of the
        connected :class:`CustomTracedVariable`, *e.g.* after setting binning
        lines.
        """

        # If the allowed values were reset, just exit
        if self.sliders.vpos.allowed_values is None:
            return

        lower = self.sliders.vpos.min_allowed
        upper = self.sliders.vpos.max_allowed
        self.sliders.vline.setBounds([lower, upper])


class CurvePlot(pg.PlotWidget):
    """
    Object displaying basic 1D curves with a draggable slider.
    """

    hover_color = HOVER_COLOR

    def __init__(self, background: Any = BGR_COLOR, name: str = None,
                 orientation: str = 'horizontal', slider_width: int = 1,
                 z_plot: bool = False, **kwargs: dict) -> None:
        """
        Initialize simple 1D plot panel.

        :param background: color of the background, can be any pyqtgraph
                           compatible color specifier
        :param orientation: orientation of the plot. Default is '`horizontal`'
        :param slider_width: width of the draggable slider
        :param z_plot: if :py:obj:`True`, plot is a main energy plot of the
                       :class:`~_3Dviewer.DataViewer3D` GUI
        :param kwargs: kwargs passed to parent :class:`~pyqtgraph.PlotWidget`
        """

        super().__init__(background=background, **kwargs)

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

        # Whether to allow changing the sliders width with arrow keys
        self.change_width_enabled = False

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

        # The position of the sliders is stored with a TracedVariable
        initial_pos = 0
        # pos = TracedVariable(initial_pos, name='pos')
        pos = CustomTracedVariable(initial_pos, name='pos')
        self.register_traced_variable(pos)

        # Set up the sliders
        # self.slider_width = TracedVariable(
        #     slider_width, name='{}.slider_width'.format(self.name))
        self.slider = pg.InfiniteLine(
            initial_pos, movable=True, angle=self.angle)
        self.set_slider_pen(color=BASE_LINECOLOR, width=slider_width)

        # Add a marker. Args are (style, position (from 0-1), size #NOTE
        # seems broken
        self.addItem(self.slider)

        # Disable mouse scrolling, panning and zooming for both axes
        self.setMouseEnabled(False, False)

        # Initialize range to [0, 1]
        self.set_bounds(initial_pos, initial_pos + 1)

        # Connect a slot (callback) to dragging and clicking events
        self.slider.sigDragged.connect(self.on_position_change)

    def get_data(self) -> tuple:
        """
        Get the currently displayed data as a tuple of arrays.

        :return: (`x_data`, `y_data`)
        """

        pdi = self.listDataItems()[0]
        return pdi.getData()

    def orientate(self) -> None:
        """
        Configure plot's layout depending on its orientation.
        """

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

    def register_traced_variable(self,
                                 traced_variable: TracedVariable) -> None:
        """
        Register :class:`CustomTracedVariable` connected to the slider.
        """

        self.pos = traced_variable
        self.pos.sig_value_changed.connect(self.set_position)
        self.pos.sig_allowed_values_changed.connect(
            self.on_allowed_values_change)

    def on_position_change(self) -> None:
        """
        Callback for dragging slider changing value of the connected
        :class:`CustomTracedVariable`.
        """

        current_pos = self.slider.value()
        # NOTE pos.set_value emits signal sig_value_changed which may lead to
        # duplicate processing of the position change.
        self.pos.set_value(current_pos)

        # if it's an energy plot, and binning option is active,
        # update also binning boundaries
        if self.binning and self.z_plot:
            z_pos = self.slider.value()
            self.left_line.setValue(z_pos - self.width)
            self.right_line.setValue(z_pos + self.width)

    def on_allowed_values_change(self) -> None:
        """
        Set new momentum slider bounds after changing allowed values of the
        connected :class:`CustomTracedVariable`, *e.g.* after setting binning
        lines.
        """

        # If the allowed values were reset, just exit
        if self.pos.allowed_values is None:
            return

        lower = self.pos.min_allowed - self.width
        upper = self.pos.max_allowed + self.width
        self.set_bounds(lower, upper)

    def set_position(self) -> None:
        """
        Set the position of the sliders whenever the value of connected
        :class:`CustomTracedVariable` has changed.
        """

        new_pos = self.pos.get_value()
        self.slider.setValue(new_pos)

    def add_binning_lines(self, pos: int, width: int) -> None:
        """
        Add not-movable lines around the specified sliders. The lines indicate
        integration area for a corresponding cut.

        :param pos: position of the slider
        :param width: number of left and right steps from the slider position
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
        self.left_line = pg.InfiniteLine(pos-width, movable=False,
                                         angle=self.angle)
        self.right_line = pg.InfiniteLine(pos+width, movable=False,
                                          angle=self.angle)
        self.left_line.setPen(color=BINLINES_LINECOLOR, width=1)
        self.right_line.setPen(color=BINLINES_LINECOLOR, width=1)
        self.addItem(self.left_line)
        self.addItem(self.right_line)

    def remove_binning_lines(self) -> None:
        """
        Remove binning lines from the :class:`ImagePlot`.
        """

        self.binning = False
        self.width = 0
        self.removeItem(self.left_line)
        self.removeItem(self.right_line)

    def set_bounds(self, lower: int, upper: int) -> None:
        """
        Set both, the displayed area of the axes and the range in which
        sliders can be dragged.

        :param lower: lowest value
        :param upper: highest value
        """

        if self.orientation == 'horizontal':
            self.setXRange(lower, upper, padding=0.0)
        else:
            self.setYRange(lower, upper, padding=0.0)
        self.slider.setBounds([lower, upper])

    def set_secondary_axis(self, min_val: float, max_val: float) -> None:
        """
        Set/update a second `x`-axis on the top.

        :param min_val: lowest axis' value
        :param max_val: highest axis' value
        """

        # Get a handle on the underlying plotItem
        plotItem = self.plotItem

        # Remove the old top-axis
        plotItem.layout.removeItem(plotItem.getAxis(self.secondary_axis))
        # Create the new axis and set its range
        new_axis = pg.AxisItem(orientation=self.secondary_axis)
        new_axis.setRange(min_val, max_val)
        # Attach it internally to the plotItem and its layout (The arguments
        plotItem.axes[self.secondary_axis]['item'] = new_axis
        plotItem.layout.addItem(new_axis, *self.secondary_axis_grid)

    def set_ticks(self, min_val: float, max_val: float, axis: str) -> None:
        """
        Set customized axis' ticks to reflect the dimensions of the physical
        data.

        :param min_val: lowest axis' value
        :param max_val: highest axis' value
        :param axis: concerned axis, can be [`bottom`, `left`, *etc*]
        """

        plotItem = self.plotItem

        # Remove the old top-axis
        plotItem.layout.removeItem(plotItem.getAxis(axis))
        # Create the new axis and set its range
        new_axis = pg.AxisItem(orientation=axis)
        new_axis.setRange(min_val, max_val)
        # Attach it internally to the plotItem and its layout (The arguments
        plotItem.axes[axis]['item'] = new_axis
        plotItem.layout.addItem(new_axis, *self.main_xaxis_grid)

    def set_slider_pen(self, color: Any = None, width: int = None,
                       hover_color: Any = None) -> None:
        """
        Set a color and thickness of the slider.

        :param color: color of the slider, can be any pyqtgraph compatible
                      color specifier
        :param width: width of the slider
        :param hover_color: hover color of the slider, can be any pyqtgraph
                            compatible color specifier
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


class CustomTracedVariable(TracedVariable):
    """
    Wrapper around :class:`data_slicer.imageplot.TracedVariable`
    (see for more details).

    In short, object taking care of traced variables using signalling
    mechanism.
    """

    def __init__(self, value: Any = None, name: str = None) -> None:
        """
        Initialize traced variable.

        :param value: initial value of the variable
        :param name: name of the variable
        """

        super(CustomTracedVariable, self).__init__(value=value, name=name)

    def _set_allowed_values(self, values: Any = None,
                            binning: int = 0) -> None:
        """
        Wrapper method to set values taking possible binning into account.

        :param values: new allowed values within which `traced_variable`
                       can be changed
        :param binning: number of applied bins, which additionally limit
                        allowed values from both sides
        """

        if binning == 0:
            values = np.array(values)
        else:
            values = np.array(values)[binning:-binning]

        self.set_allowed_values(values)