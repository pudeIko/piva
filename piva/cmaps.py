import matplotlib.colors
import numpy as np
from matplotlib import colormaps as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import colormaps
from pyqtgraph import ColorMap
from typing import Any, Union


class PIVAColorMap(ColorMap):
    """
    Simple subclass of :class:`pyqtgraph.ColorMap<pyqtgraph.ColorMap>`.
    Adds ``vmax`` and powerlaw normalization ``gamma``.
    """

    def __init__(self, pos: Any, color: Any,
                 gamma: float = 1., **kwargs: dict) -> None:
        super().__init__(pos, color, **kwargs)

        # Initialize instance variables
        self.alpha = 1
        self.vmax = 1
        self.gamma = 1

        # Retain a copy of the originally given positions
        self.original_pos = self.pos.copy()
        # Apply the powerlaw-norm
        self.set_gamma(gamma)

    def apply_transformations(self) -> None:
        """
        Recalculate the positions where the colormapping is defined by
        applying (in sequence) alpha, then a linear map to the range 
        [0, ``vmax``] and finally the powerlaw scaling: ::

            pos = pos^gamma
        """
        
        # Reset the cache in pyqtgraph.Colormap
        self.stopsCache = dict()

        # Apply alpha
        self.color[:, -1] = self.alpha

        # Linearly transform color values to the new range
        old_max = self.original_pos.max()
        old_min = self.original_pos.min()
        new_max = old_max * self.vmax
        m = (new_max - old_min) / (old_max - old_min)
        self.pos = m * (self.original_pos - old_max) + new_max

        # Apply a powerlaw norm to the positions
        self.pos = self.pos**(1 / self.gamma)

    def set_gamma(self, gamma: float = 1.) -> None:
        """
        Set the exponent for the power-law norm (``gamma``) that maps the
        colors to values. The values where the colours are defined are mapped
        like: ::

            y = x^gamma
        """
        
        self.gamma = gamma
        self.apply_transformations()

    def set_alpha(self, alpha: float) -> None:
        """ 
        Set the value of alpha for the whole colormap to ``alpha`` where
        ``alpha`` can be a float or an array of length ``len(self.color)``.
        """
        
        self.alpha = alpha
        self.apply_transformations()

    def set_vmax(self, vmax: float = 1) -> None:
        """
        Set the relative (to the maximum of the data) maximum of the 
        colorscale. 
        """
        
        self.vmax = vmax
        self.apply_transformations()


def convert_matplotlib_to_pyqtgraph(
        matplotlib_cmap: Union[str, matplotlib.colors.LinearSegmentedColormap,
                               matplotlib.colors.ListedColormap],
        alpha: float = 0.5) -> ColorMap:
    """
    Converts ``ColorMap`` object from :mod:`matplotlib` format to
    :mod:`pyqtgraph<pyqtgraph>` format.

    :param matplotlib_cmap: represents the name of a :mod:`matplotlib` colormap
    :param alpha: transparency value to be assigned to the whole cmap.
                  Default = 1.
    :return: corresponding colormap object in :mod:`pyqtgrqph`
    """

    # Get the colormap object if a colormap name is given 
    if isinstance(matplotlib_cmap, str):
        matplotlib_cmap = cm[matplotlib_cmap]
    # Number of entries in the matplotlib colormap
    N = matplotlib_cmap.N
    # Create the mapping values in the interval [0, 1]
    values = np.linspace(0, 1, N)
    # Extract RGBA values from the matplotlib cmap
    indices = np.arange(N)
    rgba = matplotlib_cmap(indices)
    # Apply alpha
    rgba[:, -1] = alpha
    # Convert to range 0-255
    rgba *= 255

    return PIVAColorMap(values, rgba)


def convert_piva_to_matplotlib(piva_cmap: PIVAColorMap,
                               cmap_name: str = 'converted_cmap') -> \
        matplotlib.colors.LinearSegmentedColormap:
    """
    Create a :mod:`matplotlib` colormap from a
    :class:`PIVAColorMap<piva.cmaps.PIVAColorMap>` instance.

    :param piva_cmap: colormap in a :mod:`piva` format
    :param cmap_name: optional name for the created cmap
    :return: colormap in a :mod:`matplotlib` format
    """

    # Reset the transformations - matplotlib can take care of them itself
    piva_cmap.set_gamma(1)
    piva_cmap.set_vmax(1)
    # Convert the colors from the range [0-255] to [0-1]
    colors = piva_cmap.color / 255
    N = len(colors)
    # Create the matplotlib colormap
    matplotlib_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N)
    return matplotlib_cmap


# +-------------------+ #
# | Prepare colormaps | # =====================================================
# +-------------------+ #

# list of selected matplotlib colormaps
my_cmaps = ['Blues', 'BrBG', 'BuGn', 'CMRmap', 'GnBu', 'Greens', 'Oranges',
            'PuRd', 'Purples', 'RdBu', 'RdPu', 'Reds', 'Spectral', 'YlOrRd',
            'afmhot', 'binary', 'bone', 'bwr', 'cividis', 'coolwarm', 'copper',
            'cubehelix', 'gist_earth', 'gist_heat', 'gnuplot', 'gnuplot2',
            'hot', 'inferno', 'jet', 'magma', 'pink', 'plasma', 'terrain',
            'turbo', 'twilight', 'viridis']


# Convert all matplotlib colormaps to pyqtgraph ones and make them available
# in the dict cmaps
cmaps = dict()
for name in colormaps():
    cmap = cm[name]
    cmaps.update({name: convert_matplotlib_to_pyqtgraph(cmap)})
