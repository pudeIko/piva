import numpy as np
import arpys_wp as wp
import tss_utilities as tss
import matplotlib as mpl
import pandas as pd
import scipy as spy
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from matplotlib import colors
import ipywidgets as ip
from ipywidgets import widgets as ipw
from cmaps import my_cmaps
from inspect import getfullargspec
import warnings

ignore_warnings = True
if ignore_warnings:
    warnings.filterwarnings('ignore', '.*invalid value encountered in power.*')
    warnings.filterwarnings('ignore', '.*Covariance of the parameters could not.*')
fwhm2sigma = 2 * np.sqrt(2 * np.log(2))

# +-----------------------------+ #
# | Interactive jupyter shit    | # ==============================================================
# +-----------------------------+ #


class Gap_Viewer:

    def __init__(self, data0, data1, fnames, viewer='fs'):

        self.data0 = data0
        self.data1 = data1
        self.fnames = fnames
        self.viewer = viewer

        self.panel = [{}, {}, {}]
        self.util_panel = Utilities_Panel(self)
        self.cell_width = self.cell_width = '98%'
        self.cell_layout = ipw.Layout(border='dashed 1px gray', margin='0% 0.5% 0.5% 0%', width=self.cell_width)
        self.style = {'description_width': 'initial', 'readout_color': 'red'}

        if self.viewer == 'fs':
            self.fs1 = None
            self.fs2 = None
            self.setup_fs_panels()
        elif self.viewer == 'bm':
            self.bm1 = None
            self.bm2 = None
            self.setup_bm_panels()
        self.edc = EDC_Cell(self)

        self.main_panel = ipw.HBox([self.panel[0]['widget'], self.panel[1]['widget'], self.panel[2]['widget']],
                                   layout=ip.Layout(width='100%'))

        self.whole_panel = ipw.VBox([self.util_panel.tab, self.main_panel], layout=ip.Layout(width='100%'))

    def sync_second_panel(self, val):
        panel = self.panel
        if self.viewer == 'fs':
            if self.util_panel.change_simul.value:
                panel[1]['energy'].value = panel[0]['energy'].value
                panel[1]['integrate_e'].value = panel[0]['integrate_e'].value
                panel[1]['kx'].value = panel[0]['kx'].value
                panel[1]['ky'].value = panel[0]['ky'].value
                panel[1]['integrate_k'].value = panel[0]['integrate_k'].value
        elif self.viewer == 'bm':
            if self.util_panel.change_simul.value:
                panel[1]['k'].value = panel[0]['k'].value
                panel[1]['integrate_k'].value = panel[0]['integrate_k'].value

    def set_k_steps(self, val):
        panel = self.panel
        if self.viewer == 'fs':
            if val:
                panel[0]['ky'].step = panel[0]['kx'].step
                panel[1]['ky'].step = panel[1]['kx'].step
            else:
                panel[0]['ky'].step = np.abs(self.fs1.data.yscale[0] - self.fs1.data.yscale[1])
                panel[1]['ky'].step = np.abs(self.fs2.data.yscale[0] - self.fs2.data.yscale[1])
        elif self.viewer == 'bm':
            if val:
                panel[0]['ky'].step = panel[0]['kx'].step
                panel[1]['ky'].step = panel[1]['kx'].step
            else:
                panel[0]['ky'].step = np.abs(self.bm1.data.yscale[0] - self.bm1.data.yscale[1])
                panel[1]['ky'].step = np.abs(self.bm2.data.yscale[0] - self.bm1.data.yscale[1])

    def setup_fs_panels(self):
        self.fs1 = FS_Cell(self, self.data0, self.panel[0], self.fnames[0])
        self.fs2 = FS_Cell(self, self.data1, self.panel[1], self.fnames[1])
        self.panel[0]['integrate_e'].observe(self.sync_second_panel, 'value')
        self.panel[0]['kx'].observe(self.sync_second_panel, 'value')
        self.panel[0]['ky'].observe(self.sync_second_panel, 'value')
        self.panel[0]['integrate_k'].observe(self.sync_second_panel, 'value')
        self.set_k_steps(self.util_panel.sync_k_steps.value)
        self.util_panel.change_simul.observe(self.sync_second_panel, 'value')
        self.util_panel.sync_k_steps.observe(self.set_k_steps, 'value')

    def setup_bm_panels(self):
        self.bm1 = BM_Cell(self, self.data0, self.panel[0], self.fnames[0])
        self.bm2 = BM_Cell(self, self.data1, self.panel[1], self.fnames[1])
        self.panel[0]['k'].observe(self.sync_second_panel, 'value')
        self.panel[0]['integrate_k'].observe(self.sync_second_panel, 'value')
        self.util_panel.change_simul.observe(self.sync_second_panel, 'value')


class MDC_Fitter:

    def __init__(self, data, fname, viewer='mdc'):

        self.data = data
        self.fname = fname
        self.viewer = viewer

        self.style = {'description_width': 'initial', 'readout_color': 'red'}

        self.panel = [{}, {}]
        self.util_panel = Utilities_Panel(self, viewer=viewer)
        self.setup_bm_panel()
        self.mdc = MDC_Cell(self)

        self.main_panel = ipw.GridspecLayout(1, 2)
        self.main_panel[0, 0] = self.panel[0]['widget']
        self.main_panel[0, 1] = self.panel[1]['widget']

        self.whole_panel = ipw.VBox([self.util_panel.tab, self.main_panel], layout=ip.Layout(width='100%'))

    def setup_bm_panel(self):
        self.bm = BM_Cell(self, self.data, self.panel[0], self.fname)


class Utilities_Panel:

    def __init__(self, main_window, viewer='edc'):
        self.mw = main_window
        self.viewer = viewer
        self.style = {'description_width': 'initial', 'readout_color': 'red'}

        self.set_panel()

    def set_panel(self):

        self.tab = ipw.Tab()
        self.set_im_tab()

        if self.viewer == 'edc':
            self.set_anal_tab_4edc()
            self.tab.children = [self.util_panel_ana, self.im_tab]
            self.tab.set_title(1, 'image')
            self.tab.set_title(0, 'analysis')
        elif self.viewer == 'mdc':
            self.set_bgr_tab()
            self.set_fit_mdc_tab()
            self.tab.children = [self.fit_mdc_tab, self.bgr_tab, self.im_tab]
            self.tab.set_title(1, 'analysis (bgr)')
            self.tab.set_title(0, 'analysis (fit)')
            self.tab.set_title(2, 'image')

    def set_im_tab(self):
        tab_layout = ipw.Layout(border='dashed 1px gray', margin='0% 0.5% 0.5% 0%')
        self.cmap_selector = ipw.Dropdown(options=my_cmaps, value='magma', description='colormap:',
                                          layout=ipw.Layout(height="auto", width="auto"))
        self.gamma_selector = ipw.FloatText(value=1, min=0.1, max=2, step=0.05, description='gamma:',
                                            layout=ipw.Layout(height="auto", width="auto"))
        self.dummy_selector = ipw.FloatText(value=1, min=0.1, max=2, step=0.05, description='dummy:',
                                            layout=ipw.Layout(height="auto", width="auto"))

        self.font_size_selector = ipw.IntText(value=12, min=3, max=48, step=1, description='fontsize:',
                                              layout=ipw.Layout(height="auto", width="auto"))
        self.fig_size_selector = ipw.IntText(value=9, min=3, max=20, step=1, description='figsize:',
                                             layout=ipw.Layout(height="auto", width="auto"))

        self.change_simul = ipw.Checkbox(value=True, description='sync_cuts', indent=False)

        im_grid = ipw.GridspecLayout(3, 5, layout=tab_layout)
        im_grid[0, 0] = self.cmap_selector
        im_grid[1, 0] = self.gamma_selector
        im_grid[0, 1] = self.dummy_selector
        im_grid[1, 1] = self.font_size_selector
        im_grid[0, 2] = self.fig_size_selector
        im_grid[1, 2] = self.change_simul

        self.im_tab = im_grid

    # edc
    def set_anal_tab_4edc(self):
        # util_cell_layout = self.util_cell_layout
        methods = ['direct', 'midpoint', 'symmetrize_@Ef', 'symmetrize', 'DFD', '1D_curvature']
        self.method_selector = ipw.Dropdown(options=methods, value=methods[0], description='method:',
                                            layout=ipw.Layout(height="auto", width="auto"))
        self.smooth = ipw.IntText(value=5, min=0, max=50, step=1, description='smooth:', layout=ipw.Layout(height="auto", width="auto"))

        self.dfd_res = ipw.FloatText(value=0.01, min=0., max=2, step=0.0001, description='resolution:',
                                     layout=ipw.Layout(height="auto", width="auto"))
        self.dfd_Ef = ipw.FloatText(value=0.0, min=-0.1, max=0.1, step=0.0001, description='Ef correction:',
                                    layout=ipw.Layout(height="auto", width="auto"))
        self.dfd_fd_cutoff = ipw.FloatText(value=0.0, min=-0.1, max=0.1, step=0.0001, description='FD cutoff:',
                                           layout=ipw.Layout(height="auto", width="auto"))

        self.span_equaly = ipw.Checkbox(value=True, description='span_equal', layout=ipw.Layout(height="auto", width="auto"), indent=False)
        self.sync_k_steps = ipw.Checkbox(value=True, description='equal_k_steps', layout=ipw.Layout(height="auto", width="auto"), indent=False)

        self.util_panel_ana = ipw.HBox([ipw.VBox([self.method_selector, self.smooth]),
                                        ipw.VBox([self.sync_k_steps, self.span_equaly]),
                                        ipw.VBox([self.dfd_res, self.dfd_Ef, self.dfd_fd_cutoff])],
                                       layout=ipw.Layout(height="auto", width="auto"))

    #mdc
    def set_bgr_tab(self):
        tab_layout = ipw.Layout(border='dashed 1px gray', margin='0% 0.5% 0.5% 0%')
        grid = ipw.GridspecLayout(3, 6, layout=tab_layout)
        self.range_selector = ipw.FloatRangeSlider(
            # value=[self.mw.data.yscale.min(), self.mw.data.yscale.max()],
            value=[-0.828, -0.229],
            min=self.mw.data.yscale.min(),
            max=self.mw.data.yscale.max(),
            step=wp.get_step(self.mw.data.yscale),
            description='range:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout_format='.3f',
            layout=ipw.Layout(height="auto", width="auto")
        )
        self.smooth_n = ipw.IntText(value=5, min=0, max=20, step=1, description='smooth (n):',
                                    layout=ipw.Layout(height="auto", width="auto"))
        self.smooth_rl = ipw.IntText(value=5, min=0, max=50, step=1, description='smooth (rl):',
                                     layout=ipw.Layout(height="auto", width="auto"))
        self.bgr_poly_order = ipw.IntText(value=1, min=0, max=10, step=1, description='bgr order:',
                                          layout=ipw.Layout(height="auto", width="auto"))
        self.bgr_range = ipw.FloatSlider(value=.25, min=0, max=1, step=0.05, description='bgr range (%):',
                                         layout=ipw.Layout(height="auto", width="auto"))

        grid[0, :2] = self.range_selector
        grid[1, 0] = self.smooth_n
        grid[1, 1] = self.smooth_rl
        grid[2, 0] = self.bgr_poly_order
        grid[2, 1] = self.bgr_range
        self.bgr_tab = grid

    def set_fit_mdc_tab(self):
        tab_layout = ipw.Layout(border='dashed 1px gray', margin='0% 0.5% 0.5% 0%')
        grid = ipw.GridspecLayout(3, 6, layout=tab_layout)
        self.lor_lbl = ipw.Label(value='Lorentzian:')
        self.fit_a0 = ipw.FloatText(value=1e3, min=0, max=1e10, step=0.01, description=r'\(a_0\):',
                                    layout=ipw.Layout(height="auto", width="auto"))
        self.fit_mu = ipw.FloatText(value=0, min=-20, max=20, step=0.001, description=r'\(\mu\):',
                                    layout=ipw.Layout(height="auto", width="auto"))
        self.fit_gamma = ipw.FloatText(value=1, min=0, max=5, step=0.001, description=r'\(\Gamma\):',
                                       layout=ipw.Layout(height="auto", width="auto"))
        self.fit_alpha = ipw.FloatText(value=1, min=0, max=10, step=0.001, description=r'\(\alpha\):',
                                       layout=ipw.Layout(height="auto", width="auto"))
        self.fit_beta = ipw.FloatText(value=1, min=0, max=10, step=0.001, description=r'\(\beta\):',
                                      layout=ipw.Layout(height="auto", width="auto"))
        self.fit_errs = ipw.Checkbox(value=False, description='fit_errs', indent=False)

        self.gauss_lbl = ipw.Label(value='Resolution:')
        self.k_resol = ipw.FloatText(value=0.0, min=0, max=10, step=0.0001, description='FWHM:',
                                     layout=ipw.Layout(height="auto", width="auto"))

        grid[0, 0] = self.lor_lbl
        grid[0, 1] = self.fit_a0
        grid[0, 2] = self.fit_mu
        grid[0, 3] = self.fit_gamma
        grid[1, 1] = self.fit_alpha
        grid[1, 2] = self.fit_beta
        grid[1, 3] = self.fit_errs
        grid[2, 0] = self.gauss_lbl
        grid[2, 1] = self.k_resol
        self.fit_mdc_tab = grid


class FS_Cell:

    def __init__(self, main_window, data, cell, fname):

        # add kinetic energy scales
        wp.add_kinetic_factor(data)
        self.mw = main_window
        self.data = data
        self.cell = cell
        self.fname = fname

        self.set_cell()

    def set_cell(self):

        cell = self.cell
        data = self.data
        style = self.mw.style
        cell['energy'] = ipw.FloatSlider(
            value=data.zscale[wp.indexof(0, data.zscale)],
            min=np.min(data.zscale),
            max=np.max(data.zscale),
            step=np.abs(data.zscale[0] - data.zscale[1]),
            description='Energy:',
            continuous_update=False,
            layout=ip.Layout(width='95%'),
            readout=False,
            style=style)

        cell['energy_lbl'] = ipw.Label(
            value=str(wp.indexof(0, data.zscale)),
            continuous_update=False,
            style=style)
        cell['e_ro'] = ipw.Label(
            value='{:.5f}'.format(cell['energy'].value),
            continuous_update=False)
        cell['energy'].observe(self.update_e_label, 'value')

        cell['integrate_e'] = ipw.IntSlider(
            value=0,
            min=0,
            max=10,
            step=1,
            description='Integration (Eb)',
            layout=ip.Layout(width='95%'),
            continuous_update=False,
            readout=False,
            style=style)

        cell['integrate_e_lbl'] = ipw.Label(
            value='{:.5f}'.format((2 * cell['integrate_e'].value) * wp.get_step(data.zscale)),
            continuous_update=False,
            style=style)
        cell['ie_ro'] = ipw.Label(
            value=str(cell['integrate_e'].value),
            continuous_update=False)
        cell['integrate_e'].observe(self.update_ie_label, 'value')

        cell['kx'] = ipw.FloatSlider(
            value=0,
            min=data.xscale.min(),
            max=data.xscale.max(),
            step=np.abs(data.xscale[0] - data.xscale[1]),
            description='kx:',
            continuous_update=False,
            layout=ip.Layout(width='95%'),
            readout=False,
            style=style)

        cell['kx_lbl'] = ipw.Label(
            value=str(wp.indexof(0, data.xscale)),
            continuous_update=False,
            style=style)
        cell['kx_ro'] = ipw.Label(
            value='{:.5f}'.format(cell['kx'].value),
            continuous_update=False)
        cell['kx'].observe(self.update_kx_label, 'value')

        cell['ky'] = ipw.FloatSlider(
            value=0.21,
            min=data.yscale.min(),
            max=data.yscale.max(),
            step=np.abs(data.yscale[0] - data.yscale[1]),
            description='ky:',
            continuous_update=False,
            layout=ip.Layout(width='95%'),
            readout=False,
            style=style)

        cell['ky_lbl'] = ipw.Label(
            value=str(wp.indexof(0, data.yscale)),
            continuous_update=False,
            style=style)
        cell['ky_ro'] = ipw.Label(
            value='{:.5f}'.format(cell['ky'].value),
            continuous_update=False)
        cell['ky'].observe(self.update_ky_label, 'value')

        cell['integrate_k'] = ipw.IntSlider(
            value=3,
            min=0,
            max=10,
            step=1,
            description='Integration (k)',
            layout=ip.Layout(width='95%'),
            continuous_update=False,
            readout=False,
            style=style)

        cell['integrate_k_lbl'] = ipw.Label(
            value='{:.5f}'.format((2 * cell['integrate_k'].value) * wp.get_step(data.xscale)),
            continuous_update=False,
            style=style)
        cell['ik_ro'] = ipw.Label(
            value=str(cell['integrate_k'].value),
            continuous_update=False)
        cell['integrate_k'].observe(self.update_ik_label, 'value')

        cell['output'] = ipw.interactive_output(self.get_cut, {
            'energy': cell['energy'],
            'integrate_e': cell['integrate_e'],
            'kx': cell['kx'],
            'ky': cell['ky'],
            'integrate_k': cell['integrate_k'],
            'cmap': self.mw.util_panel.cmap_selector,
            'gamma': self.mw.util_panel.gamma_selector,
            'fontsize': self.mw.util_panel.font_size_selector,
            'figsize': self.mw.util_panel.fig_size_selector
        })
        cell['widget'] = ipw.VBox([cell['output'],
                                   ipw.HBox([cell['energy'], cell['e_ro'], cell['energy_lbl']]),
                                   ipw.HBox([cell['integrate_e'], cell['ie_ro'], cell['integrate_e_lbl']]),
                                   ipw.HBox([cell['kx'], cell['kx_ro'], cell['kx_lbl']]),
                                   ipw.HBox([cell['ky'], cell['ky_ro'], cell['ky_lbl']]),
                                   ipw.HBox([cell['integrate_k'], cell['ik_ro'], cell['integrate_k_lbl']])],
                                  layout=self.mw.cell_layout)

    def get_cut(self, energy, integrate_e, kx, ky, integrate_k, cmap, gamma, fontsize, figsize):

        data = self.data
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        mpl.rcParams['font.size'] = fontsize
        mpl.rcParams['xtick.labelsize'] = fontsize
        mpl.rcParams['ytick.labelsize'] = fontsize

        e_idx = wp.indexof(energy, data.zscale)

        if integrate_e == 0:
            cut = data.data[:, :, e_idx]
        else:
            e_i, e_f = e_idx - integrate_e, e_idx + integrate_e
            if e_i < 0:
                e_i = 0
            if e_f > data.zscale.size:
                e_f = data.zscale.size
            cut = np.sum(data.data[:, :, e_i:e_f], axis=2)

        xx, yy = np.meshgrid(data.xscale, data.yscale)
        ax.pcolormesh(xx, yy, cut.T, cmap=cmap, norm=colors.PowerNorm(gamma=gamma))
        ax.scatter(kx, ky, c='g')
        if integrate_k != 0:
            ik = integrate_k
            dkx = np.abs(data.xscale[0] - data.xscale[1])
            min_kx = wp.indexof(kx - ik * dkx, data.xscale)
            max_kx = wp.indexof(kx + ik * dkx, data.xscale)
            min_ky = wp.indexof(ky - ik * dkx, data.yscale)
            max_ky = wp.indexof(ky + ik * dkx, data.yscale)

            min_kx = data.xscale[min_kx]
            max_kx = data.xscale[max_kx]
            min_ky = data.yscale[min_ky]
            max_ky = data.yscale[max_ky]
            ax.plot([min_kx, min_kx], [min_ky, max_ky], 'g--')
            ax.plot([min_kx, max_kx], [max_ky, max_ky], 'g--')
            ax.plot([max_kx, max_kx], [max_ky, min_ky], 'g--')
            ax.plot([max_kx, min_kx], [min_ky, min_ky], 'g--')

        ax.set_aspect('equal')
        ax.set_title(self.fname)
        plt.show()

    def update_e_label(self, val):
        new_lbl = wp.indexof(val['new'], self.data.zscale)
        self.cell['energy_lbl'].value = str(new_lbl)
        self.cell['e_ro'].value = '{:.5f}'.format(val['new'])

    def update_ie_label(self, val):
        new_lbl = (2 * val['new']) * wp.get_step(self.data.zscale)
        self.cell['integrate_e_lbl'].value = '{:.5f}'.format(new_lbl)
        self.cell['ie_ro'].value = str(val['new'])

    def update_kx_label(self, val):
        new_lbl = wp.indexof(val['new'], self.data.xscale)
        self.cell['kx_lbl'].value = str(new_lbl)
        self.cell['kx_ro'].value = '{:.5f}'.format(val['new'])

    def update_ky_label(self, val):
        new_lbl = wp.indexof(val['new'], self.data.yscale)
        self.cell['ky_lbl'].value = str(new_lbl)
        self.cell['ky_ro'].value = '{:.5f}'.format(val['new'])

    def update_ik_label(self, val):
        new_lbl = (2 * val['new']) * wp.get_step(self.data.xscale)
        self.cell['integrate_k_lbl'].value = '{:.5f}'.format(new_lbl)
        self.cell['ik_ro'].value = str(val['new'])


class BM_Cell:

    def __init__(self, main_window, data, cell, fname):

        # add kinetic energy scales
        wp.add_kinetic_factor(data)
        self.mw = main_window
        self.data = data
        self.cell = cell
        self.fname = fname
        self.cell_layout = ipw.Layout(border='dashed 1px gray', margin='0% 0.5% 0.5% 0%')
        # self.cell_width = self.mw.cell_width

        if self.mw.viewer == 'mdc':
            self.set_cell_4mdc()
        else:
            self.set_cell_4edc()

    def set_cell_4edc(self):

        cell = self.cell
        data = self.data
        style = self.mw.style
        cell['k'] = ipw.FloatSlider(
            value=0,
            min=data.yscale.min(),
            max=data.yscale.max(),
            step=wp.get_step(data.yscale),
            description='k:',
            layout=ip.Layout(width='95%'),
            readout=False,
            continuous_update=False,
            style=style)

        cell['k_lbl'] = ipw.Label(
            value=str(wp.indexof(0, data.yscale)),
            continuous_update=False)
        cell['k_ro'] = ipw.Label(
            value='{:.5f}'.format(cell['k'].value),
            continuous_update=False)
        cell['k'].observe(self.update_k_label, 'value')

        cell['integrate_k'] = ipw.IntSlider(
            value=3,
            min=0,
            max=10,
            step=1,
            description='Integration (k)',
            layout=ip.Layout(width='95%'),
            continuous_update=False,
            readout=False,
            style=style)

        cell['integrate_k_lbl'] = ipw.Label(
            value='{:.5f}'.format((2 * cell['integrate_k'].value) * wp.get_step(data.yscale)),
            continuous_update=False,
            style=style)
        cell['ik_ro'] = ipw.Label(
            value=str(cell['integrate_k'].value),
            continuous_update=False)
        cell['integrate_k'].observe(self.update_ik_label, 'value')

        cell['output'] = ipw.interactive_output(self.get_cut_4edc, {
            'k': cell['k'],
            'integrate_k': cell['integrate_k'],
            'cmap': self.mw.util_panel.cmap_selector,
            'gamma': self.mw.util_panel.gamma_selector,
            'fontsize': self.mw.util_panel.font_size_selector,
            'figsize': self.mw.util_panel.fig_size_selector
        })
        cell['widget'] = ipw.VBox([cell['output'],
                                   ipw.HBox([cell['k'], cell['k_ro'], cell['k_lbl']]),
                                   ipw.HBox([cell['integrate_k'],  cell['ik_ro'], cell['integrate_k_lbl']])],
                                  layout=self.mw.cell_layout)

    def get_cut_4edc(self, k, integrate_k, cmap, gamma, fontsize, figsize):

        data = self.data
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        mpl.rcParams['font.size'] = fontsize
        mpl.rcParams['xtick.labelsize'] = fontsize
        mpl.rcParams['ytick.labelsize'] = fontsize
        clr = 'g'

        kk, ee = np.meshgrid(data.yscale, data.zscale)
        cut = data.data[0, :, :].T
        k_bar = np.ones_like(data.zscale) * k
        ax.pcolormesh(kk, ee, cut, cmap=cmap, norm=colors.PowerNorm(gamma=gamma))
        ax.plot(k_bar, data.zscale, clr + '-')
        if integrate_k != 0:
            ikx = integrate_k
            dk = wp.get_step(data.yscale)
            low_kx = wp.indexof(k - ikx * dk, data.yscale)
            high_kx = wp.indexof(k + ikx * dk, data.yscale)

            low_kx = np.ones_like(data.zscale) * data.yscale[low_kx]
            high_kx = np.ones_like(data.zscale) * data.yscale[high_kx]
            ax.plot(low_kx, data.zscale, clr + '--')
            ax.plot(high_kx, data.zscale, clr + '--')

        ax.set_title(self.fname)
        plt.show()

    def set_cell_4mdc(self):

        cell = self.cell
        data = self.data
        style = self.mw.style
        cell['e'] = ipw.FloatSlider(
            value=0,
            min=data.zscale.min(),
            max=data.zscale.max(),
            step=wp.get_step(data.zscale),
            description='E:',
            layout=ip.Layout(height='95%', width='95%'),
            readout=False,
            continuous_update=False,
            orientation='vertical',
            style=style)

        cell['e_lbl'] = ipw.Label(
            value=str(wp.indexof(0, data.zscale)),
            continuous_update=False)
        cell['e_ro'] = ipw.Label(
            value='{:.5f}'.format(cell['e'].value),
            continuous_update=False)
        cell['e'].observe(self.update_e_label, 'value')

        cell['integrate_e'] = ipw.IntSlider(
            value=3,
            min=0,
            max=10,
            step=1,
            description='Integration (E)',
            layout=ip.Layout(width='95%'),
            continuous_update=False,
            readout=False,
            style=style)

        cell['integrate_e_lbl'] = ipw.Label(
            value='{:.5f}'.format((2 * cell['integrate_e'].value) * wp.get_step(data.zscale)),
            continuous_update=False,
            style=style)
        cell['ie_ro'] = ipw.Label(
            value=str(cell['integrate_e'].value),
            continuous_update=False)
        cell['integrate_e'].observe(self.update_ie_label, 'value')

        cell['empty'] = ipw.Label(
            value='',
            continuous_update=False)

        cell['output'] = ipw.interactive_output(self.get_cut_4mdc, {
            'e': cell['e'],
            'integrate_e': cell['integrate_e'],
            'cmap': self.mw.util_panel.cmap_selector,
            'gamma': self.mw.util_panel.gamma_selector,
            'fontsize': self.mw.util_panel.font_size_selector,
            'figsize': self.mw.util_panel.fig_size_selector
        })
        cell['widget'] = ipw.VBox([
            ipw.HBox([ipw.VBox([cell['e_lbl'], cell['e_ro'], cell['e']]), cell['output']]),
            ipw.HBox([cell['integrate_e'],  cell['ie_ro'], cell['integrate_e_lbl']])],
            layout=self.cell_layout)

    def get_cut_4mdc(self, e, integrate_e, cmap, gamma, fontsize, figsize):

        data = self.data
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        mpl.rcParams['font.size'] = fontsize
        mpl.rcParams['xtick.labelsize'] = fontsize
        mpl.rcParams['ytick.labelsize'] = fontsize
        clr = 'g'

        kk, ee = np.meshgrid(data.yscale, data.zscale)
        cut = data.data[0, :, :].T
        e_bar = np.ones_like(data.yscale) * e
        ax.pcolormesh(kk, ee, cut, cmap=cmap, norm=colors.PowerNorm(gamma=gamma))
        ax.plot(data.yscale, e_bar, clr + '-')
        if integrate_e != 0:
            ie = integrate_e
            de = wp.get_step(data.zscale)
            low_e_idx = wp.indexof(e - ie * de, data.zscale)
            high_e_idx = wp.indexof(e + ie * de, data.zscale)

            low_e = np.ones_like(data.yscale) * data.zscale[low_e_idx]
            high_e = np.ones_like(data.yscale) * data.zscale[high_e_idx]
            ax.plot(data.yscale, low_e, clr + '--')
            ax.plot(data.yscale, high_e, clr + '--')

        ax.set_title(self.fname)
        plt.show()

    def update_k_label(self, val):
        new_lbl = wp.indexof(val['new'], self.data.yscale)
        self.cell['k_lbl'].value = str(new_lbl)
        self.cell['k_ro'].value = '{:.5f}'.format(val['new'])

    def update_ik_label(self, val):
        new_lbl = (2 * val['new']) * wp.get_step(self.data.yscale)
        self.cell['integrate_k_lbl'].value = '{:.5f}'.format(new_lbl)
        self.cell['ik_ro'].value = str(val['new'])

    def update_e_label(self, val):
        new_lbl = wp.indexof(val['new'], self.data.zscale)
        self.cell['e_lbl'].value = str(new_lbl)
        self.cell['e_ro'].value = '{:.5f}'.format(val['new'])

    def update_ie_label(self, val):
        new_lbl = (2 * val['new']) * wp.get_step(self.data.zscale)
        self.cell['integrate_e_lbl'].value = '{:.5f}'.format(new_lbl)
        self.cell['ie_ro'].value = str(val['new'])


class EDC_Cell:

    def __init__(self, main_window):
        self.mw = main_window

        self.set_edc_cell()

    def set_edc_cell(self):
        cell = self.mw.panel[2]
        if self.mw.viewer == 'fs':
            cell['output'] = ipw.interactive_output(self.get_edc, {
                'kx0': self.mw.panel[0]['kx'],
                'ky0': self.mw.panel[0]['ky'],
                'integrate_k0': self.mw.panel[0]['integrate_k'],
                'kx1': self.mw.panel[1]['kx'],
                'ky1': self.mw.panel[1]['ky'],
                'integrate_k1': self.mw.panel[1]['integrate_k'],
                'method': self.mw.util_panel.method_selector,
                'dfd_res': self.mw.util_panel.dfd_res,
                'dfd_Ef': self.mw.util_panel.dfd_Ef,
                'dfd_fd_cutoff': self.mw.util_panel.dfd_fd_cutoff,
                'span_equaly': self.mw.util_panel.span_equaly,
                'smooth': self.mw.util_panel.smooth,
                'fontsize': self.mw.util_panel.font_size_selector,
                'figsize': self.mw.util_panel.fig_size_selector,
            })
        elif self.mw.viewer == 'bm':
            cell['output'] = ipw.interactive_output(self.get_edc, {
                'k0': self.mw.panel[0]['k'],
                'integrate_k0': self.mw.panel[0]['integrate_k'],
                'k1': self.mw.panel[1]['k'],
                'integrate_k1': self.mw.panel[1]['integrate_k'],
                'method': self.mw.util_panel.method_selector,
                'dfd_res': self.mw.util_panel.dfd_res,
                'dfd_Ef': self.mw.util_panel.dfd_Ef,
                'dfd_fd_cutoff': self.mw.util_panel.dfd_fd_cutoff,
                'span_equaly': self.mw.util_panel.span_equaly,
                'smooth': self.mw.util_panel.smooth,
                'fontsize': self.mw.util_panel.font_size_selector,
                'figsize': self.mw.util_panel.fig_size_selector,
            })
        cell['widget'] = ipw.VBox([cell['output']], layout=self.mw.cell_layout)

    def get_edc(self, kx0=0, ky0=0, integrate_k0=0, kx1=0, ky1=0, integrate_k1=0,
                k0=0, k1=1, smooth=0, **kwargs):

        data0 = self.mw.data0
        data1 = self.mw.data1

        if self.mw.viewer == 'fs':
            kx0_idx = wp.indexof(kx0, data0.xscale)
            ky0_idx = wp.indexof(ky0, data0.yscale)
            kx1_idx = wp.indexof(kx1, data1.xscale)
            ky1_idx = wp.indexof(ky1, data1.yscale)
            if integrate_k0 == 0:
                edc0 = data0.data[kx0_idx, ky0_idx, :]
            else:
                edc0 = wp.sum_edcs_around_k(data0, kx0, ky0, integrate_k0)

            if integrate_k1 == 0:
                edc1 = data1.data[kx1_idx, ky1_idx, :]
            else:
                edc1 = wp.sum_edcs_around_k(data1, kx1, ky1, integrate_k1)
        elif self.mw.viewer == 'bm':
            k0_idx = wp.indexof(k0, data0.yscale)
            k1_idx = wp.indexof(k1, data1.yscale)

            if integrate_k0 == 0:
                edc0 = data0.data[0, k0_idx, :]
            else:
                ik0 = integrate_k0
                if (k0_idx - ik0) < 0:
                    k0_min = 0
                else:
                    k0_min = k0_idx - ik0
                if (k0_idx + ik0) > data0.yscale.size:
                    k0_max = data0.yscale.size
                else:
                    k0_max = k0_idx + ik0
                edc0 = np.sum(data0.data[0, k0_min:k0_max, :], axis=0)

            if integrate_k1 == 0:
                edc1 = data1.data[0, k1_idx, :]
            else:
                ik1 = integrate_k1
                if (k1_idx - ik1) < 0:
                    k1_min = 0
                else:
                    k1_min = k1_idx - ik1
                if (k1_idx + ik1) > data1.yscale.size:
                    k1_max = data1.yscale.size
                else:
                    k1_max = k1_idx + ik1
                edc1 = np.sum(data1.data[0, k1_min:k1_max, :], axis=0)

        # do smoothing
        if smooth != 0:
            edc0 = wp.smooth(edc0, recursion_level=smooth)
            edc1 = wp.smooth(edc1, recursion_level=smooth)

        self.data_treatment(edc0, edc1, **kwargs)

    def data_treatment(self, edc0, edc1, method='x', dfd_res=0, dfd_Ef=0, dfd_fd_cutoff=0, span_equaly=False, **kwargs):

        data0 = self.mw.data0
        data1 = self.mw.data1

        # set the method
        if method == 'direct':
            erg0 = data0.zscale
            erg1 = data1.zscale
        elif method == 'midpoint':
            erg0 = data0.zscale
            erg1 = data1.zscale
            mp0x, mp0y = wp.find_midpoint(edc0, erg0)
            mp1x, mp1y = wp.find_midpoint(edc1, erg1)
        elif method == 'symmetrize':
            edc0, erg0 = wp.symmetrize_edc(edc0, data0.zscale)
            edc1, erg1 = wp.symmetrize_edc(edc1, data1.zscale)
        elif method == 'symmetrize_@Ef':
            edc0, erg0 = wp.symmetrize_edc_around_Ef(edc0, data0.zscale)
            edc1, erg1 = wp.symmetrize_edc_around_Ef(edc1, data1.zscale)
        elif method == 'DFD':
            erg0 = data0.zscale
            erg1 = data1.zscale
            edc0 = wp.dec_fermi_div(edc0, erg0, dfd_res, dfd_Ef, dfd_fd_cutoff, T=10)
            edc1 = wp.dec_fermi_div(edc1, erg1, dfd_res, dfd_Ef, dfd_fd_cutoff, T=100)
        elif method == '1D_curvature':
            erg0, erg1 = data0.zscale, data1.zscale
            edc0 = wp.curvature_1d(edc0)
            edc1 = wp.curvature_1d(edc1)

        # transform energy axes back to binding, but synced
        # erg1 += (data1.ekin - data0.ekin)

        if span_equaly:
            edc0 -= edc0.min()
            edc1 -= edc1.min()

        self.plot_edc(edc0, edc1, erg0, erg1, **kwargs)

    def plot_edc(self, edc0, edc1, erg0, erg1, method='x', fontsize=10, figsize=10):

        fig, ax = plt.subplots(figsize=(figsize, (figsize * 0.75)))
        mpl.rcParams['font.size'] = fontsize
        mpl.rcParams['xtick.labelsize'] = fontsize
        mpl.rcParams['ytick.labelsize'] = fontsize

        # set the method
        if method == 'midpoint':
            mp0x, mp0y = wp.find_midpoint(edc0, erg0)
            mp1x, mp1y = wp.find_midpoint(edc1, erg1)
            ax.scatter(mp0x, mp0y / edc0.max())
            ax.scatter(mp1x, mp1y / edc1.max())

        # plot
        ax.plot(erg0, wp.normalize(edc0), label=self.mw.fnames[0])
        ax.plot(erg1, wp.normalize(edc1), label=self.mw.fnames[1])
        plt.legend()
        plt.title('Normalized EDCs')
        plt.show()


class MDC_Cell:

    def __init__(self, main_window):

        self.mw = main_window
        self.cell_layout = ipw.Layout(border='dashed 1px gray', margin='0% 0.5% 0.5% 0%')

        self.set_mdc_cell()

    def set_mdc_cell(self):
        cell = self.mw.panel[1]
        cell['output'] = ipw.interactive_output(self.get_mdc, {
            'e': self.mw.panel[0]['e'],
            'integrate_e': self.mw.panel[0]['integrate_e'],
            'smooth_n': self.mw.util_panel.smooth_n,
            'smooth_rl': self.mw.util_panel.smooth_rl,
            'fontsize': self.mw.util_panel.font_size_selector,
            'figsize': self.mw.util_panel.fig_size_selector,
            'shadow': self.mw.util_panel.range_selector,
            'bgr_order': self.mw.util_panel.bgr_poly_order,
            'bgr_range': self.mw.util_panel.bgr_range,
            'a0': self.mw.util_panel.fit_a0,
            'mu': self.mw.util_panel.fit_mu,
            'gamma': self.mw.util_panel.fit_gamma,
            'alpha': self.mw.util_panel.fit_alpha,
            'beta': self.mw.util_panel.fit_beta,
            'fit_errs': self.mw.util_panel.fit_errs,
            'k_resol': self.mw.util_panel.k_resol
        })
        cell['widget'] = ipw.VBox([cell['output']], layout=self.cell_layout)

    def get_mdc(self, e=0, integrate_e=0, smooth_n=5, smooth_rl=0, **kwargs):

        data = self.mw.data

        e_idx = wp.indexof(e, data.zscale)

        if integrate_e == 0:
            mdc = data.data[0, e_idx, :]
        else:
            ie = integrate_e
            if (e_idx - ie) < 0:
                e_min = 0
            else:
                e_min = e_idx - ie
            if (e_idx + ie) > data.zscale.size:
                e_max = data.zscale.size
            else:
                e_max = e_idx + ie
            mdc = np.sum(data.data[0, :, e_min:e_max], axis=1)

        # do smoothing
        if smooth_rl != 0:
            mdc = wp.smooth(mdc, n_box=smooth_n, recursion_level=smooth_rl)

        k_axis = self.mw.data.yscale
        self.plot_mdc(mdc, k_axis, e_idx, **kwargs)

    def plot_mdc(self, mdc, k_axis, e_idx, fontsize=10, figsize=10, shadow=None, **kwargs):

        fig, ax = plt.subplots(figsize=(figsize * 1.3, (figsize * 0.75)))
        mpl.rcParams['font.size'] = fontsize
        mpl.rcParams['xtick.labelsize'] = fontsize
        mpl.rcParams['ytick.labelsize'] = fontsize

        fit_p, fit_k = wp.indexof(shadow[0], k_axis), wp.indexof(shadow[1], k_axis)
        mdc_fit = mdc[fit_p:fit_k]
        k_fit = k_axis[fit_p:fit_k]

        bgr = self.fit_bgr(mdc_fit, k_fit, **kwargs)

        # self.wrapper(**kwargs)
        self.set_fit_fun(**kwargs)
        p, cov = curve_fit(self.fit_fun, k_fit, mdc_fit - bgr, p0=self.p0)
        self.print_results(p, np.sqrt(np.diag(cov)), **kwargs)

        res_func = lambda x: self.fit_fun(x, *p)
        fit = res_func(k_fit)

        # plot
        ax.plot(k_axis, mdc, label='data')
        ax.plot(k_fit, bgr, label='bgr')
        ax.plot(k_fit, fit + bgr, label='fit')
        ax.axvspan(shadow[0], shadow[1], facecolor='0.05', alpha=0.1)
        ax.set_title('MDC @E={:.4f}'.format(self.mw.data.zscale[e_idx]))
        ax.legend()
        plt.show()

        tmp = fit + bgr
        # chi2 = chisquare(f_obs=mdc_fit, f_exp=tmp, ddof=len(p))[1]
        # self.d['chi^2'] = '{:.4f}'.format(chi2)
        self.df = pd.DataFrame(data=self.d, index=[0])
        print(self.df)

    def fit_bgr(self, mdc, k, bgr_order=0, bgr_range=0.1, **kwargs):
        n = int(mdc.size * bgr_range)
        bgr_x = np.concatenate((k[:n], k[-n:]), axis=0)
        bgr_y = np.concatenate((mdc[:n], mdc[-n:]), axis=0)
        coefs_bgr = np.polyfit(bgr_x, bgr_y, bgr_order)
        bgr_fit = np.poly1d(coefs_bgr)(k)
        return bgr_fit

    def wrapper(self, a0, mu, gamma, alpha=1, beta=1, k_resol=0, **kwargs):
        to_fit = {}
        dont_fit = {'alpha': 1, 'beta': 1, 'resol': 0}
        p0 = [a0, mu, gamma]
        if alpha != 1:
            to_fit['alpha'] = alpha
            p0.append(alpha)
            del dont_fit['alpha']
        if beta != 1:
            to_fit['beta'] = beta
            p0.append(beta)
            del dont_fit['beta']
        if k_resol != 0:
            to_fit['resol'] = k_resol
            p0.append(k_resol)
            del dont_fit['resol']

        self.to_fit = to_fit
        self.dont_fit = dont_fit
        self.p0 = p0

    def set_fit_fun(self, a0=1e3, mu=1, gamma=0.1, alpha=1, beta=1, k_resol=0, **kwargs):#**to_fit):
        # to_fit = self.to_fit
        # print(**to_fit)
        # return wp.asym_lorentzian(x, a0, mu, gamma, **to_fit)#, **self.dont_fit)

        # def fit_fun(x, a0, mu, gamma, **to_fit):
        #     return wp.asym_lorentzian(x, a0, mu, gamma, **not_to_fit)

        # self.fit_fun = fit_fun(x, a0, mu, gamma, )
        # self.fit_fun = wp.asym_lorentzian(a0, mu, gamma, **to_fit)
        # if k_resol == 0:
        #     self.fit_fun = lambda x, a0, mu, gamma, alpha, beta:
        if k_resol == 0:
            if (alpha == 1.) and (beta == 1.):
                self.fit_fun = lambda x, a0, mu, gamma: \
                    wp.asym_lorentzian(x, a0, mu, gamma, alpha=alpha, beta=alpha)
                self.p0 = [a0, mu, gamma]
            elif (alpha == 1) and (beta != 1):
                self.fit_fun = lambda x, a0, mu, gamma, beta: \
                    wp.asym_lorentzian(x, a0, mu, gamma, alpha=alpha, beta=alpha)
                self.p0 = [a0, mu, gamma, beta]
            elif (alpha != 1) and (beta == 1):
                self.fit_fun = lambda x, a0, mu, gamma, alpha: \
                    wp.asym_lorentzian(x, a0, mu, gamma, alpha=alpha, beta=beta)
                self.p0 = [a0, mu, gamma, alpha]
            else:
                self.fit_fun = lambda x, a0, mu, gamma, alpha, beta: \
                    wp.asym_lorentzian(x, a0, mu, gamma, alpha=alpha, beta=alpha)
                self.p0 = [a0, mu, gamma, alpha, beta]
        else:
            k_resol /= fwhm2sigma
            if (alpha == 1.) and (beta == 1.):
                self.fit_fun = lambda x, a0, mu, gamma, k_resol: \
                    wp.asym_lorentzian(x, a0, mu, gamma, resol=k_resol)
                self.p0 = [a0, mu, gamma, k_resol]
            elif (alpha == 1) and (beta != 1):
                self.fit_fun = lambda x, a0, mu, gamma, beta, k_resol: \
                    wp.asym_lorentzian(x, a0, mu, gamma, beta=alpha, resol=k_resol)
                self.p0 = [a0, mu, gamma, beta, k_resol]
            elif (alpha != 1) and (beta == 1):
                self.fit_fun = lambda x, a0, mu, gamma, alpha, k_resol: \
                    wp.asym_lorentzian(x, a0, mu, gamma, alpha=alpha, resol=k_resol)
                self.p0 = [a0, mu, gamma, alpha, k_resol]
            else:
                self.fit_fun = lambda x, a0, mu, gamma, alpha, beta, k_resol: \
                    wp.asym_lorentzian(x, a0, mu, gamma, alpha=alpha, beta=alpha, resol=k_resol)
                self.p0 = [a0, mu, gamma, alpha, beta, k_resol]

    def print_results(self, p, cov, fit_errs=False, **kwargs):

        if fit_errs:
            d = {'a0': int(p[0]),
                 'd a0': '{:.2f}'.format(cov[0]),
                 'mu_0': '{:.3f}'.format(p[1]),
                 'd mu_0': '{:.3f}'.format(cov[1]),
                 'Gamma': '{:.3f}'.format(p[2]),
                 'd Gamma': '{:.3f}'.format(cov[2])
                 }
            if 'k_resol' in getfullargspec(self.fit_fun)[0]:
                idx = getfullargspec(self.fit_fun)[0].index('k_resol') - 1
                d['k_resol'] = '{:.4f}'.format(p[idx] * fwhm2sigma)
                d['d k_resol'] = '{:.4f}'.format(cov[idx] * fwhm2sigma)
            if 'alpha' in getfullargspec(self.fit_fun)[0]:
                idx = getfullargspec(self.fit_fun)[0].index('alpha') - 1
                d['alpha'] = '{:.2f}'.format(p[idx])
                d['d alpha'] = '{:.2f}'.format(cov[idx])
            if 'beta' in getfullargspec(self.fit_fun)[0]:
                idx = getfullargspec(self.fit_fun)[0].index('beta') - 1
                d['beta'] = '{:.2f}'.format(p[idx])
                d['d beta'] = '{:.2f}'.format(cov[idx])
        else:
            d = {'a0': int(p[0]),
                 'mu_0': '{:.3f}'.format(p[1]),
                 'Gamma': '{:.3f}'.format(p[2])
                 }
            if 'k_resol' in getfullargspec(self.fit_fun)[0]:
                idx = getfullargspec(self.fit_fun)[0].index('k_resol') - 1
                d['k_resol'] = '{:.4f}'.format(p[idx] * fwhm2sigma)
            if 'alpha' in getfullargspec(self.fit_fun)[0]:
                idx = getfullargspec(self.fit_fun)[0].index('alpha') - 1
                d['alpha'] = '{:.2f}'.format(p[idx])
            if 'beta' in getfullargspec(self.fit_fun)[0]:
                idx = getfullargspec(self.fit_fun)[0].index('beta') - 1
                d['beta'] = '{:.2f}'.format(p[idx])

        self.d = d
        # self.df = pd.DataFrame(data=d, index=[0])


def gap_viewer(data0, data1, fnames=None):
    if fnames is None:
        fnames = ['1', '2']

    if (data0.data[:, 0, 0].size > 1) and (data0.data[:, 0, 0].size > 1):
        gv = Gap_Viewer(data0, data1, fnames, viewer='fs')
    elif (data0.data[:, 0, 0].size == 1) and (data0.data[:, 0, 0].size == 1):
        gv = Gap_Viewer(data0, data1, fnames, viewer='bm')
    else:
        print('Cannot compare band map with Fermi surface.')
        return

    return gv.whole_panel


def mdc_viewer(data, fname=None):
    if fname is None:
        fname = '1'

    mdcv = MDC_Fitter(data, fname)

    return mdcv.whole_panel


def cut_viewer(data, fname=None, mac=True):
    panel = [{}, {}, {}]
    style = {'description_width': 'initial', 'readout_color': 'red'}

    # -------- Colorscale panel
    def get_const_e_map(data, energy, integrate_e, kx, ky, integrate_kx, integrate_ky, title, cmap, fontsize, figsize):
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        mpl.rcParams['font.size'] = fontsize
        mpl.rcParams['xtick.labelsize'] = fontsize
        mpl.rcParams['ytick.labelsize'] = fontsize

        e_idx = wp.indexof(energy, data.zscale)

        if integrate_e == 0:
            cut = data.data[:, :, e_idx]
        else:
            e_i, e_f = e_idx - integrate_e, e_idx + integrate_e
            if e_i < 0:
                e_i = 0
            if e_f > data.zscale.size:
                e_f = data.zscale.size
            cut = np.sum(data.data[:, :, e_i:e_f], axis=2)

        horiz = np.ones_like(data.xscale) * ky
        vert = np.ones_like(data.yscale) * kx

        xx, yy = np.meshgrid(data.xscale, data.yscale)
        ax.pcolormesh(xx, yy, cut.T, cmap=cmap)
        clr = 'g'
        ax.plot(data.xscale, horiz, clr)
        ax.plot(vert, data.yscale, clr + '--')

        if integrate_kx != 0:
            ikx = integrate_kx
            dkx = np.abs(data.xscale[0] - data.xscale[1])
            low_kx = wp.indexof(kx - ikx * dkx, data.xscale)
            high_kx = wp.indexof(kx + ikx * dkx, data.xscale)

            low_kx = np.ones_like(data.yscale) * data.xscale[low_kx]
            high_kx = np.ones_like(data.yscale) * data.xscale[high_kx]
            ax.plot(low_kx, data.yscale, clr + '--')
            ax.plot(high_kx, data.yscale, clr + '--')

        if integrate_ky != 0:
            iky = integrate_ky
            dky = np.abs(data.yscale[0] - data.yscale[1])
            low_ky = wp.indexof(ky - iky * dky, data.yscale)
            high_ky = wp.indexof(ky + iky * dky, data.yscale)

            low_ky = np.ones_like(data.xscale) * data.yscale[low_ky]
            high_ky = np.ones_like(data.xscale) * data.yscale[high_ky]
            ax.plot(data.xscale, low_ky, clr + '--')
            ax.plot(data.xscale, high_ky, clr + '--')

        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('Scanned axis')
        ax.set_ylabel('Analyzer axis')
        plt.show()

    def set_const_e_cell(data, fname):

        def update_e_label(val):
            new_lbl = wp.indexof(val['new'], data.zscale)
            cell['energy_lbl'].value = str(new_lbl)

        def update_kx_label(val):
            new_lbl = wp.indexof(val['new'], data.xscale)
            cell['kx_lbl'].value = str(new_lbl)

        def update_ky_label(val):
            new_lbl = wp.indexof(val['new'], data.yscale)
            cell['ky_lbl'].value = str(new_lbl)

        cell = panel[0]

        cell['energy'] = ipw.FloatSlider(
            value=data.zscale[wp.indexof(0, data.zscale)],
            min=np.min(data.zscale),
            max=np.max(data.zscale),
            step=np.abs(data.zscale[0] - data.zscale[1]),
            description='Energy:',
            continuous_update=False,
            layout=ip.Layout(width='95%'),
            readout_format='.5f',
            style=style)

        cell['energy_lbl'] = ipw.Label(
            value=str(wp.indexof(0, data.zscale)),
            continuous_update=False,
            style=style)
        cell['energy'].observe(update_e_label, 'value')

        cell['integrate_e'] = ipw.IntSlider(
            value=0,
            min=0,
            max=10,
            step=1,
            description='Integration (Eb)',
            layout=ip.Layout(width='85%'),
            continuous_update=False,
            style=style)

        cell['kx'] = ipw.FloatText(
            value=0,
            min=data.xscale.min(),
            max=data.xscale.max(),
            step=np.abs(data.xscale[0] - data.xscale[1]),
            description='kx:',
            continuous_update=False,
            layout=ip.Layout(width='50%'),
            style=style)

        cell['kx_lbl'] = ipw.Label(
            value=str(wp.indexof(0, data.xscale)),
            continuous_update=False,
            style=style)
        cell['kx'].observe(update_kx_label, 'value')

        cell['ky'] = ipw.FloatText(
            value=0.21,
            min=data.yscale.min(),
            max=data.yscale.max(),
            step=np.abs(data.yscale[0] - data.yscale[1]),
            description='ky:',
            continuous_update=False,
            layout=ip.Layout(width='50%'),
            style=style)

        cell['ky_lbl'] = ipw.Label(
            value=str(wp.indexof(0, data.yscale)),
            continuous_update=False,
            style=style)
        cell['ky'].observe(update_ky_label, 'value')

        cell['integrate_kx'] = ipw.IntSlider(
            value=3,
            min=0,
            max=10,
            step=1,
            description='Integration (kx)',
            layout=ip.Layout(width='85%'),
            continuous_update=False,
            style=style)

        cell['integrate_ky'] = ipw.IntSlider(
            value=3,
            min=0,
            max=10,
            step=1,
            description='Integration (ky)',
            layout=ip.Layout(width='85%'),
            continuous_update=False,
            style=style)

        cell['output'] = ipw.interactive_output(get_const_e_map, {
            'data': ip.fixed(data),
            'energy': cell['energy'],
            'integrate_e': cell['integrate_e'],
            'kx': cell['kx'],
            'ky': cell['ky'],
            'integrate_kx': cell['integrate_kx'],
            'integrate_ky': cell['integrate_ky'],
            'title': ip.fixed(fname),
            'cmap': cmap_selector,
            'fontsize': font_size_selector,
            'figsize': fig_size_selector
        })
        cell['widget'] = ipw.VBox([cell['output'],
                                   ipw.HBox([cell['energy'], cell['energy_lbl']], layout=ip.Layout(height='75px')),
                                   cell['integrate_e'],
                                   ipw.HBox([cell['kx'], cell['kx_lbl']], layout=ip.Layout(height='75px')),
                                   ipw.HBox([cell['ky'], cell['ky_lbl']], layout=ip.Layout(height='75px')),
                                   cell['integrate_kx'], cell['integrate_ky']], layout=cell_layout)

    def get_cut(data, axis, kx, ky, integrate_kx, integrate_ky, cmap, fontsize, figsize):

        fig, ax = plt.subplots(figsize=(figsize, figsize))
        mpl.rcParams['font.size'] = fontsize
        mpl.rcParams['xtick.labelsize'] = fontsize
        mpl.rcParams['ytick.labelsize'] = fontsize

        kx_idx = wp.indexof(kx, data.xscale)
        ky_idx = wp.indexof(ky, data.yscale)

        if axis == 'anal':
            ikx = integrate_kx
            cut = np.sum(data.data[kx_idx - ikx: kx_idx + ikx, :, :], axis=0).T
            k, e = np.meshgrid(data.yscale, data.zscale)
            k_bar = np.ones_like(data.zscale) * ky
            title = 'Analyzer axis'
        elif axis == 'scanned':
            iky = integrate_ky
            cut = np.sum(data.data[:, ky_idx - iky: ky_idx + iky, :], axis=1).T
            k, e = np.meshgrid(data.xscale, data.zscale)
            k_bar = np.ones_like(data.zscale) * kx
            title = 'Scanned axis'

        ax.pcolormesh(k, e, cut, cmap=cmap)
        ax.plot(k_bar, data.zscale)
        ax.set_title(title)
        plt.show()

    def set_cut_cell(data, cell, axis):

        cell['output'] = ipw.interactive_output(get_cut, {
            'data': ip.fixed(data),
            'axis': ip.fixed(axis),
            'kx': panel[0]['kx'],
            'ky': panel[0]['ky'],
            'integrate_kx': panel[0]['integrate_kx'],
            'integrate_ky': panel[0]['integrate_ky'],
            'cmap': cmap_selector,
            'fontsize': font_size_selector,
            'figsize': fig_size_selector
        })
        cell['widget'] = ipw.VBox([cell['output']], layout=cell_layout)

    def set_k_steps(val):
        if val:
            panel[0]['ky'].step = panel[0]['kx'].step
        else:
            panel[0]['ky'].step = np.abs(data.yscale[0] - data.yscale[1])

    if mac:
        cell_width = '98%'  # Mac
    else:
        cell_width = '98%'  # Windows

    util_cell_layout = ipw.Layout()

    cmap_selector = ipw.Dropdown(options=my_cmaps, value='magma', description='colormap:', layout=util_cell_layout)
    font_size_selector = ipw.IntText(value=12, min=3, max=48, step=1, description='fontsize:', layout=util_cell_layout)
    fig_size_selector = ipw.IntText(value=10, min=3, max=20, step=1, description='figsize:', layout=util_cell_layout)

    change_simul = ipw.Checkbox(value=True, description='sync:', layout=util_cell_layout)
    span_equaly = ipw.Checkbox(value=True, description='span_equal:', layout=util_cell_layout)
    smooth_edcs = ipw.Checkbox(value=True, description='smooth_edcs:', layout=util_cell_layout)
    sync_k_steps = ipw.Checkbox(value=True, description='equal_k_steps:', layout=util_cell_layout)
    sync_k_steps.observe(set_k_steps, 'value')

    util_panel_layout = ipw.Layout(border='dashed 1px gray', margin='0% 0% 1% 0%', width='100%', style=style,
                                   flex_flow='row')
    cell_layout = ipw.Layout(border='dashed 1px gray', margin='0% 0.5% 0.5% 0%', width=cell_width, height=cell_width)

    utilities_panel = ipw.HBox([cmap_selector, font_size_selector, fig_size_selector,
                                ipw.VBox([change_simul, span_equaly]),
                                ipw.VBox([smooth_edcs, sync_k_steps])],
                               layout=util_panel_layout)

    set_const_e_cell(data, fname)
    set_cut_cell(data, panel[1], 'anal')
    set_cut_cell(data, panel[2], 'scanned')
    set_k_steps(sync_k_steps.value)

    main_panel = ipw.HBox([panel[0]['widget'], panel[1]['widget'], panel[2]['widget']], layout=ip.Layout(width='100%'))

    whole_panel = ipw.VBox([utilities_panel, main_panel], layout=ip.Layout(width='100%'))

    return whole_panel


def model_viewer(data, fname=None, saved=True):
    panel = [{}, {}]
    style = {'description_width': 'initial', 'readout_color': 'red'}
    cuts = ['horizontal', 'vertical']
    models = ['li2018_d', 'li2018_b', 'rossnagel2005_d', 'rossnagel2005_b', 'inosov2008_d', 'inosov2008_b']
    if saved:
        energy_ax = data.zscale - data.saved['hv'] + data.saved['wf']
        scanned_ax = data.saved['kx']
        slit_ax = data.saved['ky']
    else:
        energy_ax = data.zscale
        scanned_ax = data.xscale
        slit_ax = data.yscale

    def get_t_params(model):
        if 'li2018' in model:
            if model[-1] == 'd':
                t = np.array([501.3, -15.9, 557.1, -72.0, -13.9, 12.2])
            else:
                t = np.array([-45.1, 157.8, 203.2, 25.7, 0.02, 0.48])
        elif 'rossnagel2005' in model:
            if model[-1] == 'd':
                t = np.array([0.4108, -0.0726, 0.4534, -0.12]) * 1000
            else:
                t = np.array([-0.0373, 0.276, 0.2868, 0.007]) * 1000
        elif 'inosov2008' in model:
            if model[-1] == 'd':
                t = np.array([0.369, 0.074, 0.425, -0.049, 0.018]) * 1000
            else:
                t = np.array([-0.064, 0.167, 0.211, 0.005, 0.003]) * 1000
        return t

    def set_t_params(val):
        t = get_t_params(val['new'])
        t1.value = t[0]
        t2.value = t[1]
        t3.value = t[2]
        t4.value = t[3]
        t5.value = t[4]
        t6.value = t[5]

    # -------- Colorscale panel
    def get_const_e_map(data, energy, integrate_e, t1, t2, t3, t4, t5, t6, cut_orient, cut_idx, title, cmap, fontsize):
        fig, ax = plt.subplots(figsize=(10, 10))
        mpl.rcParams['font.size'] = fontsize
        mpl.rcParams['xtick.labelsize'] = fontsize
        mpl.rcParams['ytick.labelsize'] = fontsize

        tt = [t1, t2, t3, t4, t5, t6]
        tb_model = tss.TB_Ek(scanned_ax, slit_ax, *tt, x=2, flip=False) / 1000

        e_idx = wp.indexof(energy, energy_ax)

        if integrate_e == 0:
            cut = data.data[:, :, e_idx]
        else:
            e_i, e_f = e_idx - integrate_e, e_idx + integrate_e
            if e_i < 0:
                e_i = 0
            if e_f > energy_ax.size:
                e_f = energy_ax.size
            cut = np.sum(data.data[:, :, e_i:e_f], axis=2)

        if cut_orient[0] == 'h':
            line_x = np.ones_like(scanned_ax) * slit_ax[cut_idx]
            line_y = scanned_ax
        else:
            line_x = np.ones_like(slit_ax) * scanned_ax[cut_idx]
            line_y = slit_ax

        kxx, kyy = wp.a2k(data.xscale, data.yscale, 130, d_scan_ax=0.756, d_anal_ax=-0.0924, orientation='v', a=3.314,
                          work_func=4.38)
        ax.pcolormesh(kxx, kyy, cut, cmap=cmap)
        ax.contour(kyy, kxx, tb_model, [energy], colors='cyan')
        ax.plot(line_x, line_y, 'g--')

        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel('Scanned axis')
        ax.set_ylabel('Analyzer axis')
        ax.set_xlim(kxx.min(), kxx.max())
        ax.set_ylim(kyy.min(), kyy.max())
        plt.show()

    def set_const_e_cell(data, fname):

        def update_e_label(val):
            new_lbl = wp.indexof(val['new'], energy_ax)
            cell['energy_lbl'].value = str(new_lbl)

        cell = panel[0]

        cell['energy'] = ipw.FloatSlider(
            value=energy_ax[wp.indexof(0, energy_ax)],
            min=np.min(energy_ax),
            max=np.max(energy_ax),
            step=np.abs(energy_ax[0] - energy_ax[1]),
            description='Energy:',
            continuous_update=False,
            layout=ip.Layout(width='95%'),
            readout_format='.5f',
            style=style)

        cell['energy_lbl'] = ipw.Label(
            value=str(wp.indexof(0, energy_ax)),
            continuous_update=False,
            style=style)
        cell['energy'].observe(update_e_label, 'value')

        cell['integrate_e'] = ipw.IntSlider(
            value=0,
            min=0,
            max=10,
            step=1,
            description='Integration (Eb)',
            layout=ip.Layout(width='85%'),
            continuous_update=False,
            style=style)

        cell['output'] = ipw.interactive_output(get_const_e_map, {
            'data': ip.fixed(data),
            'energy': cell['energy'],
            'integrate_e': cell['integrate_e'],
            't1': t1,
            't2': t2,
            't3': t3,
            't4': t4,
            't5': t5,
            't6': t6,
            'cut_orient': cut_selector,
            'cut_idx': cut_index,
            'title': ip.fixed(fname),
            'cmap': cmap_selector,
            'fontsize': font_size_selector
        })
        cell['widget'] = ipw.VBox([cell['output'],
                                   ipw.HBox([cell['energy'], cell['energy_lbl']], layout=ip.Layout(height='75px')),
                                   cell['integrate_e']], layout=cell_layout)

    def get_cut(data, t1, t2, t3, t4, t5, t6, cut_orient, cut_idx, cmap, fontsize):

        fig, ax = plt.subplots(figsize=(10, 10))
        mpl.rcParams['font.size'] = fontsize
        mpl.rcParams['xtick.labelsize'] = fontsize
        mpl.rcParams['ytick.labelsize'] = fontsize

        if cut_orient[0] == 'h':
            ikx = 3
            cut = np.sum(data.data[cut_idx - ikx: cut_idx + ikx, :, :], axis=0).T
            k, e = np.meshgrid(slit_ax, energy_ax)
            title = 'Analyzer axis'
        else:
            iky = 3
            cut = np.sum(data.data[:, cut_idx - iky: cut_idx + iky, :], axis=1).T
            k, e = np.meshgrid(scanned_ax, energy_ax)
            title = 'Scanned axis'

        ax.pcolormesh(k, e, cut, cmap=cmap)
        ax.set_title(title)
        plt.show()

    def set_cut_cell(data, cell):

        cell['output'] = ipw.interactive_output(get_cut, {
            'data': ip.fixed(data),
            't1': t1,
            't2': t2,
            't3': t3,
            't4': t4,
            't5': t5,
            't6': t6,
            'cut_orient': cut_selector,
            'cut_idx': cut_index,
            'cmap': cmap_selector,
            'fontsize': font_size_selector
            # 'figsize': fig_size_selector
        })
        cell['widget'] = ipw.VBox([cell['output']], layout=cell_layout)

    cell_width = '98%'  # Windows

    util_cell_layout = ipw.Layout()

    cmap_selector = ipw.Dropdown(options=my_cmaps, value='viridis', description='colormap:', layout=util_cell_layout)
    cut_selector = ipw.Dropdown(options=cuts, value='horizontal', description='cut:', layout=util_cell_layout)
    cut_index = ipw.IntText(value=20, min=0, max=48, step=1, description='cut idx:', layout=ipw.Layout(width='80%'))
    font_size_selector = ipw.IntText(value=12, min=3, max=48, step=1, description='fontsize:', layout=util_cell_layout)
    # fig_size_selector = ipw.IntText(value=10, min=3, max=20, step=1, description='figsize:', layout=util_cell_layout)
    model_selector = ipw.Dropdown(options=models, value='li2018_d', description='model:', layout=util_cell_layout)
    transponse_model = ipw.Checkbox(value=False, description='transponse:', layout=ipw.Layout(width='80%'))

    t1 = ipw.FloatText(value=0, min=-1000, max=1000, step=0.1, description='t1', layout=ipw.Layout(width='80%'))
    t2 = ipw.FloatText(value=0, min=-1000, max=1000, step=0.1, description='t2', layout=ipw.Layout(width='80%'))
    t3 = ipw.FloatText(value=0, min=-1000, max=1000, step=0.1, description='t3', layout=ipw.Layout(width='80%'))
    t4 = ipw.FloatText(value=0, min=-1000, max=1000, step=0.1, description='t4', layout=ipw.Layout(width='80%'))
    t5 = ipw.FloatText(value=0, min=-1000, max=1000, step=0.1, description='t5', layout=ipw.Layout(width='80%'))
    t6 = ipw.FloatText(value=0, min=-1000, max=1000, step=0.1, description='t6', layout=ipw.Layout(width='80%'))

    util_panel_layout = ipw.Layout(border='dashed 1px gray', margin='0% 0% 1% 0%', width='100%', style=style,
                                   flex_flow='row')
    cell_layout = ipw.Layout(border='dashed 1px gray', margin='0% 0.5% 0.5% 0%', width=cell_width, height=cell_width)

    utilities_panel = ipw.HBox([ipw.VBox([cmap_selector, cut_selector, cut_index]),
                                ipw.VBox([font_size_selector, model_selector, transponse_model]),
                                ipw.VBox([t1, t2, t3]),
                                ipw.VBox([t4, t5, t6])],
                               layout=util_panel_layout)

    set_const_e_cell(data, fname)
    set_cut_cell(data, panel[1])
    model_selector.observe(set_t_params, 'value')
    # t1.observe(set_tb_model, 'value')
    # t2.observe(set_tb_model, 'value')
    # t3.observe(set_tb_model, 'value')
    # t4.observe(set_tb_model, 'value')
    # t5.observe(set_tb_model, 'value')
    # t6.observe(set_tb_model, 'value')
    # widget.observe(fun, 'value')

    main_panel = ipw.HBox([panel[0]['widget'], panel[1]['widget']], layout=ip.Layout(width='100%'))

    whole_panel = ipw.VBox([utilities_panel, main_panel], layout=ip.Layout(width='100%'))

    return whole_panel
