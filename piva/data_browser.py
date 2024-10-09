import os
import time
import importlib.util
import inspect

import numpy as np
from PyQt5.QtWidgets import QAction, QHBoxLayout, QLabel, QVBoxLayout, \
    QLineEdit, QMessageBox, QMainWindow
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

import piva.data_loaders as dl
import piva.working_procedures as wp
import piva.data_viewer_2d as p2d
import piva.data_viewer_3d as p3d
import piva.data_viewer_4d as p4d
import piva.plot_tool as pt


START_TIME = time.time()
testing = True
all_dls = {
    'Pickle': dl.DataloaderPickle,
    'SLS - SIS': dl.DataloaderSIS,
    'SLS - ADRESS': dl.DataloaderADRESS,
    'MAX IV - Bloch': dl.DataloaderBloch,
    'ALS - MERLIN (4.0.3)': dl.DataloaderMERLIN,
    'ALS - HERS (10.0.1)': dl.DataloaderHERS,
    'Diamond - i05': dl.DataloaderI05,
    'SOLARIS - URANOS': dl.DataloaderURANOS,
    'SOLEIL - CASSIOPEE': dl.DataloaderCASSIOPEE}


class DataBrowser(QMainWindow):
    """
    Initializes and displays the **DataBrowser** window.
    """

    def __init__(self) -> None:

        super(DataBrowser, self).__init__()
        # get time required for opening a window
        time_packages = time.time()
        print("loading packages: {:.3f} s".format(time_packages - START_TIME))

        self.thread_count = 0
        # self.thread = {}
        self.data_viewers = {}
        self.plotting_tools = {}
        self.file_explorer = None
        self.model = None
        self.jl_session_running = False
        self.sb_timeout = 2500
        self.working_dir = self.add_slash(os.getcwd())
        self.set_file_explorer()
        self.set_menu_bar()
        self.set_status_bar()
        self.set_details_panel()
        self.align()

        # loading custom plugins
        self.load_custom_data_loaders()
        self.load_custom_widgets()

        # set window
        self.setWindowTitle('piva data browser - ' + self.working_dir)
        self.show()
        time_init = time.time()
        print("initializing piva browser: {:.3f} s".format(time_init -
                                                           time_packages))

    @staticmethod
    def add_slash(path: str) -> str:
        """
        Add ``/`` at the end of the path string (if it's not there).

        :param path: initial path string
        :return: path with appended slash
        """

        if path[-1] != '/':
            path += '/'
        return path

    def align(self) -> None:
        """
        Align panels of the :class:`DataBrowser` GUI.
        """

        self.resize(700, 600)

        self.central_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QGridLayout()
        self.central_widget.setLayout(self.main_layout)

        self.main_layout.setMenuBar(self.menu_bar)
        self.main_layout.addWidget(self.file_explorer, 0, 0)
        self.main_layout.addWidget(self.details_panel, 0, 1)

        self.setCentralWidget(self.central_widget)

    def launch_piva(self) -> None:
        """
        Get selected file and check if it's not opened already.
        """

        fname = self.get_selected_path()

        if fname in self.data_viewers:
            already_opened_box = QMessageBox()
            already_opened_box.setIcon(QMessageBox.Information)
            already_opened_box.setText('File already opened.')
            already_opened_box.setStandardButtons(QMessageBox.Ok)
            if already_opened_box.exec() == QMessageBox.Ok:
                return

        self.open_dv(fname)

    def launch_example(self) -> None:
        """
        Open **DataViewer** with axample data.
        """

        fname = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))) + \
                       '/piva/tests/data/test_map.p'
        self.open_dv(fname)

    def mb_open_dir(self) -> None:
        """
        Open file dialog to select different working directory and change
        the highest directory of the :class:`DataBrowser` file explorer to it.
        """

        chosen_dir = str(QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select Directory', self.working_dir))
        try:
            self.working_dir = self.add_slash(chosen_dir)
            self.change_working_dir(self.working_dir)
            self.setWindowTitle('piva data browser - ' + self.working_dir)
        except IndexError:
            pass

    def open_dv(self, fname: str, example: bool = False) -> None:
        """
        Open selected file in a **DataViewer**.

        :type example: if :py:obj:`True` open example dataset
        :param fname: absolute path to selected file
        """

        if example:
            loader = all_dls['Pickle']()
            data_set = loader.load_data(fname)
        else:
            selected_loader = self.dp_dl_picker.currentText()
            if selected_loader == 'All':
                data_set = dl.load_data(fname)
            else:
                loader = all_dls[selected_loader]()
                data_set = loader.load_data(fname)

        if isinstance(data_set, np.ndarray):
            try:
                # self.add_viewer_to_linked_list(fname, 3)
                self.data_viewers[fname] = \
                    p4d.DataViewer4D(self, data_set=data_set, index=fname)
            except (Exception, AttributeError) as e:
                self.sb.showMessage(
                    'Couldn\'t load data,  format not supported.',
                    self.sb_timeout)
                if testing:
                    raise e
            return

        try:
            # choose correct viewer
            if data_set.xscale.size == 1:
                self.add_viewer_to_linked_list(fname, 2)
                self.data_viewers[fname] = \
                    p2d.DataViewer2D(self, data_set=data_set, index=fname)
            else:
                self.add_viewer_to_linked_list(fname, 3)
                self.data_viewers[fname] = \
                    p3d.DataViewer3D(self, data_set=data_set, index=fname)
        except (Exception, AttributeError) as e:
            self.sb.showMessage('Couldn\'t load data,  format not supported.',
                                self.sb_timeout)
            if testing:
                raise e

    def open_single_plotting_tool(self) -> None:
        """
        Open :class:`PlotTool` window.
        """

        thread_lbl = f'Plotting tool - {self.thread_count + 1}'
        try:
            self.plotting_tools[thread_lbl] = pt.PlotTool(self,
                                                            title=thread_lbl)
        except Exception:
            self.sb.showMessage('Couldn\'t open plotting tool',
                                self.sb_timeout)
        finally:
            self.thread_count += 1

    def reset_detail_panel(self) -> None:
        """
        Set all metadata fields to ``-``.
        """

        self.dp_scan_type.setText(self.dp_def_fill)
        self.dp_scan_start.setText(self.dp_def_fill)
        self.dp_scan_stop.setText(self.dp_def_fill)
        self.dp_scan_step.setText(self.dp_def_fill)

        self.dp_manip_x.setText(self.dp_def_fill)
        self.dp_manip_y.setText(self.dp_def_fill)
        self.dp_manip_z.setText(self.dp_def_fill)
        self.dp_manip_theta.setText(self.dp_def_fill)
        self.dp_manip_phi.setText(self.dp_def_fill)
        self.dp_manip_tilt.setText(self.dp_def_fill)
        self.dp_manip_temp.setText(self.dp_def_fill)
        self.dp_manip_press.setText(self.dp_def_fill)

        self.dp_ana_e0.setText(self.dp_def_fill)
        self.dp_ana_e1.setText(self.dp_def_fill)
        self.dp_ana_de.setText(self.dp_def_fill)
        self.dp_ana_pe.setText(self.dp_def_fill)
        self.dp_ana_dt.setText(self.dp_def_fill)
        self.dp_ana_lm.setText(self.dp_def_fill)
        self.dp_ana_am.setText(self.dp_def_fill)
        self.dp_ana_ana_slit.setText(self.dp_def_fill)
        self.dp_ana_defl_ang.setText(self.dp_def_fill)
        self.dp_ana_n_sweeps.setText(self.dp_def_fill)

        self.dp_bl_hv.setText(self.dp_def_fill)
        self.dp_bl_polar.setText(self.dp_def_fill)
        self.dp_bl_exit.setText(self.dp_def_fill)
        self.dp_bl_fe.setText(self.dp_def_fill)

    def set_details_panel(self) -> None:
        """
        Create and align all metadata fields and add them to
        :class:`DataBrowser` window.
        """

        dp = QtWidgets.QWidget()
        dp_layout = QVBoxLayout()

        bold_font = QtGui.QFont()
        bold_font.setBold(True)
        dp_def_fill = '-'
        max_len_1 = 60
        max_len_2 = int(1.5 * 50)

        # pick dataloader
        if 1:
            dp_dl_picker_layout = QHBoxLayout()
            dp_dl_picker_label = QLabel('Data loader:')
            dp_dl_picker_label.setFont(bold_font)
            self.dp_dl_picker = QtWidgets.QComboBox()
            self.dp_dl_picker.addItem('All')
            for key in all_dls.keys():
                self.dp_dl_picker.addItem(key)
            dp_dl_picker_layout.addWidget(dp_dl_picker_label)
            dp_dl_picker_layout.addWidget(self.dp_dl_picker)
            dp_dl_picker_layout.addStretch()

        dp_lbl = QLabel('Metadata')
        dp_lbl.setFont(bold_font)
        dp_layout.addWidget(dp_lbl)

        # scan details
        if 1:
            # create main label and layouts
            dp_scan_layout = QVBoxLayout()
            dp_scan_row1_layout = QHBoxLayout()
            dp_scan_row2_layout = QHBoxLayout()
            dp_scan_layout.addLayout(dp_scan_row1_layout)
            dp_scan_layout.addLayout(dp_scan_row2_layout)
            # create labels
            dp_scan_type_lbl = QLabel('Scan type: ')
            dp_scan_type_lbl.setFont(bold_font)
            dp_scan_start_lbl = QLabel('[0]:')
            dp_scan_stop_lbl = QLabel('[-1]:')
            dp_scan_step_lbl = QLabel('step:')
            # create textboxes
            self.dp_scan_type = QLineEdit(dp_def_fill)
            self.dp_scan_start = QLineEdit(dp_def_fill)
            self.dp_scan_stop = QLineEdit(dp_def_fill)
            self.dp_scan_step = QLineEdit(dp_def_fill)
            # set max width
            self.dp_scan_type.setMaximumWidth(max_len_2*2)
            self.dp_scan_start.setMaximumWidth(max_len_1)
            self.dp_scan_stop.setMaximumWidth(max_len_1)
            self.dp_scan_step.setMaximumWidth(max_len_1)
            # set read only
            self.dp_scan_type.setReadOnly(True)
            self.dp_scan_start.setReadOnly(True)
            self.dp_scan_stop.setReadOnly(True)
            self.dp_scan_step.setReadOnly(True)
            # add to layout
            dp_scan_row1_layout.addWidget(dp_scan_type_lbl)
            dp_scan_row1_layout.addWidget(self.dp_scan_type)
            dp_scan_row2_layout.addWidget(dp_scan_start_lbl)
            dp_scan_row2_layout.addWidget(self.dp_scan_start)
            dp_scan_row2_layout.addWidget(dp_scan_stop_lbl)
            dp_scan_row2_layout.addWidget(self.dp_scan_stop)
            dp_scan_row2_layout.addWidget(dp_scan_step_lbl)
            dp_scan_row2_layout.addWidget(self.dp_scan_step)
            dp_scan_row1_layout.addStretch()
            dp_scan_row2_layout.addStretch()

        # manipulator details
        if 1:
            # create main label and layouts
            dp_manip_lbl = QLabel('Manipulator')
            dp_manip_lbl.setFont(bold_font)
            dp_manip_layout = QVBoxLayout()
            dp_manip_row1_layout = QHBoxLayout()
            dp_manip_row2_layout = QHBoxLayout()
            dp_manip_row3_layout = QHBoxLayout()
            dp_manip_row4_layout = QHBoxLayout()
            dp_manip_layout.addLayout(dp_manip_row1_layout)
            dp_manip_layout.addLayout(dp_manip_row2_layout)
            dp_manip_layout.addLayout(dp_manip_row3_layout)
            dp_manip_layout.addLayout(dp_manip_row4_layout)
            # create labels
            dp_manip_col1_width = 35
            dp_manip_col2_width = 60
            dp_manip_x_lbl = QLabel('x:')
            dp_manip_x_lbl.setFixedWidth(dp_manip_col1_width)
            dp_manip_y_lbl = QLabel('y:')
            dp_manip_y_lbl.setFixedWidth(dp_manip_col1_width)
            dp_manip_z_lbl = QLabel('z:')
            dp_manip_z_lbl.setFixedWidth(dp_manip_col1_width)
            dp_manip_theta_lbl = QLabel('theta:')
            dp_manip_theta_lbl.setFixedWidth(dp_manip_col2_width)
            dp_manip_phi_lbl = QLabel('phi:')
            dp_manip_phi_lbl.setFixedWidth(dp_manip_col2_width)
            dp_manip_tilt_lbl = QLabel('tilt:')
            dp_manip_tilt_lbl.setFixedWidth(dp_manip_col2_width)
            dp_manip_temp_lbl = QLabel('T [K]:')
            dp_manip_temp_lbl.setFixedWidth(dp_manip_col1_width)
            dp_manip_press_lbl = QLabel('p [mbar]:')
            dp_manip_press_lbl.setFixedWidth(dp_manip_col2_width)
            # create textboxes
            self.dp_manip_x = QLineEdit(dp_def_fill)
            self.dp_manip_y = QLineEdit(dp_def_fill)
            self.dp_manip_z = QLineEdit(dp_def_fill)
            self.dp_manip_theta = QLineEdit(dp_def_fill)
            self.dp_manip_phi = QLineEdit(dp_def_fill)
            self.dp_manip_tilt = QLineEdit(dp_def_fill)
            self.dp_manip_temp = QLineEdit(dp_def_fill)
            self.dp_manip_press = QLineEdit(dp_def_fill)
            # set max width
            self.dp_manip_x.setMaximumWidth(max_len_1)
            self.dp_manip_y.setMaximumWidth(max_len_1)
            self.dp_manip_z.setMaximumWidth(max_len_1)
            self.dp_manip_theta.setMaximumWidth(max_len_1)
            self.dp_manip_phi.setMaximumWidth(max_len_1)
            self.dp_manip_tilt.setMaximumWidth(max_len_1)
            self.dp_manip_temp.setMaximumWidth(max_len_1)
            self.dp_manip_press.setMaximumWidth(max_len_2)
            # set read only
            self.dp_manip_x.setReadOnly(True)
            self.dp_manip_y.setReadOnly(True)
            self.dp_manip_z.setReadOnly(True)
            self.dp_manip_theta.setReadOnly(True)
            self.dp_manip_phi.setReadOnly(True)
            self.dp_manip_tilt.setReadOnly(True)
            self.dp_manip_temp.setReadOnly(True)
            self.dp_manip_press.setReadOnly(True)
            # add to layout
            dp_manip_row1_layout.addWidget(dp_manip_x_lbl)
            dp_manip_row1_layout.addWidget(self.dp_manip_x)
            dp_manip_row2_layout.addWidget(dp_manip_y_lbl)
            dp_manip_row2_layout.addWidget(self.dp_manip_y)
            dp_manip_row3_layout.addWidget(dp_manip_z_lbl)
            dp_manip_row3_layout.addWidget(self.dp_manip_z)
            dp_manip_row1_layout.addWidget(dp_manip_theta_lbl)
            dp_manip_row1_layout.addWidget(self.dp_manip_theta)
            dp_manip_row2_layout.addWidget(dp_manip_phi_lbl)
            dp_manip_row2_layout.addWidget(self.dp_manip_phi)
            dp_manip_row3_layout.addWidget(dp_manip_tilt_lbl)
            dp_manip_row3_layout.addWidget(self.dp_manip_tilt)
            dp_manip_row4_layout.addWidget(dp_manip_temp_lbl)
            dp_manip_row4_layout.addWidget(self.dp_manip_temp)
            dp_manip_row4_layout.addWidget(dp_manip_press_lbl)
            dp_manip_row4_layout.addWidget(self.dp_manip_press)
            dp_manip_row1_layout.addStretch()
            dp_manip_row2_layout.addStretch()
            dp_manip_row3_layout.addStretch()
            dp_manip_row4_layout.addStretch()

        # analyzer details
        if 1:
            # create main label and layouts
            dp_ana_lbl = QLabel('Analyzer')
            dp_ana_lbl.setFont(bold_font)
            dp_ana_layout = QVBoxLayout()
            dp_ana_row1_layout = QHBoxLayout()
            dp_ana_row2_layout = QHBoxLayout()
            dp_ana_row3_layout = QHBoxLayout()
            dp_ana_row4_layout = QHBoxLayout()
            dp_ana_row5_layout = QHBoxLayout()
            dp_ana_layout.addLayout(dp_ana_row1_layout)
            dp_ana_layout.addLayout(dp_ana_row2_layout)
            dp_ana_layout.addLayout(dp_ana_row3_layout)
            dp_ana_layout.addLayout(dp_ana_row4_layout)
            dp_ana_layout.addLayout(dp_ana_row5_layout)
            # create labels
            dp_ana_col1_width = 35
            dp_ana_col2_width = 65
            dp_ana_e0_lbl = QLabel('E[0]:')
            dp_ana_e0_lbl.setFixedWidth(dp_ana_col1_width)
            dp_ana_e1_lbl = QLabel('E[-1]:')
            dp_ana_e1_lbl.setFixedWidth(dp_ana_col1_width)
            dp_ana_de_lbl = QLabel('step:')
            dp_ana_de_lbl.setFixedWidth(dp_ana_col1_width)
            dp_ana_pe_lbl = QLabel('PE:')
            dp_ana_pe_lbl.setFixedWidth(dp_ana_col1_width)
            dp_ana_dt_lbl = QLabel('DT:')
            dp_ana_dt_lbl.setFixedWidth(dp_ana_col1_width)
            dp_ana_lm_lbl = QLabel('lens mode:')
            dp_ana_lm_lbl.setFixedWidth(dp_ana_col2_width)
            dp_ana_am_lbl = QLabel('acq mode:')
            dp_ana_am_lbl.setFixedWidth(dp_ana_col2_width)
            dp_ana_ana_slit_lbl = QLabel('ana slit:')
            dp_ana_ana_slit_lbl.setFixedWidth(dp_ana_col2_width)
            dp_ana_defl_ang_lbl = QLabel('defl [deg]:')
            dp_ana_defl_ang_lbl.setFixedWidth(dp_ana_col2_width)
            dp_ana_n_sweeps_lbl = QLabel('sweeps:')
            dp_ana_n_sweeps_lbl.setFixedWidth(dp_ana_col2_width)
            # create textboxes
            self.dp_ana_e = QLineEdit(dp_def_fill)
            self.dp_ana_e0 = QLineEdit(dp_def_fill)
            self.dp_ana_e1 = QLineEdit(dp_def_fill)
            self.dp_ana_de = QLineEdit(dp_def_fill)
            self.dp_ana_pe = QLineEdit(dp_def_fill)
            self.dp_ana_dt = QLineEdit(dp_def_fill)
            self.dp_ana_lm = QLineEdit(dp_def_fill)
            self.dp_ana_am = QLineEdit(dp_def_fill)
            self.dp_ana_ana_slit = QLineEdit(dp_def_fill)
            self.dp_ana_defl_ang = QLineEdit(dp_def_fill)
            self.dp_ana_n_sweeps = QLineEdit(dp_def_fill)
            # set max width
            self.dp_ana_e0.setMaximumWidth(max_len_2)
            self.dp_ana_e1.setMaximumWidth(max_len_2)
            self.dp_ana_de.setMaximumWidth(max_len_2)
            self.dp_ana_pe.setMaximumWidth(max_len_2)
            self.dp_ana_dt.setMaximumWidth(max_len_2)
            self.dp_ana_lm.setMaximumWidth(max_len_2)
            self.dp_ana_am.setMaximumWidth(max_len_2)
            self.dp_ana_ana_slit.setMaximumWidth(max_len_2)
            self.dp_ana_defl_ang.setMaximumWidth(max_len_2)
            self.dp_ana_n_sweeps.setMaximumWidth(max_len_2)
            # set read only
            self.dp_ana_e0.setReadOnly(True)
            self.dp_ana_e1.setReadOnly(True)
            self.dp_ana_de.setReadOnly(True)
            self.dp_ana_pe.setReadOnly(True)
            self.dp_ana_dt.setReadOnly(True)
            self.dp_ana_lm.setReadOnly(True)
            self.dp_ana_am.setReadOnly(True)
            self.dp_ana_n_sweeps.setReadOnly(True)
            self.dp_ana_ana_slit.setReadOnly(True)
            self.dp_ana_defl_ang.setReadOnly(True)
            # add labels to layouts
            dp_ana_row1_layout.addWidget(dp_ana_e0_lbl)
            dp_ana_row1_layout.addWidget(self.dp_ana_e0)
            dp_ana_row2_layout.addWidget(dp_ana_e1_lbl)
            dp_ana_row2_layout.addWidget(self.dp_ana_e1)
            dp_ana_row3_layout.addWidget(dp_ana_de_lbl)
            dp_ana_row3_layout.addWidget(self.dp_ana_de)
            dp_ana_row4_layout.addWidget(dp_ana_pe_lbl)
            dp_ana_row4_layout.addWidget(self.dp_ana_pe)
            dp_ana_row5_layout.addWidget(dp_ana_dt_lbl)
            dp_ana_row5_layout.addWidget(self.dp_ana_dt)
            dp_ana_row1_layout.addWidget(dp_ana_lm_lbl)
            dp_ana_row1_layout.addWidget(self.dp_ana_lm)
            dp_ana_row2_layout.addWidget(dp_ana_am_lbl)
            dp_ana_row2_layout.addWidget(self.dp_ana_am)
            dp_ana_row3_layout.addWidget(dp_ana_ana_slit_lbl)
            dp_ana_row3_layout.addWidget(self.dp_ana_ana_slit)
            dp_ana_row4_layout.addWidget(dp_ana_defl_ang_lbl)
            dp_ana_row4_layout.addWidget(self.dp_ana_defl_ang)
            dp_ana_row5_layout.addWidget(dp_ana_n_sweeps_lbl)
            dp_ana_row5_layout.addWidget(self.dp_ana_n_sweeps)
            dp_ana_row1_layout.addStretch()
            dp_ana_row2_layout.addStretch()
            dp_ana_row3_layout.addStretch()
            dp_ana_row4_layout.addStretch()
            dp_ana_row5_layout.addStretch()

        # beamline details
        if 1:
            # create main label and layouts
            dp_bl_lbl = QLabel('Beamline')
            dp_bl_lbl.setFont(bold_font)
            dp_bl_layout = QVBoxLayout()
            dp_bl_row1_layout = QHBoxLayout()
            dp_bl_row2_layout = QHBoxLayout()
            dp_bl_layout.addLayout(dp_bl_row1_layout)
            dp_bl_layout.addLayout(dp_bl_row2_layout)
            # create labels
            dp_bl_col1_width = 35
            dp_bl_col2_width = 75
            dp_bl_hv_lbl = QLabel('h\u03BD:')
            dp_bl_hv_lbl.setFixedWidth(dp_bl_col1_width)
            dp_bl_polar_lbl = QLabel('polarization:')
            dp_bl_polar_lbl.setFixedWidth(dp_bl_col2_width)
            dp_bl_exit_lbl = QLabel('exit:')
            dp_bl_exit_lbl.setFixedWidth(dp_bl_col1_width)
            dp_bl_fe_lbl = QLabel('front end:')
            dp_bl_fe_lbl.setFixedWidth(dp_bl_col2_width)
            # create textboxes
            self.dp_bl_hv = QLineEdit(dp_def_fill)
            self.dp_bl_polar = QLineEdit(dp_def_fill)
            self.dp_bl_exit = QLineEdit(dp_def_fill)
            self.dp_bl_fe = QLineEdit(dp_def_fill)
            # set max width
            self.dp_bl_hv.setMaximumWidth(max_len_1)
            self.dp_bl_polar.setMaximumWidth(max_len_1)
            self.dp_bl_exit.setMaximumWidth(max_len_1)
            self.dp_bl_fe.setMaximumWidth(max_len_1)
            # set read only
            self.dp_bl_hv.setReadOnly(True)
            self.dp_bl_polar.setReadOnly(True)
            self.dp_bl_exit.setReadOnly(True)
            self.dp_bl_fe.setReadOnly(True)
            # add labels to layouts
            dp_bl_row1_layout.addWidget(dp_bl_hv_lbl)
            dp_bl_row1_layout.addWidget(self.dp_bl_hv)
            dp_bl_row1_layout.addWidget(dp_bl_polar_lbl)
            dp_bl_row1_layout.addWidget(self.dp_bl_polar)
            dp_bl_row2_layout.addWidget(dp_bl_exit_lbl)
            dp_bl_row2_layout.addWidget(self.dp_bl_exit)
            dp_bl_row2_layout.addWidget(dp_bl_fe_lbl)
            dp_bl_row2_layout.addWidget(self.dp_bl_fe)
            dp_bl_row1_layout.addStretch()
            dp_bl_row2_layout.addStretch()

        # add all to detail_panel layout
        if 1:
            # dl picker
            dp_layout.addLayout(dp_dl_picker_layout)
            # scan
            dp_layout.addLayout(dp_scan_layout)
            # manipulator
            dp_layout.addWidget(dp_manip_lbl)
            dp_layout.addLayout(dp_manip_layout)
            # analyzer
            dp_layout.addWidget(dp_ana_lbl)
            dp_layout.addLayout(dp_ana_layout)
            # beamline
            dp_layout.addWidget(dp_bl_lbl)
            dp_layout.addLayout(dp_bl_layout)

        # add layout to main window
        dp_layout.addStretch()
        dp.setLayout(dp_layout)
        self.dp_def_fill = dp_def_fill
        self.details_panel_layout = dp_layout
        self.details_panel = dp

    def set_menu_bar(self) -> None:
        """
        Create menu bar and connect all methods.
        """

        menu_bar = QtWidgets.QMenuBar()
        file_menu = menu_bar.addMenu('&File')
        open_dir = QAction('Open directory', self)
        open_dir.setShortcut('Ctrl+O')
        open_dir.setStatusTip('Open directory')
        open_dir.triggered.connect(self.mb_open_dir)
        file_menu.addAction(open_dir)

        open_file = QAction('Launch piva', self)
        open_file.setShortcut('Ctrl+L')
        open_file.setStatusTip('Launch piva')
        open_file.triggered.connect(self.launch_piva)
        file_menu.addAction(open_file)

        open_menu = menu_bar.addMenu('&Open')
        single_plot = QAction('Plotting Tool', self)
        single_plot.setShortcut('Ctrl+P')
        single_plot.setStatusTip('Plotting Tool')
        single_plot.triggered.connect(self.open_single_plotting_tool)
        open_menu.addAction(single_plot)

        example_data = QAction('Example', self)
        example_data.setStatusTip('Example')
        # print(EXAMPLE_DATA, type(EXAMPLE_DATA))
        example_data.triggered.connect(self.launch_example)
        open_menu.addAction(example_data)

        file_menu.addSeparator()

        self.menu_bar = menu_bar

    def set_status_bar(self) -> None:
        """
        Create status bar on the botton of the **DataBrowser's** window.
        It will be displaying some information.
        """

        self.sb = QtWidgets.QStatusBar()
        self.setStatusBar(self.sb)

    def set_file_explorer(self) -> None:
        """
        Create the file explorer view using
        :class:`~pyqtgraph.Qt.QtWidgets.QTreeView`.
        """

        model = QtGui.QFileSystemModel()
        file_explorer = QtWidgets.QTreeView()
        file_explorer.setModel(model)

        # set root as a current directory
        cwd = QtCore.QDir.currentPath()
        model.setRootPath(cwd)
        file_explorer.setRootIndex(model.index(cwd))

        # Visual fiddling: the columns are 0: filename, 1: size, 2: date 
        # modified, 3: type
        self.shown_columns = [0, 1, 2, 3]
        self.hidden_columns = [1, 2, 3]
        # Remove unnecessary columns
        for i in self.hidden_columns:
            file_explorer.hideColumn(i)
            self.shown_columns.remove(i)

        # Connect signal for changing selection in the QTreeView
        file_explorer.selectionModel().selectionChanged.connect(
            self.on_selection_change)
        self.file_explorer = file_explorer
        self.model = model

    def change_working_dir(self, path: str) -> None:
        """
        Change working directory in a shell and update :class:`DataBrowser`
        tree view.
        """

        if QtCore.QDir.exists(QtCore.QDir(), path):
            os.chdir(path)
        else:
            return
        cwd = QtCore.QDir.currentPath()
        self.model.setRootPath(cwd)
        self.file_explorer.setRootIndex(self.model.index(cwd))

    def on_selection_change(self) -> None:
        """
        Triggered when a new item in the :class:`~QTreeView` was clicked or
        selected through arrow keys. Ensure the new item is visible in the
        tree and try to update selected metadata panel.
        """

        self.file_explorer.resizeColumnToContents(0)
        self.update_details_panel()

    def get_selected_path(self) -> str:
        """
        Get the path of selected file in the :class:`QTreeView` as a string.
        """

        index = self.file_explorer.currentIndex()
        path = self.file_explorer.model().filePath(index)
        return path

    def update_details_panel(self) -> None:
        """
        Try reading the file currently selected by :class:`QTreeView`.
        In case of success, fill the metadata panel with metadata.
        """

        selected_loader = self.dp_dl_picker.currentText()
        fname = self.get_selected_path()

        try:
            if selected_loader == 'All':
                data = dl.load_data(fname, metadata=True)
            else:
                # Instantiante a loader of selected type
                loader = all_dls[selected_loader]()
                data = loader.load_data(fname, metadata=True)
        except FileNotFoundError:
            return
        except NotImplementedError:
            self.sb.showMessage('File extension not implemented '
                                'for a chosen data loader.', self.sb_timeout)
            return
        except Exception as e:
            self.sb.showMessage('Couldn\'t load data, '
                                'file format not supported.', self.sb_timeout)
            if testing:
                raise e
            return
        finally:
            # Clean up the details panel before we fill it with new info.
            self.reset_detail_panel()

        # Reaching this point means we have succeeded in loading *something*.
        if isinstance(data, np.ndarray):
            data = data[0, 0]

        try:
            # scan
            if not (data.scan_type is None):
                self.dp_scan_type.setText('{}'.format(data.scan_type))
            if not (data.scan_dim is None):
                sd = data.scan_dim
                if len(sd) == 0:
                    pass
                else:
                    if hasattr(data, 'scan_dim'):
                        self.dp_scan_start.setText('{:.2f}'.format(
                            float(data.scan_dim[0])))
                    if hasattr(data, 'scan_dim'):
                        self.dp_scan_stop.setText('{:.2f}'.format(
                            float(data.scan_dim[1])))
                    if hasattr(data, 'scan_dim'):
                        self.dp_scan_step.setText('{:.2f}'.format(
                            float(data.scan_dim[2])))
            # manipulator
            if not (data.x is None):
                self.dp_manip_x.setText('{:.2f}'.format(float(data.x)))
            if not (data.y is None):
                self.dp_manip_y.setText('{:.2f}'.format(float(data.y)))
            if not (data.z is None):
                self.dp_manip_z.setText('{:.2f}'.format(float(data.z)))
            if not (data.theta is None):
                self.dp_manip_theta.setText('{:.2f}'.format(float(data.theta)))
            if not (data.phi is None):
                self.dp_manip_phi.setText('{:.2f}'.format(float(data.phi)))
            if not (data.tilt is None):
                self.dp_manip_tilt.setText('{:.2f}'.format(float(data.tilt)))
            if not (data.temp is None):
                self.dp_manip_temp.setText('{:.1f}'.format(float(data.temp)))
            if not (data.pressure is None):
                self.dp_manip_press.setText('{:.2e}'.format(
                    float(data.pressure)))

            # analyzer
            if not (data.zscale is None):
                self.dp_ana_e0.setText('{:.4f}'.format(data.zscale[0]))
            if not (data.zscale is None):
                self.dp_ana_e1.setText('{:.4f}'.format(data.zscale[-1]))
            if not (data.zscale is None):
                self.dp_ana_de.setText('{:.2e}'.format(
                    wp.get_step(data.zscale)))
            if not (data.PE is None):
                self.dp_ana_pe.setText('{}'.format(int(data.PE)))
            if not (data.DT is None):
                self.dp_ana_dt.setText('{}'.format(int(data.DT)))
            if not (data.lens_mode is None):
                self.dp_ana_lm.setText('{}'.format(data.lens_mode))
            if not (data.acq_mode is None):
                self.dp_ana_am.setText('{}'.format(data.acq_mode))
            if not (data.ana_slit is None):
                self.dp_ana_ana_slit.setText('{}'.format(data.ana_slit))
            if not (data.defl_angle is None):
                self.dp_ana_defl_ang.setText('{}'.format(data.defl_angle))
            if not (data.n_sweeps is None):
                self.dp_ana_n_sweeps.setText('{}'.format(int(data.n_sweeps)))

            # beamline
            if not (data.hv is None):
                self.dp_bl_hv.setText('{:.1f}'.format(float(data.hv)))
            if not (data.polarization is None):
                self.dp_bl_polar.setText(data.polarization)
            if not (data.exit_slit is None):
                self.dp_bl_exit.setText('{}'.format(float(data.exit_slit)))
            if not (data.FE is None):
                self.dp_bl_fe.setText('{}'.format(float(data.FE)))
        except AttributeError:
            self.reset_detail_panel()

    def add_viewer_to_linked_list(self, fname: str, dim: int = 2) -> None:
        """
        Add a newly opened viewer to linking lists of other with matching
        dimensionality, enabling it for linking.
        :param fname: viewer's name/index
        :param dim: dataset's dimensionality
        """

        for dvi in self.data_viewers.keys():
            if dim == self.data_viewers[dvi].util_panel.dim:
                win_list = self.data_viewers[dvi].util_panel.link_windows_list
                win_list.addItem(fname.split('/')[-1])
                win_list.model().item(win_list.count() - 1).setCheckable(True)

    def delete_viewer_from_linked_lists(self, fname: str) -> None:
        """
        Delete current **DataViewer** from linking lists of others with
        matching dimensionality.

        :param fname: current **DataViewer's** name/index
        """

        for dvi in self.data_viewers.keys():
            list = self.data_viewers[dvi].util_panel.link_windows_list
            for idx in range(list.count()):
                if fname == list.itemText(idx):
                    list.removeItem(idx)
                    break

    def load_custom_data_loaders(self) -> None:
        """
        If configured, try to load user-implemented, custom
        :class:`~data_loaders.Dataloader`, allowing to read
        other formats of data.
        """

        try:
            import piva_dataloader_importer as dli
            print('Loading custom data loaders:')

            if 'DataloaderImporter' not in dir(dli):
                print('\t', 'Couldn\'t find class DataloaderImporter.')
                return

            self.dli = getattr(dli, 'DataloaderImporter')(self)

        except ModuleNotFoundError:
            pass

    @staticmethod
    def add_dataloader_to_record(name, loader):
        all_dls[name] = loader

    def load_custom_widgets(self) -> None:
        """
        If configured, try to load user-implemented, custom
        QtWidgets, allowing to expand PIVA's functionalities.
        """

        try:
            import piva_widget_importer as pwi
            print('Loading custom widgets:')

            if 'WidgetImporter' not in dir(pwi):
                print('\t', 'Couldn\'t find class WidgetImporter.')
                return

            self.custom_widgets = {}
            self.wi = getattr(pwi, 'WidgetImporter')(self)

        except ModuleNotFoundError:
            pass
