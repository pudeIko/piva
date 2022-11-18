"""
Provides several Dataloader objects which open different kinds of data files 
- typically acquired at different sources (i.e. beamlines at various 
synchrotrons) - and crunch them into the same shape and form.
The output form is an :class:`argparse.Namespace` object like this::

    Namespace(data,
              xscale,
              yscale,
              zscale,
              angles,
              theta,
              phi,
              E_b,
              hv)

Where the entries are as follows:

======  ========================================================================
data    np.array of shape (z,y,x); this has to be a 3D array even for 2D data 
        - z=1 in that case. x, y and z are the lengths of the x-, y- and 
        z-scales, respectively. The convention (z,y,x) is used over (x,y,z) 
        as a consequence of matplotlib.pcolormesh transposing the data when 
        plotting.
xscale  np.array of shape(x); the x axis corresponding to the data.
yscale  np.array of shape(y); the y axis corresponding to the data.
zscale  np.array of shape(z); the z axis corresponding to the data.
angles  1D np.array; corresponding angles on the momentum axis of the 
        analyzer. Depending on the beamline (analyzer slit orientation) this 
        is expressed as theta or tilt. Usually coincides with either of x-, 
        y- and zscales.
theta   float or 1D np.array; the value of theta (or tilt in rotated analyzer 
        slit orientation). Is mostly used for angle-to-k conversion.
phi     float; value of the azimuthal angle phi. Mostly used for angle-to-k 
        conversion.
E_b     float; typical binding energy for the electrons represented in the 
        data. In principle there is not just a single binding energy but as 
        this is only used in angle-to-k conversion, where the typical 
        variations of the order <10eV doen't matter it suffices to give an 
        average or maximum value.
hv      float or 1D np.array; the used photon energy in the scan(s). In Case 
        of a hv-scan, this obviously coincides with one of the x-, y- or 
        z-scales.
======  ========================================================================

Note that any change in the output structure has consequences for all 
programs and routines that receive from a dataloader (which is pretty much 
everything in this module) and previously pickled files.
    Quick description on how to read stuff from .h5 files using h5py
    =================================================================================
    file = h5py.File(fname, 'r')   read from file 'fname', 'r' stands for 'read only'
    
    Now, opened file (and objects inside it) has 'keys' and 'attributes'. To see/refer 
    to them use either:
        file.keys()
        file.attrs
            
    Below few examples how to get some data/metadata from the file:
        Actual data from measurement:
            data = list(file['Electron Analyzer/Image Data'])
        Sample temperature:
            T = float(list(file['Other Instruments/Temperature B (Sample 1)'])[0])
        Comments:
            comments = str(file.attrs['Comments']).replace('\\n', '; ')
        Measurement conditions:
            list(file['Electron Analyzer/Image Data'].attrs)
    =================================================================================
"""
import ast
import os
import pickle
import re
import zipfile
from copy import deepcopy
from argparse import Namespace
from errno import ENOENT
from warnings import catch_warnings, simplefilter  # , warn

import h5py
import numpy as np
import astropy.io.fits as pyfits
from igor import binarywave, igorpy


# Fcn to build the x, y (, z) ranges (maybe outsource this fcn definition)
def start_step_n(start, step, n):
    """ 
    Return an array that starts at value `start` and goes `n` 
    steps of `step`. 
    """
    end = start + n * step
    return np.linspace(start, end, n)


class DataSet:

    def __init__(self):
        self.dataset = Namespace(
            data=None,
            xscale=None,
            yscale=None,
            zscale=None,
            ekin=None,
            kxscale=None,
            kyscale=None,
            x=None,
            y=None,
            z=None,
            theta=None,
            phi=None,
            tilt=None,
            temp=None,
            pressure=None,
            hv=None,
            wf=None,
            Ef=None,
            polarization=None,
            PE=None,
            exit_slit=None,
            FE=None,
            scan_type=None,
            scan_dim=None,
            acq_mode=None,
            lens_mode=None,
            anal_slit=None,
            n_sweeps=None,
            DT=None
        )


class Dataloader:
    """ 
    Base dataloader class (interface) from which others inherit some 
    methods (specifically the ``__repr__()`` function). 
    The `date` attribute should indicate the last date that this specific 
    dataloader worked properly for files of its type (as beamline filetypes 
    may vary with time).
    """
    name = 'Base'
    date = ''

    def __init__(self, *args, **kwargs):
        pass

    def __repr__(self, *args, **kwargs):
        return '<class Dataloader_{}>'.format(self.name)

    def print_m(self, *messages):
        """ Print message to console, adding the dataloader name. """
        s = '[Dataloader {}]'.format(self.name)
        print(s, *messages)


class DataloaderPickle(Dataloader):
    """
    Load data that has been saved using python's `pickle` module. ARPES
    pickle files are assumed to just contain the data namespace the way it
    would be returned by any Dataloader of this module.
    """
    name = 'Pickle'

    @staticmethod
    # kwarg "metadata" necessary to match arguments of all other data loaders
    def load_data(filename, metadata=False):
        # Open the file and get a handle for it
        ds = DataSet()
        with open(filename, 'rb') as f:
            filedata = pickle.load(f)

        dict_ds = vars(ds.dataset)
        dict_filedata = vars(filedata)

        for attr in dir(filedata):
            if not (attr[0] == '_'):
                dict_ds[attr] = dict_filedata[attr]
        return ds.dataset


class DataloaderSIS(Dataloader):
    """
    Object that allows loading and saving of ARPES data from the SIS
    beamline at PSI which is in hd5 format.
    """
    name = 'SIS'
    # Number of cuts that need to be present to assume the data as a map
    # instead of a series of cuts
    min_cuts_for_map = 1

    def __init__(self):

        self.datfile = None
        pass

    def load_data(self, filename, metadata=False):
        """
        Extract and return the actual 'data', i.e. the recorded map/cut.
        Also return labels which provide some indications what the data means.
        """
        # Note: x and y are a bit confusing here as the hd5 file has a
        # different notion of zero'th and first dimension as numpy and then
        # later pcolormesh introduces yet another layer of confusion. The way
        # it is written now, though hard to read, turns out to do the right
        # thing and leads to readable code from after this point.

        ds = DataSet()
        if filename.endswith('h5'):
            filedata = self.load_h5(filename, metadata=metadata)
        elif filename.endswith('zip'):
            filedata = self.load_zip(filename, metadata=metadata)
        elif filename.endswith('pxt'):
            filedata = self.load_pxt(filename, metadata=metadata)
        elif filename.endswith('ibw'):
            filedata = self.load_ibw(filename, metadata=metadata)
        else:
            raise NotImplementedError('File extension not supported.')

        dict_ds = vars(ds.dataset)
        dict_filedata = vars(filedata)

        for attr in dir(filedata):
            if not (attr[0] == '_'):
                dict_ds[attr] = dict_filedata[attr]
        return ds.dataset

    def load_h5(self, filename, metadata=False):
        """ Load and store the full h5 file and extract relevant information. """
        # Load the hdf5 file
        # Use 'rdcc_nbytes' flag for setting up the chunk cache (in bytes)
        self.datfile = h5py.File(filename, 'r')
        # Extract the actual dataset and some metadata
        h5_data = self.datfile['Electron Analyzer/Image Data']
        attributes = h5_data.attrs

        # Convert to array and make 3 dimensional if necessary
        shape = h5_data.shape
        if metadata:
            data = np.zeros(shape)
        else:
            if len(shape) == 3:
                data = np.zeros(shape)
                for i in range(shape[2]):
                    data[:, :, i] = h5_data[:, :, i]
            else:
                data = np.array(h5_data)

        data = data.T
        if len(shape) == 2:
            x = 1
            y = shape[1]
            N_E = shape[0]
            # Make data 3D
            data = data.reshape((1, y, N_E))
            # Extract the limits
            xlims = [1, 1]
            ylims = attributes['Axis1.Scale']
            elims = attributes['Axis0.Scale']
            xscale = start_step_n(*xlims, x)
            yscale = start_step_n(*ylims, y)
            energies = start_step_n(*elims, N_E)
        elif len(shape) == 3:
            x = shape[1]
            y = shape[2]
            N_E = shape[0]
            # Extract the limits
            xlims = attributes['Axis2.Scale']
            ylims = attributes['Axis1.Scale']
            elims = attributes['Axis0.Scale']
            xscale = start_step_n(*xlims, y)
            yscale = start_step_n(*ylims, x)
            energies = start_step_n(*elims, N_E)
        # Case sequence of cuts
        else:
            x = shape[0]
            y = shape[1]
            N_E = y
            data = np.rollaxis(data, 2, 0)
            # Extract the limits
            xlims = attributes['Axis1.Scale']
            ylims = attributes['Axis0.Scale']
            elims = ylims
            xscale = start_step_n(*xlims, y)
            yscale = start_step_n(*ylims, x)
            energies = start_step_n(*elims, N_E)

        # Extract some data for ang2k conversion
        metadata = self.datfile['Other Instruments']
        x_pos = metadata['X'][0]
        y_pos = metadata['Y'][0]
        z_pos = metadata['Z'][0]
        theta = metadata['Theta'][0]
        phi = metadata['Phi'][0]
        tilt = metadata['Tilt'][0]
        temp = metadata['Temperature B (Sample 1)'][0]
        pressure = metadata['Pressure AC (ACMI)'][0]
        hv = attributes['Excitation Energy (eV)']
        wf = attributes['Work Function (eV)']
        polarization = metadata['hv'].attrs['Mode'][10:]
        PE = attributes['Pass Energy (eV)']
        exit_slit = metadata['Exit Slit'][0]
        FE = metadata['FE Horiz. Width'][0]
        ekin = energies + hv - wf
        lens_mode = attributes['Lens Mode']
        acq_mode = attributes['Acquisition Mode']
        n_sweeps = attributes['Sweeps on Last Image']
        DT = attributes['Dwell Time (ms)']
        if 'Axis2.Scale' in attributes:
            scan_type = attributes['Axis2.Description'] + ' scan'
            start = attributes['Axis2.Scale'][0]
            step = attributes['Axis2.Scale'][1]
            stop = attributes['Axis2.Scale'][0] + attributes['Axis2.Scale'][1] * xscale.size
            scan_dim = [start, stop, step]
        else:
            scan_type = 'cut'
            scan_dim = []

        res = Namespace(
            data=data,
            xscale=xscale,
            yscale=yscale,
            zscale=energies,
            ekin=ekin,
            x=x_pos,
            y=y_pos,
            z=z_pos,
            theta=theta,
            phi=phi,
            tilt=tilt,
            temp=temp,
            pressure=pressure,
            hv=hv,
            wf=wf,
            polarization=polarization,
            PE=PE,
            exit_slit=exit_slit,
            FE=FE,
            scan_type=scan_type,
            scan_dim=scan_dim,
            lens_mode=lens_mode,
            acq_mode=acq_mode,
            n_sweeps=n_sweeps,
            DT=DT
        )
        h5py.File.close(self.datfile)

        return res

    def load_zip(self, filename, metadata=False):
        """ Load and store a deflector mode file from SIS-ULTRA. """
        # Prepare metadata key-value pairs for the different metadata files
        # and their expected types
        keys1 = [
                 ('width', 'n_energy', int),
                 ('height', 'n_x', int),
                 ('depth', 'n_y', int),
                 ('first_full', 'first_energy', int),
                 ('last_full', 'last_energy', int),
                 ('widthoffset', 'start_energy', float),
                 ('widthdelta', 'step_energy', float),
                 ('heightoffset', 'start_x', float),
                 ('heightdelta', 'step_x', float),
                 ('depthoffset', 'start_y', float),
                 ('depthdelta', 'step_y', float)
                ]
        keys2 = [('Excitation Energy', 'hv', float),
                 ('Acquisition Mode', 'acq_mode', str),
                 ('Pass Energy', 'PE', int),
                 ('Lens Mode', 'lens_mode', str),
                 ('Step Time', 'DT', int),
                 ('Number of Sweeps', 'n_sweeps', int),
                 ('X', 'x', float),
                 ('Y', 'y', float),
                 ('Z', 'z', float),
                 ('A', 'phi', float),
                 ('P', 'theta', float),
                 ('T', 'tilt', float),
                 ('Y', 'y', float)]

        # Load the zipfile
        with zipfile.ZipFile(filename, 'r') as z:
            # Get the created filename from the viewer
            with z.open('viewer.ini') as viewer:
                file_id = self.read_viewer(viewer)
            # Get most metadata from a metadata file
            with z.open('Spectrum_' + file_id + '.ini') as metadata_file:
                M = self.read_metadata(keys1, metadata_file)
            # Get additional metadata from a second metadata file...
            with z.open(file_id + '.ini') as metadata_file2:
                M2 = self.read_metadata(keys2, metadata_file2)
                if not hasattr(M2, 'scan_type'):
                    M2.__setattr__('scan_type', 'cut')
                if not hasattr(M2, 'scan_start'):
                    M2.__setattr__('scan_start', 0)
                if not hasattr(M2, 'scan_stop'):
                    M2.__setattr__('scan_stop', 0)
                if not hasattr(M2, 'scan_step'):
                    M2.__setattr__('scan_step', 0)
                if hasattr(M2, 'scan_step'):
                    n_dim_steps = np.abs(M2.scan_start - M2.scan_stop) / M2.scan_step
                    M2.n_sweeps /= n_dim_steps
                else:
                    pass
            # Extract the binary data from the zipfile
            if metadata:
                data_flat = np.zeros((int(M.n_y) * int(M.n_x) * int(M.n_energy)))
            else:
                with z.open('Spectrum_' + file_id + '.bin') as f:
                    data_flat = np.frombuffer(f.read(), dtype='float32')

        # Put the data back into its actual shape
        data = np.reshape(data_flat, (int(M.n_y), int(M.n_x), int(M.n_energy)))
        # Cut off unswept region
        data = data[:, :, M.first_energy:M.last_energy+1]
        # Put into shape (energy, other angle, angle along analyzer)
        data = np.moveaxis(data, 2, 0)
        # Create axes
        xscale = start_step_n(M.start_x, M.step_x, M.n_x)
        yscale = start_step_n(M.start_y, M.step_y, M.n_y)
        energies = start_step_n(M.start_energy, M.step_energy, M.n_energy)
        energies = energies[M.first_energy:M.last_energy+1]

        if yscale.size > 1:
            yscale = start_step_n(M.start_x, M.step_x, M.n_x)
            xscale = start_step_n(M.start_y, M.step_y, M.n_y)
            data = np.swapaxes(data, 1, 2)
        else:
            data = np.swapaxes(data, 0, 1)
            yscale = deepcopy(energies)

        res = Namespace(
            data=data.T,
            xscale=xscale,
            yscale=yscale,
            zscale=energies,
            ekin=energies,
            hv=M2.hv,
            PE=M2.PE,
            scan_type=M2.scan_type,
            scan_dim=[M2.scan_start, M2.scan_stop, M2.scan_step],
            lens_mode=M2.lens_mode,
            acq_mode=M2.acq_mode,
            DT=M2.DT,
            n_sweeps=int(M2.n_sweeps)
        )
        return res

    # kwarg "metadata" necessary to match arguments of all other data loaders
    def load_pxt(self, filename, metadata=False):
        """ Load and store the full h5 file and extract relevant information. """
        pxt = igorpy.load(filename)[0]
        data = pxt.data.T
        shape = data.shape
        meta = pxt.notes.decode('ASCII').split('\r')
        keys1 = [('Excitation Energy', 'hv', float),
                 ('Acquisition Mode', 'acq_mode', str),
                 ('Pass Energy', 'PE', int),
                 ('Lens Mode', 'lens_mode', str),
                 ('Step Time', 'DT', int),
                 ('Number of Sweeps', 'n_sweeps', int),
                 ('ThetaY', 'tilt', float)]
        meta_namespace = vars(self.read_pxt_ibw_metadata(keys1, meta))

        if len(shape) == 2:
            x = 1
            y = pxt.axis[1].size
            N_E = pxt.axis[0].size
            # Make data 3D
            data = data.reshape((1, y, N_E))
            # Extract the limits
            xlims = [1, 1]
            xscale = start_step_n(*xlims, x)
            yscale = start_step_n(pxt.axis[1][-1], pxt.axis[1][0], y)
            energies = start_step_n(pxt.axis[0][-1], pxt.axis[0][0], N_E)
        else:
            print('Only cuts of .pxt files are working.')
            return

        res = Namespace(
            data=data,
            xscale=xscale,
            yscale=yscale,
            zscale=energies
        )

        for key in meta_namespace.keys():
            if not (key[0] == '_'):
                res.__setattr__(key, meta_namespace[key])
        return res

    # kwarg "metadata" necessary to match arguments of all other data loaders
    def load_ibw(self, filename, metadata=False):
        """
        Load scan data from an IGOR binary wave file. Luckily someone has
        already written an interface for this (the python `igor` package).
        """
        wave = binarywave.load(filename)['wave']
        # The `header` contains some metadata
        header = wave['wave_header']
        nDim = header['nDim']
        steps = header['sfA']
        starts = header['sfB']

        # Construct the x and y scales from start, stop and n
        x = 1
        xlims = [1, 1]
        xscale = start_step_n(*xlims, x)
        yscale = start_step_n(starts[1], steps[1], nDim[1])
        zscale = start_step_n(starts[0], steps[0], nDim[0])

        # data = np.zeros((xscale.size, yscale.size, zscale.size))
        data = np.swapaxes(np.array([wave['wData']]), 1, 2)

        # Convert `note`, which is a bytestring of ASCII characters that
        # contains some metadata, to a list of strings
        note = wave['note']
        meta = note.decode('ASCII').split('\r')
        keys1 = [('Excitation Energy', 'hv', float),
                 ('Acquisition Mode', 'acq_mode', str),
                 ('Pass Energy', 'PE', int),
                 ('Lens Mode', 'lens_mode', str),
                 ('Step Time', 'DT', int),
                 ('Number of Sweeps', 'n_sweeps', int),
                 ('ThetaY', 'tilt', float)]
        meta_namespace = vars(self.read_pxt_ibw_metadata(keys1, meta))

        res = Namespace(
                data=data,
                xscale=xscale,
                yscale=yscale,
                zscale=zscale)

        for key in meta_namespace.keys():
            if not (key[0] == '_'):
                res.__setattr__(key, meta_namespace[key])

        return res

    @staticmethod
    def read_viewer(viewer):
        """ Extract the file ID from a SIS-ULTRA deflector mode output file. """
        for line in viewer.readlines():
            l = line.decode('UTF-8')
            if l.startswith('name'):
                # Make sure to split off unwanted whitespace
                return l.split('=')[1].split()[0]

    @staticmethod
    def read_metadata(keys, metadata_file):
        """ Read the metadata from a SIS-ULTRA deflector mode output file. """
        # List of interesting keys and associated variable names
        metadata = Namespace()
        for line in metadata_file.readlines():
            # Split at 'equals' sign
            tokens = line.decode('utf-8').split('=')
            for key, name, dtype in keys:
                # print(tokens[0])
                if tokens[0] == key:
                    # Split off whitespace or garbage at the end
                    value = tokens[1].split()[0]
                    # And cast to right type
                    value = dtype(value)
                    metadata.__setattr__(name, value)
                elif tokens[0] == 'Mode':
                    if tokens[1].split()[0] == 'ARPES' and tokens[1].split()[1] == 'Mapping':
                        metadata.__setattr__('scan_type', 'DA scan')
                elif tokens[0] == 'Thetay_Low':
                    metadata.__setattr__('scan_start', float(tokens[1].split()[0]))
                elif tokens[0] == 'Thetay_High':
                    metadata.__setattr__('scan_stop', float(tokens[1].split()[0]))
                elif tokens[0] == 'Thetay_StepSize':
                    metadata.__setattr__('scan_step', float(tokens[1].split()[0]))
        return metadata

    @staticmethod
    def read_pxt_ibw_metadata(keys, meta):
        metadata = Namespace()
        for line in meta:
            # Split at 'equals' sign
            tokens = line.split('=')
            for key, name, dtype in keys:
                if tokens[0] == key:
                    # Split off whitespace or garbage at the end
                    value = tokens[1].split()[0]
                    # And cast to right type
                    value = dtype(value)
                    metadata.__setattr__(name, value)
                elif tokens[0] == 'Mode':
                    if tokens[1].split()[0] == 'ARPES' and tokens[1].split()[1] == 'Mapping':
                        metadata.__setattr__('scan_type', 'DA scan')
                elif tokens[0] == 'Thetay_Low':
                    metadata.__setattr__('scan_start', float(tokens[1].split()[0]))
                elif tokens[0] == 'Thetay_High':
                    metadata.__setattr__('scan_stop', float(tokens[1].split()[0]))
                elif tokens[0] == 'Thetay_StepSize':
                    metadata.__setattr__('scan_step', float(tokens[1].split()[0]))
        return metadata


class DataloaderBloch(Dataloader):
    """
    Object that allows loading and saving of ARPES data from the SIS
    beamline at PSI which is in hd5 format.
    """
    name = 'Bloch'
    # Number of cuts that need to be present to assume the data as a map
    # instead of a series of cuts
    min_cuts_for_map = 1

    def __init__(self, filename=None):

        self.datfile = None
        pass

    def load_data(self, filename, metadata=False):
        """
        Extract and return the actual 'data', i.e. the recorded map/cut.
        Also return labels which provide some indications what the data means.
        """
        # Note: x and y are a bit confusing here as the hd5 file has a
        # different notion of zero'th and first dimension as numpy and then
        # later pcolormesh introduces yet another layer of confusion. The way
        # it is written now, though hard to read, turns out to do the right
        # thing and leads to readable code from after this point.

        ds = DataSet()
        if filename.endswith('zip'):
            filedata = self.load_zip(filename, metadata=metadata)
        elif filename.endswith('pxt'):
            filedata = self.load_pxt()
        else:
            raise NotImplementedError('File suffix not supported.')

        dict_ds = vars(ds.dataset)
        dict_filedata = vars(filedata)

        for attr in dir(filedata):
            if not (attr[0] == '_'):
                dict_ds[attr] = dict_filedata[attr]
        return ds.dataset

    def load_zip(self, filename, metadata=False):
        """ Load and store a deflector mode file from SIS-ULTRA. """
        # Prepare metadata key-value pairs for the different metadata files
        # and their expected types
        keys1 = [
            ('width', 'n_energy', int),
            ('height', 'n_y', int),
            ('depth', 'n_x', int),
            ('first_full', 'first_energy', int),
            ('last_full', 'last_energy', int),
            ('widthoffset', 'start_energy', float),
            ('widthdelta', 'step_energy', float),
            ('heightoffset', 'start_y', float),
            ('heightdelta', 'step_y', float),
            ('depthoffset', 'start_x', float),
            ('depthdelta', 'step_x', float)
        ]
        keys2 = [('Excitation Energy', 'hv', float),
                 ('Acquisition Mode', 'acq_mode', str),
                 ('Pass Energy', 'PE', int),
                 ('Lens Mode', 'lens_mode', str),
                 ('Step Time', 'DT', int),
                 ('Number of Sweeps', 'n_sweeps', int),
                 ('X', 'x', float),
                 ('Y', 'y', float),
                 ('Z', 'z', float),
                 ('A', 'phi', float),
                 ('P', 'theta', float),
                 ('T', 'tilt', float)]
        # print('elo')
        # Load the zipfile
        with zipfile.ZipFile(filename, 'r') as z:
            # Get the created filename from the viewer
            with z.open('viewer.ini') as viewer:
                file_id = self.read_viewer(viewer)
            # print('eloelo')
            # Get most metadata from a metadata file
            with z.open('Spectrum_' + file_id + '.ini') as metadata_file:
                M = self.read_metadata(keys1, metadata_file)
            # print('eloeloelo')
            # Get additional metadata from a second metadata file...
            with z.open(file_id + '.ini') as metadata_file2:
                M2 = self.read_metadata(keys2, metadata_file2)
                if not hasattr(M2, 'scan_type'):
                    M2.__setattr__('scan_type', 'cut')
                if not hasattr(M2, 'scan_start'):
                    M2.__setattr__('scan_start', 0)
                if not hasattr(M2, 'scan_stop'):
                    M2.__setattr__('scan_stop', 0)
                if not hasattr(M2, 'scan_step'):
                    M2.__setattr__('scan_step', 0)
                if hasattr(M2, 'scan_step'):
                    if M2.scan_step != 0:
                        n_dim_steps = np.abs(M2.scan_start - M2.scan_stop) / M2.scan_step
                        M2.n_sweeps /= n_dim_steps
                        M2.__setattr__('scan_dim', [M2.scan_start, M2.scan_stop, M2.scan_step])
                    else:
                        M2.scan_dim = []
                        M2.scan_type = 'cut'
                else:
                    pass
            # Extract the binary data from the zipfile
            # print('eloelo eloelo')
            if metadata:
                data_flat = np.zeros((int(M.n_y) * int(M.n_x) * int(M.n_energy)))
            else:
                with z.open('Spectrum_' + file_id + '.bin') as f:
                    data_flat = np.frombuffer(f.read(), dtype='float32')

        # Put the data back into its actual shape
        data = np.reshape(data_flat, (int(M.n_x), int(M.n_y), int(M.n_energy)))
        # Cut off unswept region
        data = data[:, :, M.first_energy:M.last_energy + 1]
        # Put into shape (energy, other angle, angle along analyzer)
        data = np.moveaxis(data, 2, 0)
        # Create axes
        xscale = start_step_n(M.start_x, M.step_x, M.n_x)
        yscale = start_step_n(M.start_y, M.step_y, M.n_y)
        energies = start_step_n(M.start_energy, M.step_energy, M.n_energy)
        energies = energies[M.first_energy:M.last_energy + 1]

        if yscale.size > 1:
            data = np.swapaxes(np.swapaxes(data, 0, 1), 1, 2)
        else:
            data = np.swapaxes(data, 0, 1)

        res = Namespace(
            data=data,
            xscale=xscale,
            yscale=yscale,
            zscale=energies
        )

        M2 = vars(M2)
        for key in M2.keys():
            if not (key[0] == '_'):
                res.__setattr__(key, M2[key])
        return res

    @staticmethod
    def read_viewer(viewer):
        """ Extract the file ID from a SIS-ULTRA deflector mode output file. """
        for line in viewer.readlines():
            l = line.decode('UTF-8')
            if l.startswith('name'):
                # Make sure to split off unwanted whitespace
                return l.split('=')[1].split()[0]

    @staticmethod
    def read_metadata(keys, metadata_file):
        """ Read the metadata from a SIS-ULTRA deflector mode output file. """
        # List of interesting keys and associated variable names
        metadata = Namespace()
        for line in metadata_file.readlines():
            # Split at 'equals' sign
            tokens = line.decode('utf-8').split('=')
            for key, name, dtype in keys:
                # print(tokens[0])
                if tokens[0] == key:
                    # Split off whitespace or garbage at the end
                    value = tokens[1].split()[0]
                    # And cast to right type
                    value = dtype(value)
                    metadata.__setattr__(name, value)
                elif tokens[0] == 'Mode':
                    if tokens[1].split()[0] == 'ARPES' and tokens[1].split()[1] == 'Mapping':
                        metadata.__setattr__('scan_type', 'DA scan')
                elif tokens[0] == 'Thetay_Low':
                    metadata.__setattr__('scan_start', float(tokens[1].split()[0]))
                elif tokens[0] == 'Thetay_High':
                    metadata.__setattr__('scan_stop', float(tokens[1].split()[0]))
                elif tokens[0] == 'Thetay_StepSize':
                    metadata.__setattr__('scan_step', float(tokens[1].split()[0]))
        return metadata


class Dataloaderi05(Dataloader):
    """
    Dataloader object for the i05 beamline at the Diamond Light Source.
    """
    name = 'i05'

    def load_data(self, fname, metadata=False):

        ds = DataSet()
        try:
            filedata = self.load_nxs(fname, metadata=metadata)
        except AttributeError:
            raise NotImplementedError

        dict_ds = vars(ds.dataset)
        dict_filedata = vars(filedata)

        for attr in dir(filedata):
            if not (attr[0] == '_'):
                dict_ds[attr] = dict_filedata[attr]

        return ds.dataset

    def load_nxs(self, filename, metadata):
        # Read file with h5py reader
        infile = h5py.File(filename, 'r')

        if metadata:
            data = np.zeros(infile['/entry1/analyser/data'].shape)
        else:
            data = np.array(infile['/entry1/analyser/data'])
        angles = np.array(infile['/entry1/analyser/angles'])
        energies = np.array(infile['/entry1/analyser/energies'])

        if len(energies.shape) == 2:
            zscale = energies[0]
        else:
            zscale = energies
        yscale = angles

        # Check if we have a scan
        if data.shape[0] == 1:
            xscale = np.array([0])
        else:
            # Otherwise, extract third dimension from scan command
            command = infile['entry1/scan_command'][()]

            # Special case for 'pathgroup'
            if command.split()[1] == 'pathgroup':
                self.print_m('is pathgroup')
                # Extract points from a ([polar, x, y], [polar, x, y], ...)
                # tuple
                points = command.split('(')[-1].split(')')[0]
                tuples = points.split('[')[1:]
                xscale = []
                for t in tuples:
                    point = t.split(',')[0]
                    xscale.append(float(point))
                xscale = np.array(xscale)

                # Now, if this was a scan with varying centre_energy, the
                # zscale contains a list of energies...
                # for now, just take the first one
            #                zscale = zscale[0]

            # Special case for 'scangroup'
            elif command.split()[1] == 'scan_group':
                self.print_m('is scan_group')
                # Extract points from a ([polar, x, y], [polar, x, y], ...)
                # tuple
                points = command.split('((')[-1].split('))')[0]
                points = '((' + points + '))'
                xscale = np.array(ast.literal_eval(points))[:, 0]

                # Now, if this was a scan with varying centre_energy, the
                # zscale contains a list of energies...
                # for now, just take the first one
                zscale = zscale[0]

            # "Normal" case
            else:
                start_stop_step = command.split()[2:5]
                start, stop, step = [float(s) for s in start_stop_step]
                xscale = np.arange(start, stop + 0.5 * step, step)

        # read metadata
        x = infile['entry1/instrument/manipulator/sax'][0]
        y = infile['entry1/instrument/manipulator/say'][0]
        z = infile['entry1/instrument/manipulator/saz'][0]
        theta = infile['entry1/instrument/manipulator/sapolar'][0]
        phi = infile['entry1/instrument/manipulator/saazimuth'][0]
        tilt = infile['entry1/instrument/manipulator/satilt'][0]

        PE = infile['entry1/instrument/analyser/pass_energy'][0]
        n_sweeps = infile['entry1/instrument/analyser/number_of_iterations'][0]
        lens_mode = str(infile['entry1/instrument/analyser/lens_mode'][0])[2:-1]
        acq_mode = str(infile['entry1/instrument/analyser/acquisition_mode'][0])[2:-1]
        DT = infile['entry1/instrument/analyser/time_for_frames'][0] * 1000

        hv = infile['entry1/instrument/monochromator/energy'][0]
        exit_slit = infile['entry1/instrument/monochromator/exit_slit_size'][0] * 1000
        FE = round(infile['entry1/instrument/monochromator/s2_horizontal_slit_size'][0], 2)
        polarization = str(infile['entry1/instrument/insertion_device/beam/final_polarisation_label'][0])[2:-1]
        temp = infile['entry1/sample/temperature'][0]
        pressure = infile['entry1/sample/lc_pressure'][0]

        # get scan info
        if infile['entry1/scan_dimensions'][0] == 1:
            scan_type = 'cut'
            scan_dim = None
        else:
            tmp = str(np.string_(infile['entry1/scan_command']))[2:-1].split()
            start, stop, step = float(tmp[2]), float(tmp[3]), float(tmp[4])
            scan_dim = [start, stop, step]
            if 'deflector' in tmp[1]:
                scan_type = 'DA'
            elif 'polar' in tmp[1]:
                scan_type = 'theta'
            elif 'energy' in tmp[1]:
                scan_type = 'hv'
            scan_type += ' scan'

        res = Namespace(
            data=data,
            xscale=xscale,
            yscale=yscale,
            zscale=zscale,
            ekin=None,
            kxscale=None,
            kyscale=None,
            x=x,
            y=y,
            z=z,
            theta=theta,
            phi=phi,
            tilt=tilt,
            temp=temp,
            pressure=pressure,
            hv=hv,
            # wf=None,
            # Ef=None,
            polarization=polarization,
            PE=PE,
            exit_slit=exit_slit,
            FE=FE,
            scan_type=scan_type,
            scan_dim=scan_dim,
            acq_mode=acq_mode,
            lens_mode=lens_mode,
            anal_slit=None,
            n_sweeps=n_sweeps,
            DT=DT
        )
        return res


######################################
######### Not updated yet ############
######################################


class DataloaderALS(Dataloader):
    """
    Object that allows loading and saving of ARPES data from the MAESTRO
    beamline at ALS, Berkely, in the newer .h5 format
    Organization of the ALS h5 file (June, 2018)::

        /-0D_Data
        | |
        | +-Cryostat_A
        | +-Cryostat_B
        | +-Cryostat_C
        | +-Cryostat_D
        | +-I0_NEXAFS
        | +-IG_NEXAFS
        | +-X                       <--- only present for xy scans (probably)
        | +-Y                       <--- "
        | +-Sorensen Program        <--- only present for dosing scans
        | +-time
        |
        +-1D_Data
        | |
        | +-Swept_SpectraN          <--- Not always present
        |
        +-2D_Data
        | |
        | +-Swept_SpectraN          <--- Usual location of data. There can be
        |                                several 'Swept_SpectraN', each with an
        |                                increasing value of N. The relevant data
        |                                seems to be in the highest numbered
        |                                Swept_Spectra.
        |
        +-Comments
        | |
        | +-PreScan
        |
        +-Headers
          |
          +-Beamline
          | |
          | +-[...]
          | +-EPU_POL               <--- Polarization (Integer encoded)
          | +-BL_E                  <--- Beamline energy (hv)
          | +-[...]
          |
          +-Computer
          +-DAQ_Swept
          | |
          | +-[...]
          | +-SSPE_0                <--- Pass energy (eV)
          | +-[...]
          | |
          +-FileFormat
          +-Low_Level_Scan
          +-Main
          +-Motors_Logical
          +-Motors_Logical_Offset
          +-Motors_Physical
          +-Motors_Sample           <--- Contains sample coordinates (xyz & angles)
          | |
          | +-[...]
          | +-SMOTOR3               <--- Theta
          | +-SMOTOR5               <--- Phi
          | +-[...]
          |
          +-Motors_Sample_Offset
          +-Notebook
    """
    name = 'ALS'

    def get(self, group, field_name):
        """
        Return the value of property *field_name* from h5File group *group*.
        *field_name* must be a bytestring (e.g. ``b'some_string'``)!
        This also returns a bytes object, remember to cast it correctly.
        Returns *None* if *field_name* was not found in *group*.
        """
        for entry in group[()]:
            if entry[1].strip() == field_name:
                return entry[2]

    def load_data(self, filename):
        h5file = h5py.File(filename, 'r')
        # The relevant data seems to be the highest numbered "Swept_SpectraN"
        # entry in "2D_Data". Find the highest N:
        Ns = [int(spectrum[-1]) for spectrum in list(h5file['2D_Data'])]
        i = np.argmax(Ns)
        path = '2D_Data/Swept_Spectra{}'.format(Ns[i])

        # Convert the data to a numpy array
        self.print_m('Converting data to numpy array (this may take a '
                     'while)...')
        data = np.array(h5file[path])

        # Get hv
        hv = self.get(h5file['Headers/Beamline/'], b'BL_E')
        hv = float(hv)

        # Build x-, y- and zscales
        # TO DO Detect cuts (i.e. factual 2D Datasets)
        nz, ny, nx = data.shape
        # xscale is the outer loop (if present)
        x_group = h5file['Headers/Low_Level_Scan']
        x_start = self.get(x_group, b'ST_0_0')
        x_end = self.get(x_group, b'EN_0_0')
        if x_start is not None:
            xscale = np.linspace(float(x_start), float(x_end), nx)
        else:
            xscale = np.arange(nx)
        # y- and zscale are the axis of one spectrum (i.e. pixels and energies)
        attributes = h5file[path].attrs
        y_start, z_start = attributes['scaleOffset']
        y_step, z_step = attributes['scaleDelta']
        yscale = start_step_n(y_start, y_step, ny)
        zscale = start_step_n(z_start, z_step, nz)
#        zscale, yscale, xscale = [np.arange(s) for s in shape]

        # TO DO pixel to angle conversion
        yscale *= 0.193/2

        # Get theta and phi
        theta = float(self.get(h5file['Headers/Motors_Sample'], b'SMOTOR3'))
        phi = float(self.get(h5file['Headers/Motors_Sample'], b'SMOTOR5'))

        res = Namespace(data=data,
                        xscale=xscale,
                        yscale=yscale,
                        zscale=zscale,
                        angles=yscale,
                        theta=theta,
                        phi=phi,
                        E_b=0,
                        hv=hv)
        return res


class DataloaderALSFits(Dataloader):
    """
    Object that allows loading and saving of ARPES data from the MAESTRO
    beamline at ALS, Berkely, which is in .fits format.
    """
    name = 'ALS .fits'
    # A factor required to reach a reasonable result for the k space
    # coordinates
    k_stretch = 1.05

    # Scanmode aliases
    CUT = 'null'
    MAP = 'Slit Defl'
    HV = 'mono_eV'
    DOPING = 'Sorensen Program'

    def __init__(self, work_func=4):
        # Assign a value for the work function
        self.work_func = work_func

        self.bintable = None

    def load_data(self, filename):
        # Open the file
        hdulist = pyfits.open(filename)

        # Access the BinTableHDU
        self.bintable = hdulist[1]

        # Try to fix FITS standard violations
        self.bintable.verify('fix')

        # Get the header to extract metadata from
        header = hdulist[0].header

        # Find out what scan mode we're dealing with
        scanmode = header['NM_0_0']
        self.print_m('Scanmode: ', scanmode)

        # Find out whether we are in fixed or swept (dithered) mode which has
        # an influence on how the data is stored in the fits file.
        swept = True if ('SSSW0' in header) else False

        # Case normal cut
        if scanmode == self.CUT:
            data = self.load_cut()
        # Case FSM
        elif scanmode == self.MAP:
            data = self.load_map(swept)
        # Case hv scan
        elif scanmode == self.HV:
            data = self.load_hv_scan()
        # Case doping scan
        elif scanmode == self.DOPING:
            # data = self.load_doping_scan()
            data = self.load_hv_scan()
        else:
            raise(IndexError('Couldn\'t determine scan type'))

        nz, ny, nx = data.shape

        # Determine whether the file uses the SS or SF prefix for metadata
        if 'SSX0_0' in header:
            pre = 'SS'
        elif 'SFX0_0' in header:
            pre = 'SF'
        else:
            raise(IndexError('Neither SSX0_0 nor SFX0_0 appear in header.'))

        # Starting pixel (energy scale)
        e0 = header[pre+'X0_0']
        # Ending pixel (energy scale)
        e1 = header[pre+'X1_0']
        # Pixels per eV
        ppeV = header[pre+'PEV_0']
        # Zero pixel (=Fermi level?)
        fermi_level = header[pre+'KE0_0']

        # Create the energy (y) scale
        energy_binning = 2
        energies_in_px = np.arange(e0, e1, energy_binning)
        energies = (energies_in_px - fermi_level)/ppeV

        # Starting and ending pixels (angle or ky scale)
        # DEPRECATED
        # y0 = header[pre+'Y0_0']
        # y1 = header[pre+'Y1_0']

        # Use this arbitrary seeming conversion factor (found in Denys'
        # script) to get from pixels to angles
        # deg_per_px = 0.193 / 2
        deg_per_px = 0.193 * self.k_stretch
        # angle_binning = 3
        angle_binning = 2

        # angles_in_px = np.arange(y0, y1, angle_binning)
        # n_y = int((y1-y0)/angle_binning)
        # angles_in_px = np.arange(0, n_y, 1)
        angles_in_px = np.arange(0, ny, 1)
        # NOTE
        # Actually, I would think that we have to multiply by
        # `angle_binning` to get the right result. That leads to
        # unreasonable results, however, and division just gives a
        # perfect looking output. Maybe the factor 0.193/2 from ALS has
        # changed with the introduction of binning?
        # angles = angles_in_px * deg_per_px / angle_binning
        angles = angles_in_px * deg_per_px / angle_binning

        # Case cut
        if scanmode == self.CUT:
            # NOTE x and y scale may need to be swapped
            yscale = energies
            xscale = angles
            zscale = None
        # Case map
        elif scanmode == self.MAP:
            x0 = header['ST_0_0']
            x1 = header['EN_0_0']
            # n_x = header['N_0_0']
            # xscale = np.linspace(x0, x1, n_x) * self.k_stretch
            xscale = np.linspace(x0, x1, nx)  # * self.k_stretch
            yscale = angles
            zscale = energies
        # Case hv scan
        elif scanmode == self.HV:
            z0 = header['ST_0_0']
            z1 = header['EN_0_0']
            # nz = header['N_0_0']
            angles_in_px = np.arange(0, nx, 1)
            angles = angles_in_px * deg_per_px / angle_binning
            xscale = angles
            yscale = energies
            zscale = np.linspace(z0, z1, nz)
        # Case doping
        elif scanmode == self.DOPING:
            z0 = header['ST_0_0']
            z1 = header['EN_0_0']
            # Not sure how the sqrt2 comes in
            angles_in_px = np.arange(0, nx, 1) * np.sqrt(2)
            angles = angles_in_px * deg_per_px / angle_binning
            xscale = angles
            # For some reason the energy determination from file fails here...
            yscale = np.arange(ny, dtype=float)
            zscale = np.linspace(z0, z1, nz)

        # For the binding energy, just take a min value as its variations
        # are small compared to the photon energy
        E_b = energies.min()

        # Extract some additional metadata (mostly for angles->k conversion)
        # TO DO Special cases for different scan types
        # Get relevant metadata from header
        theta = header['LMOTOR3']
        # beta in LMOTOR4
        phi = header['LMOTOR5']

        # The photon energy may vary in the case that this is a hv_scan
        hv = header['MONO_E']

        # In recent ALS data the above determined x, y and z scales did not
        # match the actual shape of the data...
#        self.print_m(*data.shape)
#        self.print_m(nx, ny, nz)
        scales = [zscale, yscale, xscale]
        for i, scale in enumerate(scales):
            try:
                length = len(scale)
            except Exception:
                self.print_m('length problem')
                continue
#            self.print_m(length)
            n = data.shape[i]
            if length != n:
                self.print_m(('Shape mismatch in dim {}: {} != {}. Setting ' +
                              'scale to arange.').format(i, length, n))
                scales[i] = np.arange(n)
        zscale, yscale, xscale = scales

        # NOTE angles==xscale and E_b can be extracted from yscale, thus it
        # is not really necessary to pass them on here.
        res = Namespace(
               data=data,
               xscale=xscale,
               yscale=yscale,
               zscale=zscale,
               angles=angles,
               theta=theta,
               phi=phi,
               E_b=E_b,
               hv=hv
        )
        return res

    def load_cut(self):
        """ Read data from a 'cut', which is just one slice of energy vs k. """
        # Access the actual data which is in the last element of the last
        # element of the bin table...
        fields = self.bintable.data[-1]
        data = fields[-1]

        # Reshape towards GUI expectations
        x, y = data.shape
        data = data.reshape(1, x, y)

        return data

    def load_map(self, swept):
        """
        Read data from a 'map', i.e. several energy vs k slices and bring
        them in the right shape for the gui, which is
        (energy, k_parallel, k_perpendicular)
        """
        bt_data = self.bintable.data
        n_slices = len(bt_data)

        first_slice = bt_data[0][-1]
        # The shape of the returned array must be (energy, kx, ky)
        # (n_slices corresponds to ky)
        if swept:
            n_energy, n_kx = first_slice.shape
            data = np.zeros([n_energy, n_kx, n_slices])
            for i in range(n_slices):
                this_slice = bt_data[i][-1]
                data[:, :, i] = this_slice
        else:
            n_kx, n_energy = first_slice.shape
            data = np.zeros([n_energy, n_kx, n_slices])
            for i in range(n_slices):
                this_slice = bt_data[i][-1]
                # Reshape the slice to (energy, kx)
                this_slice = this_slice.transpose()
                data[:, :, i] = this_slice

        # Convert to numpy array
        data = np.array(data)

        return data

    def load_hv_scan(self):
        """
        Read data from a hv scan, i.e. a series of energy vs k cuts
        (shape (n_kx, n_energy), each belonging to a different photon energy
        hv. The returned shape in this case must be
        (photon_energies, energy, k)
        The same is used for doping scans, where the output shape is
        (doping, energy, k)
        """
        bt_data = self.bintable.data
        n_hv = len(bt_data)
        # = header['N_0_0']

        first_cut = bt_data[0][-1]
        n_kx, n_energy = first_cut.shape  # should all be accessible from header

        data = np.zeros([n_hv, n_kx, n_energy])
        for i in range(n_hv):
            this_slice = bt_data[i][-1]
            data[i, :, :] = this_slice

        # Convert to numpy array
        data = np.array(data)

        return data


class DataloaderADRESS(Dataloader):
    """ ADRESS beamline at SLS, PSI. """
    name = 'ADRESS'

    def load_data(self, filename):
        h5file = h5py.File(filename, 'r')
        # The actual data is in the field: 'Matrix'
        matrix = h5file['Matrix']

        # The scales can be extracted from the matrix' attributes
        scalings = matrix.attrs['IGORWaveScaling']
        # units = matrix.attrs['IGORWaveUnits']
        info = matrix.attrs['IGORWaveNote']

        # Convert `units` and `info`, which is a bytestring of ASCII characters, to lists of strings
        # Put the data into a numpy array and convert to float
        data = np.array(matrix, dtype=float)
        shape = data.shape

        # 'IGORWaveUnits' contains a list of the form ['', 'degree', 'eV', units[3]]. The first three elements should
        # always be the same, but the third one may vary or not even exist. Use this to determine the scan type. Convert
        # to np.array bring it in the shape the gui expects, which is [energy/hv/1, kparallel/"/kparallel,
        # kperpendicular/"/energy] and prepare the x,y,z # scales Note: `scalings` is a 3/4 by 2 array where every line
        # contains a (step, start) pair
        if len(shape) == 3:
            # Case map or hv scan (or...?)
            data = np.rollaxis(data.T, 2, 1)
            shape = data.shape
            # Shape has changed
            xstep, xstart = scalings[3]
            ystep, ystart = scalings[1]
            zstep, zstart = scalings[2]
            xscale = start_step_n(xstart, xstep, shape[0])
            yscale = start_step_n(ystart, ystep, shape[1])
            zscale = start_step_n(zstart, zstep, shape[2])
        else:
            # Case cut
            # Make data 3-dimensional by adding an empty dimension
            data = data.reshape(1, shape[0], shape[1])
            # Shape has changed
            shape = data.shape
            ystep, ystart = scalings[1]
            zstep, zstart = scalings[2]
            xscale = np.array([1])
            yscale = start_step_n(ystart, ystep, shape[1])
            zscale = start_step_n(zstart, zstep, shape[2])

        res = Namespace(
               data=data,
               xscale=xscale,
               yscale=yscale,
               zscale=zscale
        )

        # more metadata
        metadata_list = info.decode('ASCII').split('\n')
        keys1 = [('hv', 'hv', float),
                 ('Pol', 'polarization', str),
                 ('Epass', 'PE', int),
                 ('X ', 'x', int),
                 ('Y-orig', 'y', int),
                 ('Z-orig', 'z', int),
                 ('Theta', 'theta', float),
                 ('Azimuth', 'phi', int),
                 ('Tilt', 'tilt', float),
                 ('Temp', 'temp', int),
                 ('Slit', 'exit_slit', str)]
        self.read_metadata(keys1, metadata_list, res)
        if xscale.size == 1:
            res.__setattr__('scan_type', 'cut')

        return res

    @staticmethod
    def read_metadata(keys, metadata_list, namespace):
        """ Read the metadata from a SIS-ULTRA deflector mode output file. """
        # List of interesting keys and associated variable names
        metadata = namespace
        for line in metadata_list:
            # Split at 'equals' sign
            tokens = line.split('=')
            for key, name, dtype in keys:
                if key in tokens[0]:
                    if 'Tilt' in tokens[0] and ':' in tokens[1]:
                        metadata.__setattr__('scan_type', 'Tilt scan')
                        start, step, stop = tokens[1].split(':')
                        metadata.__setattr__('scan_dim', [start, stop, step])
                        metadata.__setattr__('tilt', start)
                    elif 'hv' in tokens[0] and ':' in tokens[1]:
                        metadata.__setattr__('scan_type', 'hv scan')
                        start, step, stop = tokens[1].split(':')
                        metadata.__setattr__('scan_dim', [start, stop, step])
                        metadata.__setattr__('hv', start)
                    # Split off whitespace or garbage at the end
                    else:
                        value = tokens[1].split()[0]
                        metadata.__setattr__(name, value)
        return metadata


class DataloaderCASSIOPEE(Dataloader):
    """ CASSIOPEE beamline at SOLEIL synchrotron, Paris. """
    name = 'CASSIOPEE'
    date = '18.07.2018'

    # Possible scantypes
    HV = 'hv scan'
    FSM = 'Theta scan'

    def load_data(self, filename):
        """
        Single cuts are stored as two files: One file contians the data and
        the other the metadata. Maps, hv scans and other *external
        loop*-scans are stored as a directory containing these two files for
        each cut/step of the external loop. Thus, this dataloader
        distinguishes between directories and single files and changes its
        behaviour accordingly.
        """
        if os.path.isfile(filename):
            return self.load_from_file(filename)
        else:
            if not filename.endswith('/'):
                filename += '/'
            return self.load_from_dir(filename)

    def load_from_dir(self, dirname):
        """
        Load 3D data from a directory as it is output by the IGOR macro used
        at CASSIOPEE. The dir is assumed to contain two files for each cut::

            BASENAME_INDEX_i.txt     -> beamline related metadata
            BASENAME_INDEX_ROI1_.txt -> data and analyzer related metadata

        To be more precise, the assumptions made on the filenames in the
        directory are:

            * the INDEX is surrounded by underscores (`_`) and appears after
              the first underscore.
            * the string ``ROI`` appears in the data filename.
        """
        # Get the all filenames in the dir
        all_filenames = os.listdir(dirname)
        # Remove all non-data files
        filenames = []
        for name in all_filenames:
            if '_1_i' in name:
                metadata_file = open(dirname + name)
            if 'ROI' in name:
                filenames.append(name)

        # Get metadata from first file in list
        skip, energy, angles = self.get_metadata(dirname + filenames[0])
        keys = [('hv (eV) ', 'hv', float),
                ('x (mm) ', 'x', float),
                ('y (mm) ', 'y', float),
                ('z (mm) ', 'z', float),
                ('theta (deg) ', 'theta', float),
                ('phi (deg) ', 'phi', float),
                ('tilt (deg) ', 'tilt', float),
                ('InputB ', 'temp', float),
                ('P(mbar) ', 'pressure', float),
                ('Polarisation [0', 'polarization', str)]
        md = self.read_metadata(keys, metadata_file)

        # Get the data from each cut separately. This happens in the order
        # they appear in os.listdir() which is usually not what we want -> a
        # reordering is necessary later.
        unordered = {}
        i_min = np.inf
        i_max = -np.inf
        for name in filenames:
            # Keep track of the min and max indices in the directory
            i = int(name.split('_')[1])
            if i < i_min:
                i_min = i
            if i > i_max:
                i_max = i

            # Get the data of cut i
            this_cut = np.loadtxt(dirname+name, skiprows=skip+1)[:, 1:]
            unordered.update({i: this_cut})

        # Properly rearrange the cuts
        data = []
        for i in range(i_min, i_max+1):
            data.append(np.array(unordered[i]).T)
        data = np.array(data)

        # Get the z-axis from the metadata files
        scan_type, outer_loop, hv, thetas = self.get_outer_loop(dirname, filenames)
        thetas = sorted(thetas)
        if scan_type == self.HV:
            xscale = outer_loop
            scan_start = hv[0]
            scan_stop = hv[-1]
            scan_step = np.abs(hv[0] - hv[1])
        elif scan_type == self.FSM:
            xscale = outer_loop
            scan_start = thetas[0]
            scan_stop = thetas[-1]
            scan_step = np.abs(thetas[0] - thetas[1])
        else:
            xscale = np.arange(data.shape[0])
            scan_start = 0
            scan_stop = 0
            scan_step = 0
        yscale = angles
        zscale = energy

        res = Namespace(
            data=data,
            xscale=xscale,
            yscale=yscale,
            zscale=zscale,
            ekin=zscale,
            hv=float(md.hv),
            x=float(md.x),
            y=float(md.y),
            z=float(md.z),
            theta=float(md.theta),
            phi=float(md.phi),
            tilt=float(md.tilt),
            temp=float(md.temp),
            pressure=float(md.pressure),
            polarization=md.polarization,
            # PE=M2.PE,
            # exit_slit=exit_slit,
            # FE=FE,
            scan_type=scan_type,
            scan_dim=[scan_start, scan_stop, scan_step]
            # lens_mode=M2.lens_mode,
            # acq_mode=M2.acq_mode
        )
        return res

    def load_from_file(self, filename):
        """
        Load just a single cut. However, at CASSIOPEE they output .ibw files
        if the cut does not belong to a scan...
        """
        if filename.endswith('.ibw'):
            return self.load_from_ibw(filename)
        elif filename.endswith('pxt'):
            return self.load_pxt(filename)
        else:
            return self.load_from_txt(filename)

    def load_from_ibw(self, filename):
        """
        Load scan data from an IGOR binary wave file. Luckily someone has
        already written an interface for this (the python `igor` package).
        """
        wave = binarywave.load(filename)['wave']
        data = np.array([wave['wData']])

        # The `header` contains some metadata
        header = wave['wave_header']
        nDim = header['nDim']
        steps = header['sfA']
        starts = header['sfB']

        # Construct the x and y scales from start, stop and n
        yscale = start_step_n(starts[0], steps[0], nDim[0])
        xscale = start_step_n(starts[1], steps[1], nDim[1])

        # Convert `note`, which is a bytestring of ASCII characters that
        # contains some metadata, to a list of strings
        note = wave['note']
        note = note.decode('ASCII').split('\r')

        # Now the extraction fun begins. Most lines are of the form
        # `Some-kind-of-name=some-value`
        metadata = dict()
        for line in note:
            # Split at '='. If it fails, we are not in a line that contains
            # useful information
            try:
                name, val = line.split('=')
            except ValueError:
                continue
            # Put the pair in a dictionary for later access
            metadata.update({name: val})

        # NOTE Unreliable hv
        hv = metadata['Excitation Energy']
        res = Namespace(
                data=data,
                xscale=xscale,
                yscale=yscale,
                zscale=None,
                angles=xscale,
                theta=0,
                phi=0,
                E_b=0,
                hv=hv)
        return res

    def load_from_txt(self, filename):
        i, energy, angles = self.get_metadata(filename)
        # self.print_m('Loading from txt')
        data0 = np.loadtxt(filename, skiprows=i+1)
        # The first column in the datafile contains the angles
        # angles_from_data = data0[:, 0]
        data = np.array([data0[:, 1:]])

        res = Namespace(
            data=data,
            xscale=angles,
            yscale=energy,
            zscale=None,
            angles=angles,
            theta=1,
            phi=1,
            E_b=0,
            hv=1)
        return res

    def get_metadata(self, filename):
        """
        Extract some of the metadata stored in a CASSIOPEE output text file.
        Also try to detect the line number below which the data starts (for
        np.loadtxt's skiprows.)

        **Returns**

        ======  ================================================================
        i       int; last line number still containing metadata.
        energy  1D np.array; energy (y-axis) values.
        angles  1D np.array; angle (x-axis) values.
        hv      float; photon energy for this cut.
        ======  ================================================================
        """
        metadata = Namespace()
        with open(filename, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if line.startswith('Dimension 1 scale='):
                    energy = line.split('=')[-1].split()
                    energy = np.array(energy, dtype=float)
                elif line.startswith('Dimension 2 scale='):
                    angles = line.split('=')[-1].split()
                    angles = np.array(angles, dtype=float)
                elif line.startswith('Excitation Energy'):
                    # NOTE this hv does not reflect the actually used hv
                    # hv = float(line.split('=')[-1])
                    pass
                elif line.startswith('inputA') or line.startswith('[Data'):
                    # this seems to be the last line before the data
                    break
        return i, energy, angles

    def read_metadata(self, keys, metadata_file):
        """ Read the metadata from a SIS-ULTRA deflector mode output file. """
        # List of interesting keys and associated variable names
        metadata = Namespace()
        for line in metadata_file.readlines():
            # Split at 'equals' sign
            tokens = line.split(':')
            for key, name, dtype in keys:
                if tokens[0] == key:
                    if hasattr(metadata, name):
                        pass
                    else:
                        # Split off whitespace or garbage at the end
                        value = tokens[-1][1:-1]
                        # And cast to right type
                        if key == 'Polarisation [0':
                            if value == '0':
                                metadata.__setattr__(name, 'LV')
                            elif value == '1':
                                metadata.__setattr__(name, 'LH')
                            elif value == '2':
                                metadata.__setattr__(name, 'AV')
                            elif value == '3':
                                metadata.__setattr__(name, 'AH')
                            elif value == '4':
                                metadata.__setattr__(name, 'CR')
                            else:
                                pass
                        else:
                            metadata.__setattr__(name, value)
        return metadata

    def get_outer_loop(self, dirname, filenames):
        """
        Try to determine the scantype and the corresponding z-axis scale from
        the additional metadata textfiles. These follow the assumptions made
        in :meth:`self.load_from_dir
        <arpys.dataloaders.Dataloader_CASSIOPEE.load_from_dir>`.
        Additionally, the MONOCHROMATOR section must come before the
        UNDULATOR section as in both sections we have a key `hv` but only the
        former makes sense.
        Return a string for the scantype, the extracted z-scale and the value
        for hv for non-hv-scans (scantype, zscale, hvs[0]) or (None,
        None, hvs[0]) in case of failure.
        """
        # Step 1) Extract metadata from metadata file
        # Prepare containers
        indices, xs, ys, zs, thetas, phis, tilts, hvs = ([], [], [], [], [],
                                                         [], [], [])
        containers = [indices, xs, ys, zs, thetas, phis, tilts, hvs]
        for name in filenames:
            # Get the index of the file
            index = int(name.split('_')[1])

            # Build the metadata-filename by substituting the ROI part with i
            metafile = re.sub(r'_ROI.?_', '_i', name)

            # The values are separated from the names by a `:`
            splitchar = ':'

            # Read in the file
            with open(dirname + metafile, 'r') as f:
                for line in f.readlines():
                    if line.startswith('x (mm)'):
                        x = float(line.split(splitchar)[-1])
                    elif line.startswith('y (mm)'):
                        y = float(line.split(splitchar)[-1])
                    elif line.startswith('z (mm)'):
                        z = float(line.split(splitchar)[-1])
                    elif line.startswith('theta (deg)'):
                        theta = float(line.split(splitchar)[-1])
                    elif line.startswith('phi (deg)'):
                        phi = float(line.split(splitchar)[-1])
                    elif line.startswith('tilt (deg)'):
                        tilt = float(line.split(splitchar)[-1])
                    elif line.startswith('hv (eV)'):
                        hv = float(line.split(splitchar)[-1])
                    elif line.startswith('UNDULATOR'):
                        break
            # NOTE The order of this list has to match the order of the
            # containers
            values = [index, x, y, z, theta, phi, tilt, hv]
            for i, container in enumerate(containers):
                container.append(values[i])

        # Step 2) Check which parameters vary to determine scantype
        if ((hvs[1] - hvs[0]) > 0.4):
            scantype = self.HV
            zscale = hvs
        elif thetas[1] != thetas[0]:
            scantype = self.FSM
            zscale = thetas
        else:
            scantype = None
            zscale = None

        # Step 3) Put zscale in order and return
        if zscale is not None:
            zscale = np.array(zscale)[np.argsort(indices)]

        return scantype, zscale, hvs, thetas

    @staticmethod
    def load_pxt(filename):
        """ Load and store the full h5 file and extract relevant information. """
        pxt = igorpy.load(filename)[0]
        data = pxt.data.T
        shape = data.shape
        notes = str(pxt.notes)
        for entry in notes[2:-1].split('\\r'):
            tokens = entry.split('=')
            if tokens[0] == 'Lens Mode':
                lens_mode = tokens[1]
            elif tokens[0] == 'Excitation Energy':
                hv = float(tokens[1])
            elif tokens[0] == 'Acquisition Mode':
                acq_mode = tokens[1]
            elif tokens[0] == 'Pass Energy':
                PE = tokens[1]
            elif tokens[0] == 'Lens Mode':
                lens_mode = tokens[1]
            elif tokens[0] == 'Lens Mode':
                lens_mode = tokens[1]

        if len(shape) == 2:
            x = 1
            y = shape[0]
            N_E = shape[1]
            # Make data 3D
            data = data.reshape((1, y, N_E))
            # Extract the limits
            xlims = [1, 1]
            ylims = [pxt.axis[1][-1], pxt.axis[1][0]]
            elims = [pxt.axis[0][-1], pxt.axis[0][0]]
            xscale = start_step_n(*xlims, x)
            yscale = start_step_n(*ylims, y)
            energies = start_step_n(*elims, N_E)
        else:
            'Sorry, only cuts reading of .pxt files is working.'
            return

        res = Namespace(
            data=data,
            xscale=xscale,
            yscale=yscale,
            zscale=energies,
            hv=hv,
            ekin=energies,
            PE=PE,
            scan_type='cut',
            lens_mode=lens_mode,
            acq_mode=acq_mode
        )
        return res


# +-------+ #
# | Tools | # ==================================================================
# +-------+ #

# List containing all reasonably defined dataloaders
all_dls = [
    DataloaderPickle,
    DataloaderBloch,
    DataloaderSIS,
    DataloaderADRESS,
    Dataloaderi05,
    DataloaderCASSIOPEE,
    DataloaderALS,
    DataloaderALSFits]


# Function to try all dataloaders in all_dls
def load_data(filename, metadata=False, exclude=None, suppress_warnings=False):
    """
    Try to load some dataset *filename* by iterating through `all_dls` 
    and appliyng the respective dataloader's load_data method. If it works: 
    great. If not, try with the next dataloader. 
    Collects and prints all raised exceptions in case that no dataloader 
    succeeded.
    """
    # Sanity check: does the given path even exist in the filesystem?
    if not os.path.exists(filename):
        raise FileNotFoundError(ENOENT, os.strerror(ENOENT), filename) 

    # If only a single string is given as exclude, pack it into a list
    if exclude is not None and type(exclude) == str:
        exclude = [exclude]
    
    # Keep track of all exceptions in case no loader succeeds
    exceptions = dict()

    # Suppress warnings
    with catch_warnings():
        if suppress_warnings:
            simplefilter('ignore')
        for dataloader in all_dls:
            # Instantiate a dataloader object
            dl = dataloader()

            # Skip to the next if this dl is excluded (continue brings us 
            # back to the top of the loop, starting with the next element)
            if exclude is not None and dl.name in exclude:
                continue

            # Try loading the data
            try:
                namespace = dl.load_data(filename, metadata=metadata)
            except Exception as e:
                # Temporarily store the exception
                exceptions.update({dl: e})
                # Try the next dl
                continue

            # Reaching this point must mean we succeeded. Print warnings from this dataloader, if any occurred
            # print('Loaded data with {}.'.format(dl))
            try:
                print(dl, ': ', exceptions[dl])
            except KeyError:
                pass
            
            return namespace


# Function to create a python pickle file from a data namespace
def dump(data, filename, force=False):
    """ Wrapper for :func:`pickle.dump`. Does not overwrite if a file of 
    the given name already exists, unless *force* is True.

    **Parameters**

    ========  ==================================================================
    D         python object to be stored.
    filename  str; name of the output file to create.
    force     boolean; if True, overwrite existing file.
    ========  ==================================================================
    """
    # Check if file already exists
    if not force and os.path.isfile(filename):
        question = 'File <{}> exists. Overwrite it? (y/N)'.format(filename)
        answer = input(question)
        # If the answer is anything but a clear affirmative, stop here
        if answer.lower() not in ['y', 'yes']:
            return

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    message = 'Wrote to file <{}>.'.format(filename)
    print(message)


def load_pickle(filename):
    """ Shorthand for loading python objects stored in pickle files.

    **Parameters**

    ========  ==================================================================
    filename  str; name of file to load.
    ========  ==================================================================
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


def update_namespace(data, *attributes):
    """ Add arbitrary attributes to a :class:`Namespace <argparse.Namespace>`.

    **Parameters**

    ==========  ================================================================
    D           argparse.Namespace; the namespace holding the data and 
                metadata. The format is the same as what is returned by a 
                dataloader.
    attributes  tuples or len(2) lists; (name, value) pairs of the attributes 
                to add. Where `name` is a str and value any python object.
    ==========  ================================================================
    """
    for name, attribute in attributes:
        data.__dict__.update({name: attribute})


def add_attributes(filename, *attributes):
    """ Add arbitrary attributes to an argparse.Namespace that is stored as a 
    python pickle file. Simply opens the file, updates the namespace with 
    :func:`update_namespace <arpys.dataloaders.update_namespace>` and writes 
    back to file.
  
    **Parameters**

    ==========  ================================================================
    filename    str; name of the file to update.
    attributes  tuples or len(2) lists; (name, value) pairs of the attributes 
                to add. Where `name` is a str and value any python object.
    ==========  ================================================================
    """
    dataloader = DataloaderPickle()
    data = dataloader.load_data(filename)
    
    update_namespace(data, *attributes)
    
    dump(data, filename, force=True)


def reshape_pickled(fname):

    scan = load_pickle(fname)
    print(fname)
    print(f'org data shape: {scan.data.shape}')
    print(f'org xscale shape:{scan.xscale.size}')
    print(f'org yscale shape:{scan.yscale.size}')
    print(f'org zscale shape:{scan.zscale.size}')
    if np.any((scan.data.shape[0] == 1, scan.data.shape[1] == 1, scan.data.shape[2] == 1)):
        tmp = deepcopy(scan)
        scan.xscale = tmp.zscale
        scan.yscale = tmp.xscale
        scan.zscale = tmp.yscale
        scan.data = np.ones((1, tmp.xscale.size, tmp.yscale.size))
        scan.data[0, :, :] = np.rollaxis(tmp.data[0, :, :], 1)
        print(f'new data shape: {scan.data.shape}')
        print(f'new xscale shape:{scan.xscale.size}')
        print(f'new yscale shape:{scan.yscale.size}')
        print(f'new zscale shape:{scan.zscale.size}')
        dump(scan, (fname + '_rs'))
    else:
        tmp = deepcopy(scan)
        scan.data = tmp.data.T
        dump(scan, (fname + '_rs'))

