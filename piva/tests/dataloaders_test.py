"""
Automated test for all Dataloaders from implemented beamlines.
"""
import numpy as np
import piva.data_loaders as dl
import argparse
import os
import pkg_resources

PATH = pkg_resources.resource_filename('piva', 'tests/data/')


def test_dataloaders() -> None:
    """
    Run test.
    """

    print()
    for fname in sorted(os.listdir(PATH)):
        if 'pickle' in fname:
            dli = dl.DataloaderPickle()
        elif 'sis' in fname:
            dli = dl.DataloaderSIS()
        elif 'bloch' in fname:
            dli = dl.DataloaderBloch()
        elif 'adress' in fname:
            dli = dl.DataloaderADRESS()
        elif 'i05' in fname:
            dli = dl.DataloaderI05()
        elif 'cassiopee' in fname:
            dli = dl.DataloaderCASSIOPEE()
        elif 'merlin' in fname and (not fname.endswith('txt')):
            dli = dl.DataloaderMERLIN()
        elif 'uranos' in fname:
            dli = dl.DataloaderURANOS()
        else:
            continue
        print(f'{fname:{30}}{dli}')
        ds = dli.load_data(PATH + fname)
        # check if the mandatory attributes are loaded correctly
        assert isinstance(ds, argparse.Namespace)
        assert type(ds.data) is np.ndarray
        assert len(ds.data.shape) == 3
        assert type(ds.xscale) is np.ndarray
        assert ds.xscale.size == ds.data.shape[0]
        assert type(ds.yscale) is np.ndarray
        assert ds.yscale.size == ds.data.shape[1]
        assert type(ds.zscale) is np.ndarray
        assert ds.zscale.size == ds.data.shape[2]
        assert type(ds.ekin) is None or np.ndarray
        assert type(ds.kxscale) is None or np.ndarray
        assert type(ds.kyscale) is None or np.ndarray
        assert type(ds.x) is None or float
        assert type(ds.y) is None or float
        assert type(ds.z) is None or float
        assert type(ds.theta) is None or float
        assert type(ds.phi) is None or float
        assert type(ds.tilt) is None or float
        assert type(ds.temp) is None or float
        assert type(ds.pressure) is None or float
        assert type(ds.hv) is None or float
        assert type(ds.wf) is None or float
        assert type(ds.Ef) is None or float
        assert type(ds.polarization) is None or str
        assert type(ds.PE) is None or int
        assert type(ds.exit_slit) is None or int
        assert type(ds.FE) is None or int
        assert type(ds.scan_type) is None or str
        assert type(ds.scan_dim) is None or list
        assert type(ds.acq_mode) is None or str
        assert type(ds.lens_mode) is None or str
        assert type(ds.ana_slit) is None or str
        assert type(ds.defl_angle) is None or float
        assert type(ds.n_sweeps) is None or int
        assert type(ds.DT) is None or int
        assert type(ds.date) is None or str
        assert type(ds.data_provenance) is dict


if __name__ == "__main__":
    import pytest

    path = pkg_resources.resource_filename('piva', 'tests/dataloaders_test.py')
    pytest.main(['-v', '-s', path])
