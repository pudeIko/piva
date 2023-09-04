"""

"""

import piva.data_loader as dl
import argparse
import os

# PATH = os.path.expanduser('~') + '/piva-paper/tests/'
PATH = './tests/data/'


def test_dataloaders():
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
        elif 'merlin' in fname and not fname.endswith('txt'):
            dli = dl.DataloaderMERLIN()
        else:
            continue
        print(dli)
        ds = dli.load_data(PATH + fname)
        # check if the mandatory attributes are loaded correctly
        assert isinstance(ds, argparse.Namespace)
        assert len(ds.data.shape) == 3
        assert ds.xscale.size == ds.data.shape[0]
        assert ds.yscale.size == ds.data.shape[1]
        assert ds.zscale.size == ds.data.shape[2]
