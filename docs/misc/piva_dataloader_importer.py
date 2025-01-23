import numpy as np
from piva.data_loaders import Dataloader, Dataset
from piva.data_browser import DataBrowser
import h5py


class DataloaderImporter:
    """
    Dataloader importer for custom **Dataloaders** written for PIVA.
    """

    def __init__(self, data_browser: DataBrowser) -> None:
        """
        Initialize DataloaderImporter.

        :param data_browser: `DataBrowser` of the current session
        """

        self.db = data_browser
        self.db.dp_dl_picker.insertSeparator(self.db.dp_dl_picker.count())

        self.add_dataloader1()

    def add_dataloader1(self) -> None:
        """
        Example method for importing custom Dataloader.
        """

        loader = CustomDataloader
        dl_label = '{}'.format(loader.name)

        self.db.dp_dl_picker.addItem(dl_label)
        self.db.add_dataloader_to_record(dl_label, loader)

        print('\t', dl_label)


class CustomDataloader(Dataloader):
    """
    Simple example of a custom Dataloader.
    """

    name = 'Custom_Dataloader'

    def __init__(self) -> None:
        super(CustomDataloader, self).__init__()

    # kwarg "metadata" necessary to match arguments of all other data loaders
    def load_data(self, filename: str, metadata: bool = False) -> Dataset:
        """
        Recognize correct format and load data from the file.

        :param filename: absolute path to the file
        :param metadata: if :py:obj:`True`, read only metadata and size of the
                         dataset. Not used here, but required to mach format
                         of other **Dataloaders**. See :meth:`load_ses_zip`
                         for more info.
        :return: loaded dataset with available metadata
        """

        # dummy *if* condition to make the example work. One should remove
        # 'isinstance' part and change extension in 'endswith'
        if filename.endswith('h5') or isinstance(filename, str):
            self.custom_loading_method(filename, metadata=metadata)
        else:
            raise NotImplementedError

        self.ds.add_org_file_entry(filename, self.name)
        return self.ds

    def custom_loading_method(self, filename: str, metadata: bool = False):
        """
        This method is intended to read a file, extract its data and metadata,
        and pass them into a :class:`Dataset` object.

        Its content is designed to work with a simulated test file available
        on the documentation website but can be freely replaced as needed.

        :param filename: absolute path to the file
        :param metadata: if :py:obj:`True`, read only metadata and size of the
                         dataset. Not used here, but required to mach format
                         of other **Dataloaders**. See :meth:`load_ses_zip`
                         for more info.
        """

        # Extract the dataset and some metadata
        datfile = h5py.File(filename, 'r')
        h5_data = datfile['spectrum']
        yscale = datfile['dimensions'].attrs['momentum']
        zscale = datfile['dimensions'].attrs['energy']

        # Convert to array and make 3D if necessary
        shape = h5_data.shape
        if metadata:
            data = np.zeros(shape)
        else:
            data = np.array(h5_data)

        data = data.T
        if len(shape) == 2:
            ny = shape[1]
            nz = shape[0]
            # Make data 3D
            data = data.reshape((1, ny, nz))
            xscale = np.array([1])
            scan_type = 'cut'
        else:
            # adjust if scan has higher dimensionality
            pass

        # Extract available metadata
        meta = datfile['metadata'].attrs
        x_pos = meta['x']
        y_pos = meta['y']
        z_pos = meta['z']
        temp = meta['temp']
        pressure = meta['pres']
        hv = meta['hv']
        polarization = meta['polar']
        DT = meta['DT']

        self.ds.data = data
        self.ds.xscale = xscale
        self.ds.yscale = yscale
        self.ds.zscale = zscale
        self.ds.x = x_pos
        self.ds.y = y_pos
        self.ds.z = z_pos
        self.ds.temp = temp
        self.ds.pressure = pressure
        self.ds.hv = hv
        self.ds.polarization = polarization
        self.ds.scan_type = scan_type
        self.ds.DT = DT

        h5py.File.close(datfile)
