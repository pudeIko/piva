import numpy as np
from piva.data_loaders import Dataloader, Dataset
from piva.data_browser import DataBrowser


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
        dl_label = 'Custom Dataloader - {}'.format(loader.name)

        self.db.dp_dl_picker.addItem(dl_label)
        self.db.add_dataloader_to_record(dl_label, loader)

        print('\t', loader.name, dl_label)


class CustomDataloader(Dataloader):
    """
    Simple example of a custom Dataloader.
    """

    name = 'Custom1'

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

        # dummy if condition to make the example work. One should remove
        # 'isinstance' part and change extension in 'endswith'
        if filename.endswith('h5') or isinstance(filename, str):
            self.custom_loading_method(filename, metadata=metadata)
        else:
            raise NotImplementedError

        self.ds.add_org_file_entry(filename, self.name)
        return self.ds

    # function to implement. It supposed to read the file, extract data and
    # metadata from it and pass them into Dataset object.
    def custom_loading_method(self, filename, metadata=False):
        n = 100
        self.ds.data = np.random.random((n, n, n))
        self.ds.xscale = np.arange(n)
        self.ds.yscale = np.arange(n)
        self.ds.zscale = np.arange(n)
