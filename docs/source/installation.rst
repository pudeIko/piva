.. _sec-installation:

Installation
============

The installation of :mod:`piva` has been tested on macOS, Windows, and Linux.
You can install it either from source or using a package
manager. The following guidelines are designed to help you avoid most common
installation issues.

To begin, regardless of the installation method you choose, download and
install `Conda <https://www.anaconda.com/download>`_ to set up a virtual
environment for the installation.


Installation from Sources
-------------------------

To ensure all dependencies are properly handled, it is recommended to install
the package from source by cloning the GitHub repository::

    git clone https://github.com/pudeIko/piva.git

.. note::
    Installing the package from GitHub may require git. If it's not already
    installed, you can add it using::

        conda install git


Next, navigate to the downloaded directory and run the following command::

    cd piva
    conda env create -f environment.yml

This will create a virtual environment named `piva-env` and install
:mod:`piva` in editable mode, allowing for easier modifications and
enhancements to the code.

To start the software and launch the :ref:`DataBrowser <sec-db>`, simply run::

    db


Installation via PyPi
---------------------
Alternatively, :mod:`piva` can be installed using the PyPI package manager.

This approach requires creating a virtual environment manually first. As in
the example below, it is recommended to use Python version 3.10.8::

      conda create --name piva-env python==3.10.8
      [some output]
      conda activate piva-env

Inside the activated virtual environment, upgrade ``pip`` and install
:mod:`piva`::

    pip install --upgrade pip
    pip install piva

To start the software and open the :ref:`DataBrowser <sec-db>` window, simply
run::

    db



.. _sec-testing:

Testing
=======

Once installed, correct configuration of the package can be verified by
following methods:

1) From the **Menu bar** of the opened :ref:`DataBrowser <sec-db>`, navigate
   to **Open** -> **Example**. This will bring up an example dataset that can
   be used to test the functionalities of the :mod:`piva` package and get a
   feel for the GUI.

2) Using implemented automated tests.

    - To check proper behavior of implemented Dataloaders run::

        python -m piva.tests.dataloaders_test

      Which will print to the terminal list of correctly loaded files.


    - DataViewers can be tested with::

        python -m piva.tests.viewers_test

      This will start new :mod:`piva` session, execute sequence of actions
      emulating a physical user and test basic functionalities of the GUI.

    - Functinalities using JupyterLab can be checked with a semi-automated
      test by running::

        python -m piva.tests.jupyter_test

      This will create example Jupyter notebooks, start a JupyterLab server,
      stop the server, and remove the created files.

      .. note::
         When running on Windows, users might need to stop the server (started
         on port 56789) manually. To do so, after executing the test, run::

           jupyter-lab stop 56789


Successful execution of the tests should give a message like::

    -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
    =============== 2 passed, 1 warning in 81.50s (0:01:21) ===============

.. note::
   Running on Linux with wayland.
   If you are faced with an error of the form 
   `Warning: Ignoring XDG_SESSION_TYPE=wayland on Gnome. Use QT_QPA_PLATFORM=wayland to run on Wayland anyway.`
   you can work around this by setting `XDG_SESSION_TYPE=xcb` (as opposed 
   to `wayland` as the error message would suggest).
   Do this either by running `export XDG_SESSION_TYPE=xcb` in the shell you
   are working with, or directly at the `db` command:
   `XDG_SESSION_TYPE=xcb; db`.



Dependencies
============

This software is built upon on a number of other open-source frameworks.
The complete list of packages is:

.. include:: ../../requirements.txt
   :code:

Most notably, this includes:

    - `pyqtgraph <https://pyqtgraph.readthedocs.io/en/latest/>`_ for fast live
      visualizations and widgets,
    - `numpy <https://numpy.org/>`_ for numerical operations,
    - `jupyterlab <https://jupyter.org>`_ for running deeper analysis and
      implementation of the experimental logbooks
    - `matplotlib <https://matplotlib.org/>`_ for plot exporting
      functionalities.

