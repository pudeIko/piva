.. _sec-installation:

Installation
============

The installation of :mod:`piva` has been tested on macOS, Windows, and Linux.

The easiest way to install the package is to use
`pip <https://pip.pypa.io/en/stable/>`_, by simply typing the following in the
command line::

   pip install piva

Alternatively, you can also install it directly from the git repository::

    pip install git+https://github.com/pudeIko/piva.git

.. note::
    Cloning the git repository using ``pip`` downloads only the source files,
    skipping *i.a.* example files used for :ref:`testing <sec-testing>`. To
    clone the entire repository, go to the directory where you want to store
    the package and run::

        git clone https://github.com/pudeIko/piva.git
        cd piva
        pip install -e ./

    This will download and install :mod:`piva` locally.


.. note::
    Setting up package through github might require installing ``git`` first::

        conda install git

Virtual environment
-------------------

In order to avoid conflicts with some system dependencies, good practice
suggests to create a virtual environment. As in the example below, it is
recommended to use ``python`` version 3.10.8.

Detailed instructions for Anaconda users follow:

1) Open "Anaconda Prompt" 

2) Create an environment with a custom name (here ``piva-env``), specified
   ``python`` version and activate it::

      % conda create --name piva-env python==3.10.8
      [some output]
      % conda activate piva-env
      (piva-env) %

3) Enter your virtual environment and make sure latest version of ``pip``
   is installed::

      (piva-env) % pip install --upgrade pip

4) Run the following commands to download and install piva with all its
   dependencies::

      (piva-env) % pip install piva


4) Now you can run :mod:`piva` by simply typing ::

      (piva-env) % db
   
Which should bring up a :ref:`DataBrowser <sec-db>` window.


.. _sec-testing:

Testing
-------

Once installed, correct configuration of the package can be verified by
following methods:

1) From the **Menu bar** of the opened :ref:`DataBrowser <sec-db>`, navigate
   to **Open** -> **Example**. This will bring up an example dataset that can
   be used to test the functionalities of the :mod:`piva` package and get a
   feel for the GUI.

2) Using implemented automated tests.

    - To check proper behavior of implemented Dataloaders run::

        (piva-env) % python -m piva.tests.dataloaders_test

      Which will print to the terminal list of correctly loaded files.


    - DataViewers can be tested with::

        (piva-env) % python -m piva.tests.viewers_test

      This will start new :mod:`piva` session, execute sequence of actions
      emulating a physical user and test basic functionalities of the GUI.

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
   are working with, or directly at the `db` command: `XDG_SESSION_TYPE=xcb; db`.

Dependencies
------------

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

