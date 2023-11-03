.. _sec-installation:

Installation
============

The installation of :mod:`piva` has been tested on macOS, Windows and Linux.

The easiest way to install the package is to use 
`pip <https://pip.pypa.io/en/stable/>`_. Just type the following on a command 
line::

   pip install piva

Alternatively, one can also install it directly from the git repo::

    pip install git+https://github.com/pudeIko/piva.git


.. note::
    Cloning git repo using ``pip`` downloads only source files, skipping *i.a.*
    example files used for :ref:`testing <sec-testing>`. To clone entire
    repository go to the directory in which you want to store the package and
    run::

        git clone https://github.com/pudeIko/piva.git
        cd piva
        pip install -e ./

    To download and install :mod:`piva` locally.

.. note::
    Setting up package through github might require installing ``git`` first::

        conda install git

Virtual environment
-------------------

In order to not mess up some system dependencies, good practice suggests
creating a virtual environment. As in example below, it is recommended to use
``python`` version 3.10.8.

Detailed instructions for Anaconda users follow:

1) Open "Anaconda Prompt" 

2) Create an environment with a custom name (here ``piva-env``), specified
   ``python`` version and activate it::

      % conda create --name piva-env python==3.10.8
      [some output]
      % conda activate piva-env
      (piva-env) %

3) Inside your virtual environment, run the following commands to download and 
   install piva with all its dependencies::

      (piva-env) % pip install piva


4) Now you can run :mod:`piva` by simply typing ::

      (piva-env) % db
   
Which should bring up a :ref:`DataBrowser <sec-db>` window.


.. _sec-testing:

Testing
-------

Once installed, correct configuration of the package can be verified by
following methods:

1) From the **Menu bar** of opened :ref:`DataBrowser <sec-db>` navigate to
   **Open** -> **Example**. This will bring up an example dataset, which can
   be used for test functionalities of the :mod:`piva` package and get a
   feeling of the GUI.

2) Using implemented automated tests.

    - To check proper behavior of implemented Dataloaders run::

        (piva-env) % python -m piva.tests.dataloaders_test

      Which will print to the command line list of correctly loaded files.


    - DataViewers can be tested with::

        (piva-env) % python -m piva.tests.viewers_test

      This will start new :mod:`piva` session, execute sequence of actions
      emulating physical user and test basic functionalities of the GUI.

Successful execution of the tests should give a message like::

    -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
    =============== 2 passed, 1 warning in 81.50s (0:01:21) ===============


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
    - `jupytelab <https://jupyter.org>`_ for running deeper analysis and
      implementation of the experimental logbooks
    - `matplotlib <https://matplotlib.org/>`_ for plot exporting
      functionalities.

