Installation
============

The installation of ``piva`` has been tested on Linux.

he easiest way to install the package is to use 
`pip <https://pip.pypa.io/en/stable/>`_. Just type the following on a command 
line::

   pip install piva

If the above does not appear to work, it might help to try installation from 
a virtual environment. 

Anaconda
--------

Detailed instructions for Anaconda users follow:

1) Open "Anaconda Prompt" 

2) In order not to mess up your dependencies, create a virtual 
   environment. In the example we will be using python version 3.7.5, but 
   most other versions should work as well. ::

      $ conda create --name testenv python==3.7.5
      [some output]
      $ conda activate testenv
      (testenv) $

3) Inside your virtual environment, run the following commands to download and 
   install piva with all its dependencies (the first one is just to 
   upgrade pip to the latest version)::
   
      (testenv) $ pip install --upgrade pip
      (testenv) $ pip install data_slicer
   
   This will create a lot of console output. If everything succeeded, you should 
   see something like ``Successfully installed piva`` towards the end.

4) Test the installation by starting up a databrowser::

      (testenv) $ db
   
This should bring up a :ref:`databrowser window <sec-db>`.

..
    .. _sec-verifying:

    Verifying your installation
    ---------------------------
    
    Once installed, you can run a set of automated tests in order to check if the 
    main features work as expected.
    To do this, issue the following on the command line::
    
       python -m data_slicer.tests.run_tests
    
    The result should be that some text is printed to the console and some 
    windows open, with a few things happening in them before they quickly close 
    again.
    Basically, these tests simulate a few interactions that the user could have 
    with these windows and verify that they worked with some checks.
    If all went well you might see some warnings, but no notifications of any
    ``failures``.
    It could, for example, look like this::
       
       ================== 4 passed, 16 warnings in 14.92 s ==================
    
    .. note::
       The fact that all tests passed does not guarantee that *everything* is in 
       working order - but it's a very good sign.
    
    If interested, you can also run these tests individually and interact with 
    the respective windows by calling them like so::
    
       python -m data_slicer.tests.test_XXX
    
    where ``XXX`` is any of ``pit``, ``freeslice``, ``threedwidget``.

Upgrading
---------

The following command will attempt to upgrade ``piva`` to the latest 
published version::

   pip install --upgrade piva

It is usually a good idea to upgrade ``pip`` itself before running above 
command::

   pip install --upgrade pip

.. Note::
   Run these commands from within the same (virtual) environment as you've 
   installed ``piva`` in.

Dependencies
------------

This software is built upon on a number of other open-source frameworks.
The complete list of packages is:

.. include:: ../../requirements.txt
   :code:

Most notably, this includes 
`pyqtgraph <https://pyqtgraph.readthedocs.io/en/latest/>`_ for fast live 
visualizations and widgets, `numpy <https://numpy.org/>`_ for numerical 
operations and `matplotlib <https://matplotlib.org/>`_ for plot exporting 
functionalities.

