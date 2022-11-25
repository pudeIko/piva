.. _sec-file-formats:

File formats
============

This section discusses what it means for a datafile to be *recognizable* and 
thus *readable* by :mod:`piva`.

With ARPES data, the tendency seems to be that each beamline and each lab uses 
their own file format and conventions.
Therefore, in order to be able to handle these different files there really 
is no other way than to have a dedicated script or program which converts 
these into a common form.

Within :mod:`piva`, this is taken care of by the :ref:`data_loader.py 
file<sec-data-loader-file>`.
This file defines a data structure that is used consistently within :mod:`piva`.
Furthermore, it provides the :class:`~Dataloader` class as a base structure 
for data loaders from different sources (beamlines, electron analyzers, labs).
A dedicated dataloader exists for some ARPES beamlines.
If your beamline is not yet supported, you will not be able to read your data 
into :mod:`piva` unless a new data loader is written and added to that file.

You are highly encouraged to share your self-written data loaders with the 
world by adding it to :mod:`piva`'s source code.
This is ideally done `directly through github 
<https://github.com/pudeIko/piva/pulls>`_ or alternatively by contacting the 
development team:

.. include:: contact.rst

