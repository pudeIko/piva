.. _sec-custom_widgets:


Custom Widgets
==============

The architecture of :mod:`piva` highly supports modularity by allowing users to
implement their own :class:`QtWidgets`, capable of performing various tasks and
expanding package's functionalities.

Configuration of the custom written widgets is accomplished through
:class:`WidgetImporter` object that must fulfill two requirements:

- :class:`WidgetImporter` class must be defined in
  ``piva_widget_importer.py`` module (python file)
- Path to the ``piva_widget_importer.py`` needs to be added to the
  ``$PYTHONPATH`` of your virtual environment. A simple guide on how to do
  it on any operating system can be found `here <https://stackoverflow.com/
  questions/10738919/how-do-i-add-a-path-to-pythonpath-in-virtualenv>`_.

At the beginning of each session :mod:`piva` will search for the
:mod:`piva_widget_importer` module, attempt to import :class:`WidgetImporter`
class and execute whatever code it contains. The most basic examples of the
implemented :class:`WidgetImporter` and :class:`CustomWidget` can be downloaded
from the links below:

- :download:`WidgetImporter <../misc/piva_widget_importer.py>`
- :download:`CustomWidget <../misc/custom_widget1.py>`

The above example creates a simple window (shown below) that can be opened
from the menu bar.

.. figure:: ../img/custom_widget.png
    :alt: Image not found.

Everyone is highly encouraged to share tested, self-written custom widgets with
the world by adding it to :mod:`piva`'s source code. This is ideally done
`directly through github  <https://github.com/pudeIko/piva>`_ or
alternatively by contacting the development team:

.. include:: contact.rst

