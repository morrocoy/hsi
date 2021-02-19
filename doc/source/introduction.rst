Introduction
============

What is hsi?
------------

HSI is user interface library for Python that provides functionality commonly
used for the analysis of hyperspectral data, in particularly, images data.
A vast of engineering and science applications profit from
hyperspectral imaging, such as in agriculture, medicin, or food
processing `[wiki] <https://en.wikipedia.org/wiki/Hyperspectral_imaging>`_.

The modul library is particularly designed for clinical practice. Besides
general functionalities to deal with hyperspectral images, it provides
various algorithmes and methods to characterize tissue oxygenation and
other parameters that are of importance in supervising healing processes.

HSI makes use of the Qt GUI platform and PyQtGraph for scientific data
visualization. A number of widgets is provided for interactive analyses with
minimal effort.

It is known to run on Linux and Windows.


Features
--------

Amongst the core features of hsi are:

* Basic handling of hyperspectral image data including file io, filtering, and
  formatting
* Analysis tools to characterize tissue oxygenation and other parameters
* Widgets for marking/selecting plot regions
* Widgets to visualize 2D histograms and automatically slice multi-dimensional
  image data
* TODO

..
    * Basic data visualization primitives: Images, line and scatter plots
    * Fast enough for realtime update of video/plot data
    * Interactive scaling/panning, averaging, FFTs, SVG/PNG export
    * Widgets for marking/selecting plot regions
    * Widgets for marking/selecting image region-of-interest and automatically
      slicing multi-dimensional image data
    * Framework for building customized image region-of-interest widgets
    * Docking system that replaces/complements Qt's dock system to allow more
      complex (and more predictable) docking arrangements


.. _examples:

Examples
--------

HSI includes an extensive set of examples that can be accessed by
running either ``python -m hsi.examples`` [#editableInstall]_ or ::

    import hsi.examples
    hsi.examples.run()

TODO

..
    HSI includes an extensive set of examples that can be accessed by
    running either ``python -m pyqtgraph.examples`` [#editableInstall]_ or ::

        import hsi.examples
        hsi.examples.run()

    Or, if the project repository is local, you can run``python examples/`` from
    the source root.

    This will start a launcher with a list of available examples. Select an item
    from the list to view its source code and double-click an item to run the
    example.

    Note If you have installed pyqtgraph with ``python setup.py develop``
    then the examples are incorrectly exposed as a top-level module. In this case,
    use ``import examples; examples.run()``.



.. rubric:: Footnotes

.. [#editableInstall] This method does not work when pyqtgraph is installed in editable mode.
