# -*- coding: utf-8 -*-
"""Harmonize QT bindings

* Automatically import either PyQt5 or PySide depending on availability
* Allow to import QtCore/QtGui pyqtgraph.Qt without specifying which Qt wrapper
  you want to use.
* Declare QtCore.Signal, .Slot in PyQt4

"""

from pyqtgraph.Qt import QT_LIB, PYQT4, PYQT5
from pyqtgraph.Qt import QtWidgets, QtGui, QtCore

if QT_LIB in [PYQT4, PYQT5]:

    # Add pyqtSlot as Slot for consistency
    if not hasattr(QtCore, 'Slot'):
        QtCore.Slot = QtCore.pyqtSlot


