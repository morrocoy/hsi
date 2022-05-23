# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:35:08 2021

@author: kpapke
"""
import numpy as np

from ...bindings.Qt import QtWidgets, QtCore
from ...log import logmanager

logger = logmanager.getLogger(__name__)

__all__ = ['QParamRegionWidget']


class QParamRegionWidget(QtWidgets.QWidget):
    """ Config widget with two spinboxes that control the parameter bounds."""

    sigValueChanged = QtCore.Signal(str, list)

    def __init__(self, *args, **kwargs):
        """ Constructor
        """
        parent = kwargs.get('parent', None)
        super(QParamRegionWidget, self).__init__(parent=parent)

        if len(args) == 1:
            kwargs['parent'] = args[0]
        elif len(args) == 2:
            kwargs['name'] = args[0]
            kwargs['parent'] = args[1]
        elif len(args) == 3:
            kwargs['name'] = args[0]
            kwargs['value'] = args[1]
            kwargs['parent'] = args[2]
        elif len(args) == 4:
            kwargs['name'] = args[0]
            kwargs['value'] = args[1]
            kwargs['scale'] = args[2]
            kwargs['parent'] = args[3]

        self.name = kwargs.get('name', None)  # parameter name
        self.label = kwargs.get('label', self.name)  # parameter label

        self.dvalue = [None, None]  # default value
        self.scale = kwargs.get('scale', 1.)  # scale for presentation

        # set default value
        val = kwargs.get('value', [None, None])
        self.setValueDefault(val)

        self.varLabel = QtWidgets.QLabel()
        self.lowerBoundSpinBox = QtWidgets.QDoubleSpinBox(self)
        self.upperBoundSpinBox = QtWidgets.QDoubleSpinBox(self)

        self.lowerBoundSpinBox.setKeyboardTracking(False)
        self.upperBoundSpinBox.setKeyboardTracking(False)

        # configure widget views
        self._setupViews(*args, **kwargs)

        # connect signals
        self.lowerBoundSpinBox.valueChanged.connect(
            lambda val: self._triggerSigValueChanged((val, None)))
        self.upperBoundSpinBox.valueChanged.connect(
            lambda val: self._triggerSigValueChanged((None, val)))

    def _setupViews(self, *args, **kwargs):
        self.mainLayout = QtWidgets.QFormLayout()
        self.mainLayout.setContentsMargins(0, 0, 0, 0) # left, top, right, bottom
        self.mainLayout.setSpacing(3)
        self.setLayout(self.mainLayout)

        self.varLabel.setText(self.label)
        self.varLabel.setIndent(5)
        self.varLabel.setMinimumWidth(50)
        self.varLabel.setStyleSheet("border: 0px;")

        # maxWidth = kwargs.get('maximumWidth', 67)
        singleStep = kwargs.get('singleStep', 0.1)
        decimals = kwargs.get('decimals', 3)

        # self.setMaximumWidth(maxWidth)
        self.setSingleStep(singleStep)
        self.setDecimals(decimals)
        self.setBounds([-1e5, 1e5])
        self.setEnabled(True)

        # set value
        if self.dvalue[0] is None:
            self.lowerBoundSpinBox.setValue(self.lowerBoundSpinBox.minimum())
        else:
            self.lowerBoundSpinBox.setValue(self.dvalue[0] * self.scale)

        if self.dvalue[1] is None:
            self.upperBoundSpinBox.setValue(self.upperBoundSpinBox.maximum())
        else:
            self.upperBoundSpinBox.setValue(self.dvalue[1] * self.scale)

        layout = QtWidgets.QHBoxLayout()

        layout.addStretch()
        layout.addWidget(self.lowerBoundSpinBox)
        layout.addWidget(self.upperBoundSpinBox)
        self.mainLayout.addRow(self.varLabel, layout)

    def _triggerSigValueChanged(self, bounds=[None, None]):
        lbnd, ubnd = bounds

        if lbnd is None:
            lbnd = self.lowerBoundSpinBox.value()
        if ubnd is None:
            ubnd = self.upperBoundSpinBox.value()

        lbnd = lbnd / self.scale
        ubnd = ubnd / self.scale
        self.sigValueChanged.emit(self.name, [lbnd, ubnd])

    def reset(self):
        if self.dvalue[0] is None:
            self.lowerBoundSpinBox.setValue(self.lowerBoundSpinBox.minimum())
        else:
            self.lowerBoundSpinBox.setValue(self.dvalue[0] * self.scale)

        if self.dvalue[1] is None:
            self.upperBoundSpinBox.setValue(self.upperBoundSpinBox.maximum())
        else:
            self.upperBoundSpinBox.setValue(self.dvalue[1] * self.scale)

    def setDecimals(self, val):
        self.lowerBoundSpinBox.setDecimals(val)
        self.upperBoundSpinBox.setDecimals(val)
        pass

    def setEnabled(self, val):
        self.lowerBoundSpinBox.setEnabled(val)
        self.upperBoundSpinBox.setEnabled(val)

    def setLabel(self, label):
        self.label = label
        self.varLabel.setText(label)

    def setMaximumWidth(self, val):
        super(QParamRegionWidget, self).setMaximumWidth(val)
        width = int((val - 50) // 2 - 8)
        self.lowerBoundSpinBox.setMaximumWidth(width)
        self.upperBoundSpinBox.setMaximumWidth(width)


    def setName(self, name, label=None):
        self.name = name
        if label is not None:
            self.label = label
            self.varLabel.setText(label)

    def setBounds(self, val=[None, None]):
        if val is None:
            bounds = [None, None]
        elif type(val) in [list, tuple, np.ndarray] and len(val) == 2:
            bounds = [val[0], val[1]]
        else:
            raise ValueError("Argument `val` must be list, tuple or "
                             "1D ndarray of length 2. Got {}".format(range))

        lbnd, ubnd = bounds
        if lbnd is None:
            lbnd = -1e5
        else:
            lbnd = self.scale * lbnd
        if ubnd is None:
            ubnd = 1e5
        else:
            ubnd = self.scale * ubnd
        self.lowerBoundSpinBox.setRange(lbnd, ubnd)
        self.upperBoundSpinBox.setRange(lbnd, ubnd)

    def setScale(self, val):
        lbnd, ubnd = self.value()
        lbnd = lbnd / self.scale * val
        ubnd = ubnd / self.scale * val

        self.scale = val
        self.lowerBoundSpinBox.setValue(lbnd)
        self.upperBoundSpinBox.setValue(ubnd)

    def setSingleStep(self, val):
        self.lowerBoundSpinBox.setSingleStep(val)
        self.upperBoundSpinBox.setSingleStep(val)

    def setValue(self, val):
        if val is None:
            bounds = [None, None]
        elif type(val) in [list, tuple, np.ndarray] and len(val) == 2:
            bounds = [val[0], val[1]]
        else:
            raise ValueError("Argument val must be list, tuple or "
                             "1D ndarray of length 2. Got {}".format(val))

        lbnd, ubnd = bounds
        if lbnd is None:
            lbnd = self.lowerBoundSpinBox.minimum()
        else:
            lbnd = self.scale * lbnd
        if ubnd is None:
            ubnd = self.upperBoundSpinBox.maximum()
        else:
            ubnd = self.scale * ubnd

        self.lowerBoundSpinBox.blockSignals(True)
        self.upperBoundSpinBox.blockSignals(True)
        self.lowerBoundSpinBox.setValue(lbnd)
        self.upperBoundSpinBox.setValue(ubnd)
        self.lowerBoundSpinBox.blockSignals(False)
        self.upperBoundSpinBox.blockSignals(False)

        self._triggerSigValueChanged()

    def setValueDefault(self, val):
        if val is None:
            bounds = [None, None]
        elif type(val) in [list, tuple, np.ndarray] and len(val) == 2:
            bounds = [val[0], val[1]]
        else:
            raise ValueError("Argument val must be list, tuple or "
                             "1D ndarray of length 2. Got {}".format(val))

        self.dvalue = bounds

    def value(self):
        lbnd = 1./self.scale * self.lowerBoundSpinBox.value()
        ubnd = 1./self.scale * self.upperBoundSpinBox.value()
        return [lbnd, ubnd]
