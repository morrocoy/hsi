# -*- coding: utf-8 -*-
"""
Demonstrate ability of ImageItem to be used as a canvas for painting with
the mouse.

"""

from pyqtgraph.Qt import QtCore, QtGui
import numpy as np
import pyqtgraph as pg

app = QtGui.QApplication([])

## Create window with GraphicsView widget
w = pg.GraphicsView()
w.show()
w.resize(800,800)
w.setWindowTitle('pyqtgraph example: Draw')

# view = pg.ViewBox()
# w.setCentralItem(view)

## lock the aspect ratio
# view.setAspectLocked(True)


## Create image item
img = pg.ImageItem(np.zeros((640,480, 3)))
# view.addItem(img)


plotItem = pg.PlotItem()
plotItem.addItem(img)
w.setCentralItem(plotItem)
# w.addItem(plotItem)
## Set initial view bounds
# view.setRange(QtCore.QRectF(0, 0, 640, 480))

## start drawing with 3x3 brush
kern = np.array([
    [0.0, 0.5, 0.0],
    [0.5, 1.0, 0.5],
    [0.0, 0.5, 0.0]
])

# kern = np.array([
#     [0.0, 127, 0.0],
#     [127, 255, 127],
#     [0.0, 127, 0.0]
# ])

brush = (255, 0, 255)
kern = np.array([
    [(255, 0, 255)],
])

brush = 0.01*np.array([255, 0, 255])
brush = 1*np.array([1., 0, 1.])

kern_template = np.array([
    [0.0, 0.5, 0.0],
    [0.5, 1.0, 0.5],
    [0.0, 0.5, 0.0]
])

kern_template = np.array([
    [0.0, 0.5, 0.5, 0.0],
    [0.5, 1.0, 1.0, 0.5],
    [0.5, 1.0, 1.0, 0.5],
    [0.0, 0.5, 0.5, 0.0],
])


kern = np.outer(kern_template, brush).reshape(kern_template.shape + (3,))
# kern = np.array([
#     [[255, 0, 255]]
# ])

img.setDrawKernel(kern, mask=kern, center=(1,1), mode='add')
img.setLevels([[0, 1], [0, 1], [0, 1]])

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
