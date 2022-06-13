from collections.abc import Callable
import warnings

import numpy
import pyqtgraph as pg

from ...bindings.Qt import QtWidgets, QtCore
from ...log import logmanager

logger = logmanager.getLogger(__name__)


__all__ = ['ImageItem']


class ImageItem(pg.ImageItem):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`

    pg.ImageItem with additional features to select a region of interest.

    =============================== ===================================================
    **Signals:**
    sigROISelectionFinished(self)
    =============================== ===================================================
    """
    sigROISelectionFinished = QtCore.Signal(object, numpy.ndarray)

    def __init__(self, *args, **kwargs):
        super(ImageItem, self).__init__(*args, **kwargs)
        self.track_mouse_pos = []  # for roi selection
        self.drawEnabled = False

    def drawAt(self, pos, ev=None):
        if self.axisOrder == "col-major":
            pos = [int(pos.x()), int(pos.y())]
        else:
            pos = [int(pos.y()), int(pos.x())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0,dk.shape[0]]
        sy = [0,dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0]+ dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1]+ dk.shape[1]]

        for i in [0,1]:
            dx1 = -min(0, tx[i])
            dx2 = min(0, self.image.shape[0]-tx[i])
            tx[i] += dx1+dx2
            sx[i] += dx1+dx2

            dy1 = -min(0, ty[i])
            dy2 = min(0, self.image.shape[1]-ty[i])
            ty[i] += dy1+dy2
            sy[i] += dy1+dy2

        ts = (slice(tx[0],tx[1]), slice(ty[0],ty[1]))
        ss = (slice(sx[0],sx[1]), slice(sy[0],sy[1]))
        mask = self.drawMask
        src = dk

        if isinstance(self.drawMode, Callable):
            self.drawMode(dk, self.image, mask, ss, ts, ev)
        else:
            src = src[ss]
            if self.drawMode == 'set':
                if mask is not None:
                    mask = mask[ss]
                    self.image[ts] = self.image[ts] * (1-mask) + src * mask
                else:
                    self.image[ts] = src
            elif self.drawMode == 'add':
                self.image[ts] += src
            else:
                raise Exception("Unknown draw mode '%s'" % self.drawMode)
            self.updateImage()

    def mouseDragEvent(self, ev):
        # define region of interest by drag event
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        elif self.drawEnabled and self.drawKernel is not None:
            ev.accept()

            if ev.isStart():
                logger.debug("Start drawing ROI Mask.")
                self.track_mouse_pos = []

            pos = ev.pos()
            self.drawAt(pos, ev)

            if self.axisOrder == "col-major":
                pos = [int(pos.x()), int(pos.y())]
            else:
                pos = [int(pos.x()), int(pos.y())]
            self.track_mouse_pos.append(pos)

            if ev.isFinish():
                logger.debug("Finish drawing ROI Mask.")
                # print(self.track_mouse_pos)
                pnts = numpy.array(self.track_mouse_pos, dtype=int)
                self.image.fill(0)
                self.sigROISelectionFinished.emit(self, pnts)

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            if self.raiseContextMenu(ev):
                ev.accept()
        if self.drawEnabled and self.drawKernel is not None and \
                ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.drawAt(ev.pos(), ev)

    def hoverEvent(self, ev):
        if not ev.isExit() and self.drawEnabled and \
                self.drawKernel is not None and \
                ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
            ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton) ## we don't use the click, but we also don't want anyone else to use it.
            ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)
        elif not ev.isExit() and self.removable:
            ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)  ## accept context menu clicks

    def setDrawEnabled(self, b):
        self.drawEnabled = b