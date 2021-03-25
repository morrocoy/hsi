""" Color legend.

    Consists of a vertical color bar with a draggable axis that displays the values of the colors.
    Optionally a histogram may be displayed.
"""
import os
import numpy as np
import pyqtgraph as pg

from ...bindings.Qt import QtWidgets, QtCore
from ...log import logmanager
from ...misc import versionStrToTuple
from ...misc import check_is_an_array, check_class

import cProfile
import pstats

X_AXIS = pg.ViewBox.XAxis
Y_AXIS = pg.ViewBox.YAxis
BOTH_AXES = pg.ViewBox.XYAxes
CODE_PROFILING = False

logger = logmanager.getLogger(__name__)


def assertIsLut(lut):
    """ Checks that lut is Nx3 or Nx4 array of dtype uint8

        If dtype is not an 8 bits integer the resulting image will also not be 8 bits, but e.g.
        32-bits. This not only costs more memory but is also significantly slower.
    """
    check_is_an_array(lut)
    assert lut.ndim == 2, "Expected 2 dimensional LUT. Got {}D".format(lut.ndim)
    assert lut.shape[1] == 3 or lut.shape[1] == 4, \
        "Second dimension of LUT should be length 3 or 4. Got: {}".format(lut.shape[1])
    assert lut.dtype == np.uint8, \
        "LUT should have dtype np.uint8, got {}".format(lut.dtype)


def extentLut(lut):
    """ Duplicates the last item of the Lookup Table

        This is necessary because the pyqtgraph.makeARGB function has a wrong default scale. It
        should be equal to the length of the LUT, but it's set to len(lut)-1. We therefore add a
        fake LUT entry. See issue 792 of PyQtGraph.
        It has been fixed in PyQtGraph 1.11.0
    """
    assertIsLut(lut)
    extendedLut = np.append(lut, [lut[-1, :]], axis=0)
    return extendedLut


def isExtended(lut):
    """ Returns True if the lookup table has been extended with isExtended.
        I.e. returns True if the last and second to last LUT entries are the same
    """
    assertIsLut(lut)
    return np.array_equal(lut[-1, :], lut[-2, :])


class ColorBarItem(pg.GraphicsWidget):
    """ Color legend for an image plot.

        Draws a color legend. It consists of a color bar and axis that show the range of the colors.
        The color legend visualizes a lookup table (LUT) that maps a floating point value to an
        RBG color value. With the lookup table colors can be assigned to an image.

        By default a (rotated) histogram is drawn that shows the distribution of the values of all
        pixels in the array.

        The histogram is linked to a PyQtGraph ImageItem. An ImageItem has a LUT member.
        Unfortunately there is a bug in PyQtGraph when colorizing an image (issue 792). As a
        work-around we extent this LUT by duplicating the last item. You must therefore set the
        lut with ColorLegendItem.setLut, which adds this duplicate item (if not yet done). This
        class therefore has a side effect.

    """
    sigLevelsChanged = QtCore.Signal(tuple)

    def __init__(self,
                 imageItem=None,
                 label=None,
                 showHistogram=True,
                 subsampleStep='auto',
                 histHeightPercentile = 99.0,
                 histHeight = 50,
                 width = 20,
                 orientation='right',
                 maxTickLength=10):
        """ Constructor.

        Parameters
        ----------

        imageItem : PyQtGraph.ImageItem
            ImageItem to which this colorbar is be linked.
        label: str, optional
            The text to show next to the axis.
        showHistogram : bool, optional
            If True (the default), a histogram of image values is drawn.
        histHeightPercentile : float
            Only use this percentil when scaling the histogram height. Often an
            image has one color that occurs very often and this then will
            completely dominate the histogram. By discarding this color when
            scaling the histogram the other color occurrences will become
            visible. By default we use the 99 percentile, meaning the 1% of the
            bins with the highest values will be discarded.
        orientation : ('right' or 'bottom'), optional
            The orientation of the colorbar.
        subsampleStep : int
            The step size that is used to subsample the array when calculating
            the histogram. Can be a scalar, tuple with two elements, or 'auto'.
        maxTickLength : int
            Maximum tick length of the color axis. Default = 10

        """
        pg.GraphicsWidget.__init__(self)

        check_class(imageItem, pg.ImageItem, allowNone=True)

        if CODE_PROFILING:
            # Profiler that measures the drawing of the inspectors.
            self._profFileName = "col_legend.prof"

            logger.warning("Profiling is on for {}. This may cost a bit of CPU time.")
            self._profiler = cProfile.Profile()

        self.histogramHeight = histHeight
        self.histHeightPercentile = histHeightPercentile
        self.orientation = orientation
        self._histogramIsVisible = showHistogram
        self._imageItem = None


        # List of mouse buttons that reset the color range when clicked.
        # You can safely modify this list.
        self.resetRangeMouseButtons = [QtCore.Qt.MiddleButton]

        self._subsampleStep = subsampleStep

        # Histogram
        self.histViewBox = pg.ViewBox(enableMenu=False)
        self.histViewBox.disableAutoRange(BOTH_AXES)

        self.histPlotDataItem = pg.PlotDataItem()

        if self.orientation == 'right':
            self.histViewBox.setMouseEnabled(x=False, y=True)
            self.histViewBox.setFixedWidth(self.histogramHeight)
            self.histPlotDataItem.rotate(90)
        else:
            self.histViewBox.setMouseEnabled(x=True, y=False)
            self.histViewBox.setFixedHeight(self.histogramHeight)


        self.histViewBox.addItem(self.histPlotDataItem)
        self.fillHistogram()

        # Color scale. Viewbox ranges from 0 to number of colors
        self.colorScaleViewBox = pg.ViewBox(enableMenu=False, border=None)

        self.colorScaleViewBox.disableAutoRange(pg.ViewBox.XYAxes)
        self.colorScaleViewBox.setMouseEnabled(x=False, y=False)

        if self.orientation == 'right':
            self.colorScaleViewBox.setMinimumWidth(5)
            self.colorScaleViewBox.setMaximumWidth(25)
            self.colorScaleViewBox.setWidth(width)
        else:
            self.colorScaleViewBox.setMinimumHeight(5)
            self.colorScaleViewBox.setMaximumHeight(25)
            self.colorScaleViewBox.setFixedHeight(width)

        self.colorScaleImageItem = pg.ImageItem() # image data will be set in setLut
        self.colorScaleViewBox.addItem(self.colorScaleImageItem)
        self.colorScaleViewBox.setZValue(10)

        # Overlay viewbox that will have always have the same geometry as the colorScaleViewBox.
        # The histograms and axis item are linked to this viewbox
        self.overlayViewBox = pg.ViewBox(
            enableMenu=False, border=pg.mkPen(pg.getConfigOption('foreground'), width=1))
        self.overlayViewBox.setZValue(100)

        # Axis that shows the ticks and values
        if self.orientation == 'right':
            self.axisItem = pg.AxisItem(
                orientation='right', linkView=self.overlayViewBox,
                showValues=True, maxTickLength=maxTickLength, parent=self)
            self.histViewBox.linkView(pg.ViewBox.YAxis, self.overlayViewBox)
        else:
            self.axisItem = pg.AxisItem(
                orientation='bottom', linkView=self.overlayViewBox,
                showValues=True, maxTickLength=maxTickLength, parent=self)
            self.histViewBox.linkView(pg.ViewBox.XAxis, self.overlayViewBox)

        # Overall layout
        self.mainLayout = QtWidgets.QGraphicsGridLayout()
        self.setLayout(self.mainLayout)
        self.mainLayout.setContentsMargins(1, 1, 1, 1)
        self.mainLayout.setSpacing(0)

        if self.orientation == 'right':
            self.mainLayout.addItem(self.histViewBox, 0, 0)
            self.mainLayout.addItem(self.colorScaleViewBox, 0, 1)
            self.mainLayout.addItem(self.axisItem, 0, 2)
        else:
            self.mainLayout.addItem(self.histViewBox, 0, 0)
            self.mainLayout.addItem(self.colorScaleViewBox, 1, 0)
            self.mainLayout.addItem(self.axisItem, 2, 0)
        self.overlayViewBox.setParentItem(self.colorScaleViewBox.parentItem())

        # Connect signals
        self.colorScaleViewBox.geometryChanged.connect(self._updateOverlay)

        # It might also trigger an update when the axis is resized (but can't reproduce it
        # anymore). If needed we could test if the range has changed substantially before updating
        # image levels.

        if self.orientation == 'right':
            self.overlayViewBox.sigYRangeChanged.connect(self._updateImageLevels)
        else:
            self.overlayViewBox.sigXRangeChanged.connect(self._updateImageLevels)

        self.setLabel(label)
        self.showHistogram(showHistogram)
        self.setImageItem(imageItem)


    def finalize(self):
        if CODE_PROFILING:
            logger.info("Saving profiling information to {}"
                        .format(os.path.abspath(self._profFileName)))
            profStats = pstats.Stats(self._profiler)
            profStats.dump_stats(self._profFileName)

    def _updateOverlay(self):
        """ Makes the overlay the same size as the colorScaleViewBox
        """
        self.overlayViewBox.setGeometry(self.colorScaleViewBox.geometry())


    def getImageItem(self):
        """ Returns the PyQtGraph ImageItem that this legend is linked to.
        """
        return self._imageItem


    def setImageItem(self, imageItem):
        """ Links the legend to an image item.
        """
        logger.debug("setImageItem")
        check_class(imageItem, pg.ImageItem, allowNone=True)

        # Remove old imageItem
        if self._imageItem:
            self._imageItem.sigImageChanged.disconnect(self.onImageChanged)

        self._imageItem = imageItem
        self._imageItem.sigImageChanged.connect(self.onImageChanged)

        if self._imageItem.lut is not None:
            self.setLut(self._imageItem.lut) # extents lut if necessary
        self.onImageChanged()


    def onImageChanged(self):
        """ Called when new image data has been set in the linked ImageItem

            Updates the histogram and colorize the image (_updateImageLevels)
        """
        self._updateHistogram()
        self._updateImageLevels()


    @QtCore.pyqtSlot()
    def _updateImageLevels(self):
        """ Updates the image levels from the color levels of the
        """
        if CODE_PROFILING:
            self._profiler.enable()
        levels = self.getLevels()
        if self._imageItem is not None:
            self._imageItem.setLevels(levels)

        self.sigLevelsChanged.emit(levels)

        if CODE_PROFILING:
            self._profiler.enable()


    @property
    def subsampleStep(self):
        """ The step size that is used to subsample the array when calculating the histogram
        """
        return self._subsampleStep


    @subsampleStep.setter
    def subsampleStep(self, value):
        """ The step size that is used to subsample the array when calculating the histogram

            :value: can be a scalar, a tuple with two elements, or 'auto'.
        """
        self._subsampleStep = value


    def _updateHistogram(self):
        """ Updates the histogram with data from the image
        """
        if not self._histogramIsVisible or self._imageItem is None or self._imageItem.image is None:
            self.histPlotDataItem.clear()
            return

        try:
            imgArr = self._imageItem.image
            histRange = self._calcHistogramRange(imgArr, step=self.subsampleStep)

            histBins = self._calcHistogramBins(
                histRange, forIntegers=self._imageItemHasIntegerData(self._imageItem))

            histogram = self._imageItem.getHistogram(bins=histBins, range=histRange)

            assert histogram[0] is not None, "Histogram empty in imageChanged()" # when does this happen again?

        except Exception as ex:
            logger.warning("Unable to calculate histogram: {}".format(ex))
            self.histPlotDataItem.clear()
        else:
            self.histPlotDataItem.setData(*histogram)

            # Discard outliers when setting the histogram height so one dominant color in the
            # image doesn't make the other colors occurences unreadable.
            histValues = histogram[1]
            histYrange = np.percentile(histValues, (self.histHeightPercentile, ))[0]

            if self.orientation == 'right':
                self.histViewBox.setRange(xRange=(-histYrange, 0), padding=None)
            else:
                self.histViewBox.setRange(yRange=(histYrange, 0), padding=None)



    @classmethod
    def _calcHistogramRange(cls, imgArr, step='auto', targetImageSize=200):
        """ Calculates bins for the histogram.

            We explicitly calculate the range because in PyQtGraph 0.10.0 the histogram range is
            calculated without taking NaNs into account (this was fixed in PyQtGraph issue #699.
            Also it allows you to derive a subclasses of ColorLegendItem and override this.

            This function was based on PyQtGraph.ImageItem.getHistogram() commit efaf61f

            The *step* argument causes pixels to be skipped when computing the histogram to save
            time. If *step* is 'auto', then a step is chosen such that the analyzed data has
            dimensions roughly *targetImageSize* for each axis.

            * Integer images will have approximately 500 bins,
              with each bin having an integer width.
            * All other types will have 500 bins.

            :returns: (minRange, maxRange) tuple.
        """
        if imgArr is None or imgArr.size == 0:
            return None, None

        if step == 'auto':
            step = (max(1, int(np.ceil(imgArr.shape[0] / targetImageSize))),
                    max(1, int(np.ceil(imgArr.shape[1] / targetImageSize))))

        if np.isscalar(step):
            step = (step, step)

        stepData = imgArr[::step[0], ::step[1]]

        logger.debug("step: {}, stepData.shape: {}".format(step, stepData.shape))

        mn = np.nanmin(stepData)
        mx = np.nanmax(stepData)

        return (mn, mx)


    @classmethod
    def _imageItemHasIntegerData(cls, imageItem):
        """ Returns True if the imageItem contains integer data.

            Allows users to override this.
        """
        check_class(imageItem, pg.ImageItem, allowNone=True)

        if imageItem is None or imageItem.image is None:
            return False
        else:
            return imageItem.image.dtype.kind in "ui"


    @classmethod
    def _calcHistogramBins(cls, histRange, numBins=500, forIntegers=False):
        """ Calculates bins for the histogram give a subsampled image array (stepData)

            This function was based on PyQtGraph.ImageItem.getHistogram() commit efaf61f

            :returns: bins for use in numpy.histogram().
        """
        mn, mx = histRange

        if mn is None or np.isnan(mn) or mx is None or np.isnan(mx):
            # the data are all-nan
            return None, None

        if forIntegers:
            # For integer data, we select the bins carefully to avoid aliasing
            step = np.ceil((mx-mn) / float(numBins))
            logger.debug("mn {}, mx {}, step {}, mx+1.01*step = {}"
                         .format(mn, mx, step, mx+1.01*step))
            bins = np.arange(mn, mx+1.01*step, step, dtype=np.int)
        else:
            # for float data, let numpy select the bins.
            bins = np.linspace(mn, mx, numBins)

        if len(bins) == 0:
            bins = [mn, mx]

        return bins


    def mouseClickEvent(self, mouseClickEvent):
        """ Resets the color range if the middle mouse button is clicked.
        """
        if mouseClickEvent.button() in self.resetRangeMouseButtons:
            mouseClickEvent.accept()
            self.resetColorLevels()


    def getLabel(self):
        """ Returns the axis label. Returns None if no label present.
        """
        return self.axisItem.labelText


    def setLabel(self, text):
        """ Sets the axis label.

            Use None for no label (an empty string will take the screen space).
        """
        self.axisItem.setLabel(text)
        self.axisItem.showLabel(text is not None) # Is not done in axisItem.setLabel when None


    @QtCore.pyqtSlot()
    def resetColorLevels(self):
        """ Sets the color levels from the min and max of the image
        """
        logger.debug("Reset scale")

        img = self._imageItem.image
        check_is_an_array(img, allow_none=True)

        if img is None:
            levels = (0.0, 1.0)
        else:
            levels = (np.nanmin(img), np.nanmax(img))

        self.setLevels(levels)


    def getLevels(self):
        """ Gets the value range of the legend
        """
        levels = self.axisItem.range # which equals self.histViewBox.state['viewRange'][Y_AXIS]

        if self.orientation == 'right':
            assert self.axisItem.range == self.overlayViewBox.state['viewRange'][Y_AXIS], \
                "Sanity check failed {} != {}".format(self.axisItem.range,
                                           self.overlayViewBox.state['viewRange'][Y_AXIS])
        else:
            assert self.axisItem.range == self.overlayViewBox.state['viewRange'][X_AXIS], \
                "Sanity check failed {} != {}".format(self.axisItem.range,
                                           self.overlayViewBox.state['viewRange'][X_AXIS])
        return tuple(levels)


    @QtCore.pyqtSlot(tuple)
    def setLevels(self, levels, padding=0.0):
        """ Sets the value range of the legend.

            :param int padding: percentage that will be added to the color range.
                Use None for PyQtGraph's padding algorithm. Use 0 for no padding.
        """
        #logger.debug("ColorLegendItem.setLevels: {}".format(levels), stack_info=False)
        lvlMin, lvlMax = levels
        # Note: overlayViewBox.setYRange will call _updateImageLevels, which will
        # emit sigLevelsChanged
        if self.orientation == 'right':
            self.overlayViewBox.setYRange(lvlMin, lvlMax, padding=padding)
        else:
            self.overlayViewBox.setXRange(lvlMin, lvlMax, padding=padding)


    def getLut(self):
        """ Returns the Lookup table of the image item.
        """
        return self._imageItem.lut


    def setLut(self, lut):
        """ Sets the lookup table to the image item.

            :param ndarray Array: an N x 3 array.
        """
        logger.debug("------ setLut called")
        assertIsLut(lut)

        pgVersionInfo = versionStrToTuple(pg.__version__)
        if pgVersionInfo < (0, 11):
            if not isExtended(lut):
                # The lookup table in the imageItem must be extended. See extentLut doc string
                logger.debug("Side effect: duplicating last item of LUT in image item.")
                extendedLut = extentLut(lut)
            else:
                # The lookup table in the imageItem already is extended. Draw the original
                extendedLut = lut
                lut = np.copy(lut[0:-1, :])

            assert len(lut) == len(extendedLut) - 1, "Sanity check"
        else:
            extendedLut = lut

        if self._imageItem:
            logger.debug("Setting image item to extended lut")
            self._imageItem.setLookupTable(extendedLut)

        # Draw a color scale that shows the LUT.
        barWidth = 1
        m, n = lut.shape
        imgAxOrder = pg.getConfigOption('imageAxisOrder')
        if imgAxOrder == 'col-major' and self.orientation == 'right':
            lutImg = np.ones(shape=(barWidth, m, n), dtype=lut.dtype)
            lutImg[...] = lut[np.newaxis, :, :]
        elif imgAxOrder == 'col-major' and self.orientation == 'bottom':
            lutImg = np.ones(shape=(m, barWidth, n), dtype=lut.dtype)
            lutImg[...] = lut[:, np.newaxis, :]
        elif imgAxOrder == 'row-major' and self.orientation == 'right':
            lutImg = np.ones(shape=(m, barWidth, n), dtype=lut.dtype)
            lutImg[...] = lut[:, np.newaxis, :]
        elif imgAxOrder == 'row-major' and self.orientation == 'bottom':
            lutImg = np.ones(shape=(barWidth, m,  n), dtype=lut.dtype)
            lutImg[...] = lut[np.newaxis, :, :]
        else:
            raise AssertionError("Unexpected imageAxisOrder config value: {}".format(imgAxOrder))

        # if imgAxOrder == 'col-major' and self.orientation == 'right':
        #     lutImg = np.ones(shape=(barWidth, len(lut), 3), dtype=lut.dtype)
        #     lutImg[...] = lut[np.newaxis, :, :]
        # elif imgAxOrder == 'col-major' and self.orientation == 'bottom':
        #     lutImg = np.ones(shape=(len(lut), barWidth, 3), dtype=lut.dtype)
        #     lutImg[...] = lut[:, np.newaxis, :]
        # elif imgAxOrder == 'row-major' and self.orientation == 'right':
        #     lutImg = np.ones(shape=(len(lut), barWidth, 3), dtype=lut.dtype)
        #     lutImg[...] = lut[:, np.newaxis, :]
        # elif imgAxOrder == 'row-major' and self.orientation == 'bottom':
        #     lutImg = np.ones(shape=(barWidth, len(lut),  3), dtype=lut.dtype)
        #     lutImg[...] = lut[np.newaxis, :, :]
        # else:
        #     raise AssertionError("Unexpected imageAxisOrder config value: {}".format(imgAxOrder))


        logger.debug("lutImg.shape: {}".format(lutImg.shape))
        self.colorScaleImageItem.setImage(lutImg)

        yRange = [0, len(lut)]
        logger.debug("Setting colorScaleViewBox yrange to: {}".format(yRange))

        # Do not set disableAutoRange to True in setRange; it triggers 'one last' auto range.
        # This is why the viewBox' autorange must be False at construction.

        if self.orientation == 'right':
            self.colorScaleViewBox.setRange(
                xRange=[0, barWidth], yRange=yRange, padding=0.0,
                update=False, disableAutoRange=False)
        else:
            self.colorScaleViewBox.setRange(
                xRange=yRange, yRange=[0, barWidth], padding=0.0,
                update=False, disableAutoRange=False)


    @property
    def histogramIsVisible(self):
        """ Indicates if the histogram is visible
        """
        return self._histogramIsVisible


    # @QtCore.Slot(bool)
    @QtCore.pyqtSlot(bool)
    def showHistogram(self, show):
        """ Toggle histogram on or off.
        """
        #logger.debug("showHistogram: {}".format(show))
        self._histogramIsVisible = show
        if show:
            if self.orientation == 'right':
                self.histViewBox.setFixedWidth(self.histogramHeight)
            else:
                self.histViewBox.setFixedHeight(self.histogramHeight)
            self.histPlotDataItem.show()
            self._updateHistogram()
        else:
            if self.orientation == 'right':
                self.histViewBox.setFixedWidth(1) # zero gives error
            else:
                self.histViewBox.setFixedHeight(1)  # zero gives error
            self.histPlotDataItem.hide()


    def fillHistogram(self, fill=True, level=0.0, color=(100, 100, 200)):
        """ Fills the histogram
        """
        if fill:
            self.histPlotDataItem.setFillLevel(level)
            self.histPlotDataItem.setFillBrush(color)
        else:
            self.histPlotDataItem.setFillLevel(None)
