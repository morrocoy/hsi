# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:22:49 2022

@author: kpapke
"""
import numpy
from scipy import ndimage, signal
from skimage.exposure import rescale_intensity

from ..core.hs_formats import convert
from ..core.hs_formats import HSIntensity
from ..log import logmanager

from .hs_base_analysis import HSBaseAnalysis


logger = logmanager.getLogger(__name__)


__all__ = ['HSFat']


class HSFat(HSBaseAnalysis):
    """
    Class to analyze hyper spectral image data by using the Moussa's algorithms
    for fat detection.

    Attributes
    ----------
    wavelen :  numpy.ndarray
        The wavelengths at which the spectral data are sampled.
    spectra :  numpy.ndarray
        The spectral data.
    hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
        The hsformat for the hyperspectral data. Should be one of:

            - :class:`HSIntensity<hsi.HSIntensity>`
            - :class:`HSAbsorption<hsi.HSAbsorption>`
            - :class:`HSExtinction<hsi.HSExtinction>`
            - :class:`HSRefraction<hsi.HSRefraction>`

    keys : dict
        A dictionary of solution parameter keys and associated labels.

    """

    def __init__(self, spectra=None, wavelen=None, hsformat=HSIntensity):
        """ Constructor

        Parameters
        ----------
        spectra :  numpy.ndarray, optional
            The spectral data.
        wavelen :  numpy.ndarray, optional
            The wavelengths at which the spectral data are sampled.
        hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The hsformat for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`
        """
        super(HSFat, self).__init__(spectra, wavelen, hsformat)
        self.keys = ['li0', 'li1', 'li2', 'li3']
        self.labels = [
            "Fat Angle across 900-920 nm",
            "Fat index 1: NDI 925/960 nm",
            "Fat index 2: NDI 925/875 nm",
            "Fat 2nd Derivative @ 925 nm",
        ]

    def evaluate(self, mask=None):
        """Evaluate fat index values.

        Parameters
        ----------
        mask : (tuple, list, or numpy.ndarray), optional
            Evaluate the fit only for selected spectra using either a tuple,
            list, array of integer arrays, one for each dimension, or a boolean
            array serving as a mask.
        """
        # vector of unknowns for each spectrum

        if self.spectra is None:
            self._anaVarVector = None
            self._anaVarBounds = None

        else:
            # wavelength axis
            wavelen = self.wavelen

            # number of wavelengths
            m = len(self.spectra)

            # unfiltered target vector in a reshaped 2-dimensional hsformat
            spectra = self.spectra.reshape(m, -1)

            # filtered target vector in a reshaped 2-dimensional hsformat
            intensity = convert(
                HSIntensity, self.hsformat, self.spectra, wavelen)
            spectra_fat = -numpy.log(numpy.abs(intensity))
            spectra_fat[spectra_fat == numpy.inf] = 0
            spectra_fat = ndimage.uniform_filter(spectra_fat, size=7)
            spectra_fat = numpy.pad(
                spectra_fat, pad_width=((1, 1), (0, 0), (0, 0)),
                mode='symmetric')
            spectra_fat = numpy.diff(spectra_fat, n=2, axis=0)
            spectra_fat = ndimage.uniform_filter(spectra_fat, size=7)
            spectra_fat = signal.savgol_filter(
                spectra_fat, window_length=9, polyorder=2, axis=0,
                mode='mirror')
            spectra_fat = spectra_fat.reshape(m, -1)

            # retrieve the selected spectra
            index_mask = self._ravel_mask(mask)

            m, n = spectra.shape  # number wavelengths, spectra
            self._anaVarVector = numpy.zeros((4, n))
            self._anaVarScales = numpy.ones((4, n))
            self._anaVarBounds = numpy.zeros((4, 2))
            self._anaVarBounds[:, 1] = 1

            b = spectra_fat[:, index_mask]

            self._anaVarVector[0, index_mask] = self.evaluate_fat1(
                b, wavelen)
            self._anaVarVector[1, index_mask] = self.evaluate_fat2(
                b, wavelen)
            self._anaVarVector[2, index_mask] = self.evaluate_fat3(
                b, wavelen)
            self._anaVarVector[3, index_mask] = self.evaluate_fat4(
                b, wavelen)

    @staticmethod
    def evaluate_fat1(spectra, wavelen, reg0=None):
        # method 1: Fat angle index
        if reg0 is None:
            reg0 = [900e-9, 915e-9]

        idx0 = numpy.where((wavelen >= reg0[0]) * (wavelen <= reg0[1]))[0]
        # val0 = numpy.mean(spectra[idx0], axis=0, dtype=numpy.float64)

        y1 = spectra[idx0]
        x1 = range(len(idx0))

        ratio = numpy.arctan2(x1[-1] - x1[0], y1[-1] - y1[0])
        ratio = rescale_intensity(
            ratio, (numpy.min(ratio), numpy.max(ratio)), (0, 1))

        return ratio.astype('float64')

    @staticmethod
    def evaluate_fat2(spectra, wavelen, reg0=None):
        # method 2: ratio index at 925 nm and 965 nm
        if reg0 is None:
            reg0 = [925e-9, 960e-9]

        idx0 = numpy.where((wavelen >= reg0[0]) * (wavelen <= reg0[1]))[0]
        # val0 = numpy.mean(spectra[idx0], axis=0, dtype=numpy.float64)

        y1 = spectra[idx0] + 1.

        ratio = (y1[-1] - y1[0]) / (y1[-1] + y1[0])
        ratio[ratio == 0] = numpy.min(ratio)  # something strange
        ratio = rescale_intensity(
            ratio, (numpy.min(ratio), numpy.max(ratio)), (0, 1))

        return ratio.astype('float64')

    @staticmethod
    def evaluate_fat3(spectra, wavelen, reg0=None):
        # method 3: ratio index at 925 nm and 875 nm
        if reg0 is None:
            reg0 = [875e-9, 925e-9]

        spectra[spectra == 0] = 1

        idx0 = numpy.where((wavelen >= reg0[0]) * (wavelen <= reg0[1]))[0]
        y1 = spectra[idx0] + 1.

        ratio = -(y1[-1] - y1[0]) / (y1[-1] + y1[0])
        ratio[ratio == 0] = numpy.min(ratio)  # something strange
        ratio = rescale_intensity(
            ratio, (numpy.min(ratio), numpy.max(ratio)), (0, 1))

        return ratio.astype('float64')

    @staticmethod
    def evaluate_fat4(spectra, wavelen, reg0=None):
        # method 4: second derivative at 925 nm
        if reg0 is None:
            reg0 = [925e-9, 925e-9]

        spectra[spectra == 0] = 1

        idx0 = numpy.where((wavelen >= reg0[0]) * (wavelen <= reg0[1]))[0]
        y1 = spectra[idx0]

        ratio = y1[0]
        ratio[ratio == 1.] = numpy.min(ratio)  # something strange
        ratio = rescale_intensity(
            ratio, (numpy.min(ratio), numpy.max(ratio)), (0., 1.))

        ratio = 1. - ratio  # invert
        ratio[ratio == 1.] = 0.

        return ratio.astype('float64')
