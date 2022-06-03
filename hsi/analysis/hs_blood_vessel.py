# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 13:09:21 2022

@author: kpapke
"""
import numpy
from scipy import ndimage, signal
# from skimage.exposure import rescale_intensity
from ..core.hs_functions import rescale_intensity

from ..core.hs_formats import convert
from ..core.hs_formats import HSIntensity, HSAbsorption
from ..core.hs_functions import snv
from ..log import logmanager

from .hs_base_analysis import HSBaseAnalysis


logger = logmanager.getLogger(__name__)


__all__ = ['HSBloodVessel']


class HSBloodVessel(HSBaseAnalysis):
    """
    Class to analyze hyper spectral image data by using the Moussa's algorithms
    for the detection of blood vessels.

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
        super(HSBloodVessel, self).__init__(spectra, wavelen, hsformat)
        self.keys = ['bv0', 'bv1', 'bv2']
        self.labels = [
            "Angle Index SNV at 625-720 nm",
            "Mean at 750-950 nm",
            "Mask Combination",
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

            # filtered target vector in a reshaped 2-dimensional hsformat
            intensity = convert(
                HSIntensity, self.hsformat, self.spectra, wavelen)
            Abs = convert(
                HSAbsorption, self.hsformat, self.spectra, wavelen)
            Abs[Abs == numpy.inf] = 0
            AbsMean = ndimage.uniform_filter(Abs, size=5)
            spectra_snv = snv(AbsMean)

            # unfiltered target vector in a reshaped 2-dimensional hsformat
            spectra = AbsMean.reshape(m, -1)
            intensity = intensity.reshape(m, -1)
            spectra_snv = spectra_snv.reshape(m, -1)

            # retrieve the selected spectra
            index_mask = self._ravel_mask(mask)

            m, n = spectra.shape  # number wavelengths, spectra
            self._anaVarVector = numpy.zeros((4, n))
            self._anaVarScales = numpy.ones((4, n))
            self._anaVarBounds = numpy.zeros((4, 2))
            self._anaVarBounds[:, 1] = 1

            b0 = intensity[:, index_mask]
            b1 = spectra_snv[:, index_mask]

            self._anaVarVector[0, index_mask] = self.evaluate_bv_0(
                b1, wavelen)
            self._anaVarVector[1, index_mask] = self.evaluate_bv_1(
                b0, wavelen)
            self._anaVarVector[2, index_mask] = rescale_intensity(
                (1. - self._anaVarVector[0, index_mask]) *
                self._anaVarVector[1, index_mask], (0, 1), (0, 1))

    @staticmethod
    def evaluate_bv_0(spectra, wavelen, reg0=None):
        # method 1: angle index at 625-720 nm
        if reg0 is None:
            reg0 = [625e-9, 720e-9]

        idx0 = numpy.where((wavelen >= reg0[0]) * (wavelen <= reg0[1]))[0]
        # val0 = numpy.mean(spectra[idx0], axis=0, dtype=numpy.float64)

        y1 = spectra[idx0]
        x1 = range(len(idx0))

        ratio = numpy.arctan2(x1[-1] - x1[0], y1[-1] - y1[0])
        # ratio = rescale_intensity(
        #     ratio, (numpy.min(ratio), numpy.max(ratio)), (0, 100))

        # ratio[ratio < 1.5707963267948966] = 1.5707963267948966

        ratio = numpy.clip(ratio, numpy.pi/2, numpy.pi)
        ratio = rescale_intensity(
            ratio, (numpy.min(ratio), numpy.max(ratio)), (0, 1))

        return ratio.astype('float64')

    @staticmethod
    def evaluate_bv_1(spectra, wavelen, reg0=None):
        # method 2: mean intensity at 750-950 nm
        if reg0 is None:
            reg0 = [750e-9, 950e-9]

        idx0 = numpy.where((wavelen >= reg0[0]) * (wavelen <= reg0[1]))[0]
        ratio = numpy.mean(spectra[idx0], axis=0, dtype=numpy.float64)
        ratio = rescale_intensity(
            ratio, (numpy.min(ratio), numpy.max(ratio)), (0, 1))

        return ratio.astype('float64')
