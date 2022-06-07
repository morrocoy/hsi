# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:00:00 2022

@author: kpapke
"""
import numpy
from scipy import ndimage, signal
from ..core.hs_functions import rescale_intensity, snv

from ..core.hs_formats import convert
from ..core.hs_formats import HSIntensity
from ..log import logmanager

from .hs_base_analysis import HSBaseAnalysis


logger = logmanager.getLogger(__name__)


__all__ = ['HSOxygen']


class HSOxygen(HSBaseAnalysis):
    """
    Class to analyze hyper spectral image data by using the Moussa's algorithms
    for oxygen detection.

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
        super(HSOxygen, self).__init__(spectra, wavelen, hsformat)
        self.prefix = "lipids_"
        self.keys = [
            'oxygen_ox0',
            # 'oxygen_ox1',
            # 'oxygen_ox2',
        ]
        self.labels = [
            "Oxygen Angle across 630-710 nm",
            # "label_1",
            # "label_2",
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

            spectra_oxy = -numpy.log(numpy.abs(intensity))
            spectra_oxy[spectra_oxy == numpy.inf] = 0
            spectra_oxy = ndimage.uniform_filter(spectra_oxy, size=5)
            spectra_oxy = snv(spectra_oxy)
            spectra_oxy = spectra_oxy.reshape(m, -1)

            # retrieve the selected spectra
            index_mask = self._ravel_mask(mask)

            m, n = spectra.shape  # number wavelengths, spectra

            _nkeys = len(self.keys)
            self._anaVarVector = numpy.zeros((_nkeys, n))
            self._anaVarScales = numpy.ones((_nkeys, n))
            self._anaVarBounds = numpy.zeros((_nkeys, 2))
            self._anaVarBounds[:, 1] = 1

            b = spectra_oxy[:, index_mask]

            self._anaVarVector[0, index_mask] = self.evaluate_oxygen_0(
                b, wavelen)
            # self._anaVarVector[1, index_mask] = self.evaluate_oxygen_1(
            #     b, wavelen)

    @staticmethod
    def evaluate_oxygen_0(spectra, wavelen, reg0=None):
        # method 0: oxygenation from angle at 630-710nm
        if reg0 is None:
            reg0 = [630e-9, 710e-9]

        idx0 = numpy.where((wavelen >= reg0[0]) * (wavelen <= reg0[1]))[0]
        # val0 = numpy.mean(spectra[idx0], axis=0, dtype=numpy.float64)

        y1 = spectra[idx0]
        x1 = range(len(idx0))

        ratio = numpy.arctan2(y1[-1] - y1[0], x1[-1] - x1[0])
        ratio = 10 * numpy.rad2deg(ratio)
        ratio[ratio < -34.] = -33.
        ratio[ratio > -8.] = -8.
        ratio[ratio == 0.] = -35.
        ratio = rescale_intensity(ratio, (-34., -8.), (0, 1))

        return ratio.astype('float64')

    # @staticmethod
    # def evaluate_oxygen_1(spectra, wavelen, reg0=None):
    #     # method 1: Inverse fat angle index as water index
    #     # using negative angle of method 5 and different clipping
    #     if reg0 is None:
    #         reg0 = [900e-9, 915e-9]
    #
    #     idx0 = numpy.where((wavelen >= reg0[0]) * (wavelen <= reg0[1]))[0]
    #     # val0 = numpy.mean(spectra[idx0], axis=0, dtype=numpy.float64)
    #
    #     y1 = spectra[idx0]*2000
    #     x1 = range(len(idx0))
    #
    #     ratio = numpy.arctan2(y1[-1] - y1[0], x1[-1] - x1[0])
    #     ratio = numpy.rad2deg(ratio)
    #     ratio[ratio < -15] = -14
    #     ratio[ratio > 83] = 83
    #     ratio[ratio == 0] = -15
    #     ratio = rescale_intensity(ratio, (-15, 83), (0, 1))
    #
    #     return ratio.astype('float64')