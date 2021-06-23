# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:42:17 2021

@author: kpapke
"""
import numpy

from scipy.interpolate import interp1d

from ..log import logmanager

logger = logmanager.getLogger(__name__)

__all__ = ['HSTissueComponent']


class HSTissueComponent:
    """
    Class to represent a tissue component.

    Objects of this class may be used to load mass attenuation coefficients,
    to interpolate them on given wavelength samples.

    Attributes
    ----------
    absorption : numpy.ndarray
        The interpolated spectral information for the attenuation
        coefficient [cm-1].
    wavelen :  numpy.ndarray, optional
        The wavelengths [nm] at which the spectral information for the
        attenuation coefficient will be interpolated.
    """

    def __init__(self, yn, xn, wavelen=None):
        """ Constructor

        Parameters
        ----------
        yn :  numpy.ndarray
            The spectral information for the attenuation coefficient.
        xn :  numpy.ndarray
            The wavelengths at which the spectral information is sampled.
        wavelen :  numpy.ndarray, optional
            The wavelengths at which the spectral information will be
            interpolated.

        """
        self._wavelen = None  # raw data wavelength.
        self._absorption = None  # raw data for absorption coefficient
        self._interp = None  # interpolator

        self.wavelen = None  # interpolated wavelength
        self.absorption = None  # interpolated absorption coefficient

        # set spectral data and
        self.set_data(yn, xn, wavelen)

    def interpolate(self):
        """Realign spectral data according to the interpolation points."""
        if self.wavelen is None or self._interp is None:
            self.absorption = None
            logger.debug("Interpolator or interpolation points undefined. "
                         "Skip interpolation")
            return

        if numpy.array_equal(self.wavelen, self._wavelen):
            self.absorption = self._absorption
        else:
            self.absorption = self._interp(self.wavelen)

    def set_data(self, yn, xn, wavelen=None):
        """Set the spetral data.

        Parameters
        ----------
        yn :  numpy.ndarray
            The spectral information for the attenuation coefficient.
        xn :  numpy.ndarray
            The wavelengths [nm] at which the spectral information is sampled.
        wavelen :  numpy.ndarray, optional
            The wavelengths [nm] at which the spectral information will be
            interpolated.
        """
        if isinstance(yn, list):
            yn = numpy.array(yn)
        if not isinstance(yn, numpy.ndarray) or yn.ndim > 1:
            raise Exception(
                "Nodal y data for base spectrum must be 1D ndarray.")

        if isinstance(xn, list):
            xn = numpy.array(xn)
        if not isinstance(xn, numpy.ndarray) or xn.ndim > 1:
            raise Exception(
                "Nodal x data for base spectrum must be 1D ndarray.")

        if len(xn) != len(yn):
            raise Exception(
                "Nodal x and y data for base spectrum must be of same length.")

        self._absorption = yn.view(numpy.ndarray)
        self._wavelen = xn.view(numpy.ndarray)

        if wavelen is not None:
            if isinstance(wavelen, list):
                wavelen = numpy.array(wavelen)
            if not isinstance(wavelen, numpy.ndarray) or wavelen.ndim > 1:
                raise Exception("Interpolation x data must be 1D ndarray.")
            self.wavelen = wavelen.view(numpy.ndarray)

        else:
            self.wavelen = xn.view(numpy.ndarray)

        # set interpolator
        self.set_interpolator(kind='linear')

    def set_interpolator(self, kind='linear', bounds_error=None,
                         fill_value=numpy.nan, assume_sorted=False):
        """Set the interpolator for the base spectrum.

        Forwards all arguments to :class:`scipy.interpolate.interp1d`.
        """
        if self._wavelen is None:
            raise Exception("Nodal Data missing for Interpolation.")

        self._interp = interp1d(
            self._wavelen, self._absorption, kind=kind, fill_value=fill_value,
            bounds_error=bounds_error, assume_sorted=assume_sorted)
        self.interpolate()

    def set_interp_points(self, wavelen):
        """Set interpolation points used to realignment the spectral data.

        Parameters
        ----------
        wavelen :  numpy.ndarray
            The wavelengths at which the spectral information will be
            interpolated.
        """
        if isinstance(wavelen, list):
            wavelen = numpy.array(wavelen)
        if not isinstance(wavelen, numpy.ndarray) or wavelen.ndim > 1:
            raise Exception("Interpolation samples must be 1D ndarray.")

        self.wavelen = wavelen.view(numpy.ndarray)
        self.interpolate()
