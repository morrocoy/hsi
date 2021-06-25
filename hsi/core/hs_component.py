# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 07:55:39 2021

@author: kpapke
"""
import numpy
from scipy.interpolate import interp1d

from ..log import logmanager

from .hs_formats import HSFormatFlag, HSFormatDefault, convert

logger = logmanager.getLogger(__name__)


__all__ = ['HSComponent']


class HSComponent:
    """
    Class to represent a base vector associated with a certain spectrum.

    Objects of this class may be used to represent a spectral data set,
    hence, they contain spectral information at a give set wavelengths referred
    as the nodal points. In addition, the spectral information may be
    re-sampled on a unique set of wavelengths which are referred as the
    interpolation points and are typically common to all base vectors. Each
    base vector provides a weight by which they are normalize and weighted to
    each other. Moreover, lower and upper bounds may be defined for the
    absolute weights.

    Attributes
    ----------
    name :  str
        Name of the base spectrum.
    label :  str
        Label of the base spectrum .
    weight :  float
        The scaling factor for the base spectrum.
    bounds :  list
        The lower and upper bounds for the scaling factor.
    xNodeData :  numpy.ndarray
        The wavelengths at which the spectral data are sampled.
    yNodeData :  numpy.ndarray
        The spectral data.
    xIntpData :  numpy.ndarray
        The wavelengths at which the spectral information will be interpolated.
    yIntpData :  numpy.ndarray
        The spectral data interpolated at xIntpData.
    hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
        The hsformat for the hyperspectral data. Should be one of:

            - :class:`HSIntensity<hsi.HSIntensity>`
            - :class:`HSAbsorption<hsi.HSAbsorption>`
            - :class:`HSExtinction<hsi.HSExtinction>`
            - :class:`HSRefraction<hsi.HSRefraction>`


    """

    _counter = 0  # counter to allow auto labeling for each instance

    def __init__(self, yn, xn, x=None, name=None, label=None,
                 hsformat=HSFormatDefault, weight=1., bounds=None):
        """ Constructor

        Parameters
        ----------
        yn :  numpy.ndarray
            The spectral data.
        xn :  numpy.ndarray
            The wavelengths at which the spectral data are sampled.
        x :  numpy.ndarray, optional
            The wavelengths at which the spectral information will be
            interpolated.
        name :  str, optional
            Name of the base spectrum.
        label :  str
            Label of the base spectrum.
        hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The hsformat for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`

        weight :  float, optional
            The weight for the base spectrum.
        bounds :  list or tuple, optional
            The lower and upper bounds for the scaling factor.

        """
        self.xNodeData = None
        self.yNodeData = None

        self._interp = None
        self.xIntpData = None
        self.yIntpData = None

        # name identifier
        if name is None:
            self.name = "vec_%d" % type(self)._counter
            type(self)._counter += 1
        else:
            self.name = name

        # alias name for a better description
        if label is None:
            self.label = self.name
        else:
            self.label = label

        if bounds is None:
            bounds = [None, None]

        # spectral hsformat
        if not HSFormatFlag.has_flag(hsformat):
            raise Exception("Unknown hsformat '{}'.".format(hsformat))
        self.hsformat = hsformat

        # weight and lower and upper bounds for the weight
        self.weight = weight
        self.bounds = None
        self.scale = None

        self.set_bounds(bounds)
        self.set_data(yn, xn, x)

    def get_scaled_bounds(self):
        """Get the normalized lower and upper bounds for the weight ."""
        lbnd, ubnd = self.bounds
        if lbnd is not None:
            lbnd = lbnd / self.weight / self.scale
        if ubnd is not None:
            ubnd = ubnd / self.weight / self.scale
        return [lbnd, ubnd]

    def get_scaled_value(self):
        """Get the base vector normalized by the weight value."""
        return self.yIntpData * self.scale

    def interpolate(self):
        """Realign spectral data according to the interpolation points."""
        if self._interp is None:
            raise Exception("Interpolator not defined.")
        if self.xIntpData is None or self.xNodeData is None:
            raise Exception(
                "Interpolation or nodal x data for resampling missing.")

        if numpy.array_equal(self.xIntpData, self.xNodeData):
            self.yIntpData = self.yNodeData
        elif (numpy.min(self.xNodeData) > numpy.min(self.xIntpData) or
              numpy.max(self.xNodeData) < numpy.max(self.xIntpData)):
            raise ValueError("Interpolation is attempted on a value outside "
                             "of the range of x.")
        else:
            self.yIntpData = self._interp(self.xIntpData)

    def len(self):
        """Get length of base vector (resampled spectral data)."""
        if self.yIntpData is not None:
            return len(self.yIntpData)
        else:
            return None

    @property
    def shape(self):
        if self.yIntpData is not None:
            return len(self.yIntpData)
        else:
            return tuple()

    def set_data(self, yn, xn, x=None):
        """Set the spetral data.

        Parameters
        ----------
        yn :  numpy.ndarray
            The spectral data.
        xn :  numpy.ndarray
            The wavelengths at which the spectral data are sampled.
        x :  numpy.ndarray, optional
            The wavelengths at which the spectral information will be
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

        self.yNodeData = yn.view(numpy.ndarray)
        self.xNodeData = xn.view(numpy.ndarray)

        if x is not None:
            if isinstance(x, list):
                x = numpy.array(x)
            if not isinstance(x, numpy.ndarray) or x.ndim > 1:
                raise Exception("Interpolation x data must be 1D ndarray.")
            self.xIntpData = x.view(numpy.ndarray)

        else:
            self.xIntpData = xn.view(numpy.ndarray)

        self.set_interp(kind='cubic', bounds_error=True)

    def set_interp(self, kind='linear', bounds_error=None,
                   fill_value=numpy.nan, assume_sorted=False):
        """Set the interpolator for the base spectrum.

        Forwards all arguments to :class:`scipy.interpolate.interp1d`.
        """
        if self.xNodeData is None:
            raise Exception("Nodal Data missing for Interpolation.")

        self._interp = interp1d(
            self.xNodeData, self.yNodeData, kind=kind, fill_value=fill_value,
            bounds_error=bounds_error, assume_sorted=assume_sorted)
        self.interpolate()

    def set_interp_points(self, x):
        """Set interpolation points used to realignment the spectral data.

        Parameters
        ----------
        x :  numpy.ndarray
            The wavelengths at which the spectral information will be
            interpolated.
        """
        if isinstance(x, list):
            x = numpy.array(x)
        if not isinstance(x, numpy.ndarray) or x.ndim > 1:
            raise Exception("Interpolation x data for base spectrum must be "
                            "1D ndarray.")

        self.xIntpData = x.view(numpy.ndarray)
        self.interpolate()

    def set_bounds(self, bounds):
        """Set the absolute bounds.

        Parameters
        ----------
        bounds :  list
            The lower and upper bounds for the scaling factor.
        """
        if bounds is None:
            self.bounds = [None, None]
        elif type(bounds) in [list, tuple, numpy.ndarray] and len(bounds) == 2:
            if bounds[0] is None or numpy.isinf(bounds[0]):
                lbnd = None
            elif isinstance(bounds[0], (int, float)):
                lbnd = bounds[0]
            else:
                lbnd = None

            if bounds[1] is None or numpy.isinf(bounds[1]):
                ubnd = None
            elif isinstance(bounds[1], (int, float)):
                ubnd = bounds[1]
            else:
                ubnd = None

            self.bounds = [lbnd, ubnd]
        else:
            raise ValueError("Argument 'bounds' must be two element list or "
                             "tuple. Got {}".format(bounds))

        self.udpdate_scale()

    def set_format(self, hsformat):
        """Set the hsformat of the the spectral data.

        Parameters
        ----------
        hsformat : :obj:`HSFormatFlag<hsi.HSFormatFlag>`
            The spectral hsformat to be set. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`

        """
        if not HSFormatFlag.has_flag(hsformat):
            raise Exception("Unknown hsformat '{}'.".format(hsformat))

        if self.yIntpData is not None:
            old_format = self.hsformat
            self.yNodeData = convert(
                hsformat, old_format, self.yNodeData, self.xNodeData)
            self.set_interp(kind='cubic', bounds_error=True)

        self.hsformat = hsformat

    def set_weight(self, value, bounds=None):
        """Set the weight and absolute bounds.

        Parameters
        ----------
        value :  float
            The weight for the base spectrum.
        bounds :  list
            The lower and upper bounds for the scaling factor.
        """
        self.weight = value

        if bounds is None:
            self.bounds = [None, None]
        elif type(bounds) in [list, tuple, numpy.ndarray] and len(bounds) == 2:
            self.bounds = [bounds[0], bounds[1]]
        else:
            raise ValueError("Argument 'bounds' must be two element list or "
                             "tuple. Got {}".format(bounds))

        self.udpdate_scale()

    def udpdate_scale(self):
        lbnd, ubnd = self.bounds

        if lbnd is None and ubnd is None:
            self.scale = 1. / self.weight
            # self.scale = 1.
        elif ubnd is None:
            self.scale = abs(self.weight - lbnd) / self.weight
        elif lbnd is None:
            self.scale = (ubnd - self.weight) / self.weight
        else:
            self.scale = (ubnd - lbnd) / self.weight
