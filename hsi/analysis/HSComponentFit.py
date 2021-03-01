# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:30:26 2020

@author: papkai
"""
import os.path
import numpy as np
from scipy.optimize import minimize, OptimizeResult, nnls, lsq_linear

from .. import CONFIG_OPTIONS
from .. import __version__

from ..misc import getPkgDir
from ..core.HSFile import HSFile
from ..core.formats import HSFormatFlag, HSAbsorption, HSFormatDefault, convert

from .HSComponent import HSComponent
from .HSComponentFile import HSComponentFile

import logging

LOGGING = True
# LOGGING = False
logger = logging.getLogger(__name__)
logger.propagate = LOGGING


__all__ = ['HSComponentFit']


if CONFIG_OPTIONS['enableBVLS']:
    import bvls
    def bvls_f(*args, **kwargs):
        return bvls.bvls(*args, **kwargs)
else:
    def bvls_f(*args, **kwargs):
        return None


class HSComponentFit:
    """
    Class to approximate hyper spectral image data by a weighted sum of base
    spectra in order to analysize their individual contributions.

    Features:

    - load base spectra configuration
    - linear, nonlinear, constrained and unconstrained approximations

    Attributes
    ----------
    xData :  numpy.ndarray
        The wavelengths at which the spectral data are sampled.
    yData :  numpy.ndarray
        The spectral data.
    baseVectors : dict of HSComponent
        A dictionary of base vector to represent the spectral data.
    roi : list of float
        The lower and upper bounds for the wavelength region of interest.
    roiIndex : list of int
        The lower and upper bound indices for the wavelength region of interest.
    format :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
        The format for the hyperspectral data. Should be one of:

            - :class:`HSIntensity<hsi.HSIntensity>`
            - :class:`HSAbsorption<hsi.HSAbsorption>`
            - :class:`HSExtinction<hsi.HSExtinction>`
            - :class:`HSRefraction<hsi.HSRefraction>`


    """

    def __init__(self, y=None, x=None, bounds=None, format=HSAbsorption):
        """ Constructor

        Parameters
        ----------
        y :  numpy.ndarray, optional
            The spectral data.
        x :  numpy.ndarray, optional
            The wavelengths at which the spectral data are sampled.
        bounds :  list or tuple, optional
            The lower and upper bounds for the region of interest.
        format :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The format for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`
        """
        self.xData = None  # wavelength axis
        self.yData = None  # image data flatten to 2D ndarray

        self.baseVectors = {}  # dict of base vector to represent spectral data

        self._lsTrgVector = None
        self._lsVarVector = None  # vector of unknowns
        self._lsResVector = None  # residual vector (squared Euclidean 2-norm)
        self._lsSysMatrix = None  # assembly of normalized based vector
        self._lsVarScales = None  # scale factors for each unknown variable
        self._lsVarBounds = None  # bounds for each unknown variable

        # list of bounds for the region of interest
        self.roi = [None, None]
        self.roiIndex = [None, None]

        # check format, if not previously defined also set the format
        if not HSFormatFlag.hasFlag(format):
            raise Exception("Unknown format '{}'.".format(format))
        self.format = format

        # Forwards data arguments to self.setData() if available
        if y is not None:
            self.setData(y, x)

        # adopt roi bounds and corresponing indices
        self.setROI(bounds)


    def _func(self, x, a, b):
        # residual function to be minimized
        eps = np.einsum('ij,j->i', a, x) - b
        return np.einsum('i,i->', eps, eps) / len(eps)
        # return np.sum(eps ** 2) / len(eps)


    def _ravelMask(self, mask=None):
        """Convert a mask to raveled indices for fitting selectively.

        Parameters
        ----------
        mask : (tuple, list, or numpy.ndarray), optional
            Evaluate the fit only for selected spectra using either a tuple,
            list, array of integer arrays, one for each dimension, or a boolean
            array serving as a mask.

        Returns
        -------
        raveled_indices: numpy.ndarray
            An array of indices into the flattened version of
            :attr:`~.HSImageLSAnalysis.yData` apart from the first dimension
            which corresponds to the wavelength.
        """
        shape = self.yData.shape[1:]
        if isinstance(mask, (tuple, list, np.ndarray)) and len(mask) == len(shape):
            # tuple, list, or array of integer arrays, one for each dimension
            raveledMask = np.ravel_multi_index(mask, shape, mode='raise')
            if not hasattr(raveledMask, '__len__'):  # convert integer to list
                raveledMask = [raveledMask]
            logger.debug("Index Mask: {} to {} {}".format(mask, raveledMask, type(raveledMask)))
        elif isinstance(mask, np.ndarray) and mask.shape == shape:
            # array of boolean providing the same shape as the spectral data
            rmask = mask.reshape([-1])
            raveledMask = np.where(rmask)[0]  # where returns a tuple of array
            logger.debug("Boolean Mask: {} to {}".format(mask, raveledMask))
        else:
            # select all spectra if no mask defined
            raveledMask = range(int(np.prod(shape)))
            logger.debug("No Mask")

        return raveledMask

        # if type(index) in (tuple, list) and len(index) != len(shape):
        #     self.testIndex = np.ravel_multi_index(index, shape, mode='clip')
        # elif isinstance(index, int):
        #     self.testIndex = np.clip(index, 0, np.prod(shape)-1)
        # else:
        #     self.testIndex = 0


    def addBaseVector(self, y, x, name=None, label=None, format=None,
                      weight=1., bounds=[None, None]):
        """Add base vector to represent spectral data

        Forwards all arguments to :class:`hsi.analysis.HSBaseVector`.
        The wavelengths at which the spectral information will be interpolated
        are taken from the internally stored array
        :attr:`~.HSImageLSAnalysis.xData` that is common to all base vectors.

        Parameters
        ----------
        y :  numpy.ndarray
            The base spectral data.
        x :  numpy.ndarray
            The wavelengths at which the base spectral data are sampled.
        name :  str, optional
            Name of the base spectrum.
        label :  str, optional
            Label of the base spectrum.
        weight :  float, optional
            The weight for the base spectrum.
        bounds :  list or tuple, optional
            The lower and upper bounds for the scaling factor.
        """
        if format is None:
            format = self.format

        if self.xData is None:
            raise Exception("Spectral x data are missing for base vector.")
        else:
            self.baseVectors[name] = HSComponent(
                y, x, self.xData, name=name, label=label, format=format,
                weight=weight, bounds=bounds)


    def clear(self):
        """ Clear all spectral information including base vectors."""
        self.xData = None  # wavelength axis
        self.yData = None  # image data flatten to 2D ndarray

        self.roiIndex = [None, None]
        self.baseVectors.clear()

        self._lsTrgVector = None
        self._lsVarVector = None
        self._lsResVector = None
        self._lsSysMatrix = None
        self._lsVarScales = None
        self._lsVarBounds = None


    def fit(self, method='gesv', **kwargs):
        """Fit spectral data

        A least square problem is solved to Approximate spectral data by a
        weighted sum of the base vectors

        Parameters
        ----------
        method : str
            The least square method. Should be one of the method supported by
            :func:`HSImageLSAnalysis.fitLinear` or
            :func:`HSImageLSAnalysis.fitNonLinear`.
        **kwargs : dict
            Key word arguments serving as options for the fitting methods.
            Depending whether the method is linear or not, the arguments are
            forwarded to :func:`HSImageLSAnalysis.fitLinear` or
            :func:`HSImageLSAnalysis.fitNonLinear`.
        """
        meth = method.lower()
        if meth in ('gesv', 'lstsq', 'bvls', 'bvls_f', 'nnls', 'trf'):
            self.fitLinear(method, **kwargs)
        elif meth in ('cg', 'nelder-mead', 'l-bfgs-b', 'powell', 'slsqp',
                      'tnc', 'trust-constr'):
            self.fitNonlinear(method, **kwargs)
        else:
            raise ValueError("Method {} not supported.".format(method))


    def fitLinear(self, method='gesv', normal=True, mask=None,
                  tol=1e-10, maxiter=None, verbose=0, **kwargs):
        """Linear constrained or unconstrained fit.

        Parameters
        ----------
        method : str, optional
            The least square method. Should be one of

                - 'bvls'     : Bounded value least square algorithm.
                - 'bvls_f'   : Bounded value least square algorithm, fortran.
                - 'gesv'     : Linear matrix equation (unconstrained).
                - 'lstsq'    : Least square algorithm (unconstrained).
                - 'nnls'     : Non-negative least squares.
                - 'trf'      : Trust region reflective algorithm.

        normal : bool, optional
            Flag for using the normal form: :math:`\mathbf{A}^{\mathsf{T}}
            \mathbf{A}\mathbf{x} = \mathbf{A}^{\mathsf{T}} \mathbf{b}`.
        mask : (tuple, list, or numpy.ndarray), optional
            Evaluate the fit only for selected spectra using either a tuple,
            list, array of integer arrays, one for each dimension, or a boolean
            array serving as a mask.
        tol : float, optional
            The iteration stops when the residual :math:`r` statisfies
            :math:`(r^k - r^{k+1})/\max\{|r^k|,|r^{k+1}|,1\} \leq \mathrm{tol}`.
        maxiter : int, optional
            The Maximum number of iterations.
        verbose : int, optional
             Controls the frequency of output according to the method.
        **kwargs : dict
            A dictionary of additional solver specific options.
        """
        if self.yData is None or not self.baseVectors:
            return

        il, iu = self.roiIndex

        a = self._lsSysMatrix[il:iu]  # matrix of base vectors
        b = self._lsTrgVector[il:iu]  # spectra to be fitted
        x = self._lsVarVector.view()  # vector of unknowns
        r = self._lsResVector.view()  # residuals

        # m, n = b.shape  # number of wavelengths or spectra, respectively
        # k, n = x.shape  # number of wavelengths or spectra, respectively

        lbnd = self._lsVarBounds[:, 0]
        ubnd = self._lsVarBounds[:, 1]

        # configure methods
        meth = method.lower()
        # Ensure normal form when using numpy.linalg.solve()
        if meth == 'gesv':
            normal = True
        # fall back to scipy.optimize.lsq_linear() if bvls not available
        if meth == 'bvls_f' and not CONFIG_OPTIONS['enableBVLS']:
            meth = 'bvls'

        # normal equations
        if normal:
            ap = np.einsum('ji,jk->ki', a, a)  # transp(A) x A
            bp = np.einsum('ji,jk', a, b)  # transp(A) x b
        else:
            ap = a.copy()
            bp = b.copy()

        # retrieve the selected spectra
        index_mask = self._ravelMask(mask)

        # linear matrix equation (using LAPACK routine _gesv).
        if meth == 'gesv': # solves linear problem trans(A)A x = trans(A)b
            if len(index_mask) < 10:
                for i in index_mask:
                    x[:, i] = np.linalg.solve(ap, bp[:, i])
            else:
                ap_inv = np.linalg.inv(ap)
                x[:, index_mask] = ap_inv @ bp[:, index_mask]

        # least square equation
        elif meth == 'lstsq':
            logger.debug("Mask {}".format(index_mask))
            x[:, index_mask], res, rank, sigma = np.linalg.lstsq(
                ap, bp[:, index_mask], rcond=None)

        # non-negative least squares (wrapper for a FORTRAN).
        # Solve argmin_x || Ax - b ||_2 for x>=0.
        elif meth == 'nnls':
            for i in index_mask:
                x[:, i], res = nnls(ap, bp[:, i])

        # trust region reflective algorithm
        elif meth == 'trf':
            for i in index_mask:
                state = lsq_linear(
                    ap, bp[:, i], bounds=(lbnd, ubnd), method='trf',
                    max_iter=maxiter, tol=tol, **kwargs)
                x[:, i] = state.x

        # bounded-variable least-squares algorithm.
        elif meth == 'bvls':
            for i in index_mask:
                state = lsq_linear(
                    ap, bp[:, i], bounds=(lbnd, ubnd), method='bvls',
                    max_iter=maxiter, tol=tol, **kwargs)

                x[:, i] = state.x

        # bounded-variable least-squares algorithm (fortran implementation).
        elif meth == 'bvls_f':
            for i in index_mask:
                x[:, i] = bvls_f(ap, bp[:, i], bounds=(lbnd, ubnd))

        # residual vector (squared Euclidean 2-norm)
        # r[:] = np.sum((a @ x - b) ** 2, axis=0)
        r[index_mask] = np.sum((a @ x[:, index_mask] - b[:, index_mask]) ** 2,
                               axis=0)


    def fitNonlinear(self, method='slsqp', normal=False, mask=None,
                     jac='3-point', tol=1e-15, gtol=1e-15, xtol=1e-15,
                     maxiter=100, maxfev=1000, verbose=0, **kwargs):
        """Non-linear constrained or unconstrained fit.

        Parameters
        ----------
        method : str
            The least square method. Should be one of

                - 'cg'       : Conjugate gradient algorithm (unconstrained).
                - 'l-bfgs-b' : Constrained BFGS algorithm.
                - 'nelder-mead' : Nelder-Mead algorithm (unconstrained).
                - 'powell'   : Powell algorithm.
                - 'slsqp'    : Sequential least squares Programming.
                - 'tnc'    : Truncated Newton (TNC) algorithm.
                - 'trust-constr' : Trust-region constrained algorithm.

        normal : bool
            Flag for using the normal form: :math:`\mathbf{A}^{\mathsf{T}}
            \mathbf{A}\mathbf{x} = \mathbf{A}^{\mathsf{T}} \mathbf{b}`.
        mask : (tuple, list, or numpy.ndarray), optional
            Evaluate the fit only for selected spectra using either a tuple,
            list, array of integer arrays, one for each dimension, or a boolean
            array serving as a mask.
        jac : {'3-point', '5-point'}, optional
        tol : float, optional
            The iteration stops when the residual :math:`r` statisfies
            :math:`(r^k - r^{k+1})/\max\{|r^k|,|r^{k+1}|,1\} \leq \mathrm{tol}`.
        gtol : float, optional
            The iteration stops when :math:`max{|\mathrm{proj}(g_i) |
            i = 1, ..., n} \leq \mathrm{gtol}` where :math:`\mathrm{proj}(g_i)`
            is the i-th component of the projected gradient.
        xtol : float, optional
            The iteration stops when the multi-dimensional variable satisfies
            :math:`(x^k - x^{k+1})/\max\{|x^k|,|x^{k+1}|,1\} \leq \mathrm{xtol}`.
        maxiter : int, optional
            The Maximum number of iterations.
        maxfev : int, optional
            The Maximum number of function evaluations.
        verbose : int, optional
             Controls the frequency of output according to the method.
        **kwargs : dict
            A dictionary of additional solver specific options.
        """
        if self.yData is None or not self.baseVectors:
            return

        il, iu = self.roiIndex

        a = self._lsSysMatrix[il:iu]  # matrix of base vectors
        b = self._lsTrgVector[il:iu]  # spectra to be fitted
        x = self._lsVarVector.view()  # vector of unknowns
        r = self._lsResVector.view()  # residuals

        m, n = b.shape  # number of wavelengths or spectra, respectively
        k, n = x.shape  # number of wavelengths or spectra, respectively

        x0 = 1. / self._lsVarScales  # normalized starting vector

        lbnd = self._lsVarBounds[:, 0]  # lower bounds
        ubnd = self._lsVarBounds[:, 1]  # upper bounds
        bounds = list(zip(lbnd, ubnd))

        # normal equations
        if normal:
            ap = np.einsum('ji,jk->ki', a, a)  # transp(A) x A
            bp = np.einsum('ji,jk', a, b)  # transp(A) x b
        else:
            ap = a.copy()
            bp = b.copy()

        # configure methods
        meth = method.lower()
        if meth == 'cg':  # conjugate gradient algorithm
            options = {'gtol': gtol, 'maxiter': maxiter, 'disp': verbose > 0,
                       **kwargs}
        elif meth == 'l-bfgs-b':  # Constrained BFGS algorithm.
            options = {'gtol': gtol, 'ftol': tol, 'maxiter': maxiter,
                       'maxfun': maxfev, 'disp': verbose > 0, **kwargs}
        elif meth == 'nelder-mead':  # Nelder-Mead algorithm.
            options = {'xatol': xtol, 'fatol': tol, 'maxiter': maxiter,
                       'maxfev': maxfev, 'disp': verbose > 0, **kwargs}
        elif meth == 'powell':  # Powell algorithm.
            options = {'xtol': xtol, 'ftol': tol, 'maxiter': maxiter,
                       'maxfev': maxfev, 'disp': verbose > 0, **kwargs}
        elif meth == 'slsqp':  # Sequential least squares Programming.
            options = {'ftol': tol, 'maxiter': maxiter,
                       'disp': verbose > 0, **kwargs}
        elif meth == 'tnc':  # Truncated Newton (TNC) algorithm.
            options = {'ftol': tol, 'gtol': gtol, 'xtol': xtol,
                       'maxiter': maxiter, 'maxfun': maxfev,
                       'disp': verbose > 0, **kwargs}
        elif meth == 'trust-constr':  # Trust-region constrained algorithm.
            options = {'xtol': xtol, 'gtol': gtol, 'maxiter': maxiter,
                       'disp': verbose > 0, 'verbose': verbose, **kwargs}
        else:
            raise ValueError("Method {} not supported.".format(method))


        # retrieve the selected spectra
        index_mask = self._ravelMask(mask)

        # rst = OptimizeResult()
        for i in index_mask:
            rst = minimize(self._func, x0, args=(ap, bp[:, i]), method=meth,
                           jac=jac, bounds=bounds, options=options)
            x[:, i] = rst.x

        # residual vector (squared Euclidean 2-norm)
        # r[:] = np.sum((a @ x - b) ** 2, axis=0)
        r[index_mask] = np.sum((a @ x[:, index_mask] - b[:, index_mask]) ** 2,
                               axis=0)


    def getResiduals(self):
        _shape = self.yData.shape
        return self._lsResVector.reshape(_shape[1:])


    def getVarVector(self, unpack=False, clip=True):
        """Get the solution vector for each spectrum.

        Parameters
        ----------
        unpack :  bool
            If true split the solution vector in a dictionary according to the
            labes of the base vectors.
        """
        if clip:
            lbnd = self._lsVarBounds[:, 0]
            ubnd = self._lsVarBounds[:, 1]
            x =  self._lsVarScales * np.clip(
                self._lsVarVector, lbnd[:, None], ubnd[:, None])
        else:
            x = self._lsVarScales * self._lsVarVector

        if unpack:
            shape = self.yData.shape[1:]
            return {key: x[i].reshape(shape) for i, key in
                    enumerate(self.baseVectors.keys())}
        else:
            k, n = self._lsVarVector.shape  # number of variables, spectra
            shape = (k,) + self.yData.shape[1:]
            return x.reshape(shape)


    def loadtxt(self, filePath, mode='bvec'):
        """Load the base vectors and spectral data from a file.

        Parameters
        ----------
        filePath : str
            The intput file path.
        """
        self.clear()
        with HSComponentFile(filePath) as file:
            vectors, spectra, wavelen = file.read()
            format = file.format

        # return empty dictionary if no wavelength information is provided
        if wavelen is None:
            return self.baseVectors

        # Set spectral test data
        logger.debug("Set spectral test data.")

        if mode in ('spec', 'all') and 'spec' in spectra:
            self.xData = wavelen
            self.yData = convert(self.format, format, spectra['spec'], wavelen)

        # set base vectors
        if mode in ('bvec', 'all'):
            logger.debug("Set loaded base vectors.")
            if self.xData is None:
                self.xData = wavelen
            for vec in vectors.values():
                vec.setFormat(self.format)  # adopt format
                vec.setInterpPnts(self.xData)
            self.baseVectors.update(vectors)

        return self.baseVectors # return dict of base vectors if no errors occur


    def model(self, format=None):
        a = self._lsSysMatrix.view()
        x = self._lsVarVector.view()
        b = np.einsum('ij,jk->ik', a, x)
        mspectra = b.reshape(self.yData.shape)

        if format is None:
            return mspectra
        else:
            if not HSFormatFlag.hasFlag(format):
                raise Exception("Unknown format '{}'.".format(format))
            return convert(format, self.format, mspectra, self.xData)


    def prepareLSProblem(self):
        """Prepare least square problem for fitting procedure"""

        if self.yData is None or not self.baseVectors:
            self._lsTrgVector = None
            self._lsVarVector = None
            self._lsResVector = None
            self._lsSysMatrix = None
            self._lsVarScales = None
            self._lsVarBounds = None

        else:

            k = len(self.baseVectors)  # number of unknown variables
            m = len(self.yData)  # number of wavelengths

            # target vector: spectral data in a reshaped 2-dimensional format
            self._lsTrgVector = self.yData.reshape(m, -1)  # (wavelen, spectrum)
            m, n = self._lsTrgVector.shape  # number of wavelengths, spectra

            # vector of unknowns for each spectrum
            self._lsVarVector = np.empty((k + 1, n))  # (variable, spectrum)

            # vector of residuals for each spectrum
            self._lsResVector = np.empty(n)  # (spectrum, )

            self._lsSysMatrix = np.empty((m, k + 1))  # (wavelen, base vector)
            self._lsVarScales = np.empty((k + 1, 1))  # (variable, )
            self._lsVarBounds = np.empty((k + 1, 2))  # (variable, bound)
            for i, vec in enumerate(self.baseVectors.values()):
                self._lsSysMatrix[:, i] = vec.getScaledValue()
                self._lsVarScales[i, :] = vec.weight * vec.scale
                self._lsVarBounds[i, :] = vec.getScaledBounds()
                logger.debug("bounds: {}".format(self._lsVarBounds[i, :]))

            # variable to correct offset
            self._lsSysMatrix[:, -1] = 1.
            self._lsVarScales[-1, :] = 1.
            self._lsVarBounds[-1, :] = [-1e9, 1e9]


    def savetxt(self, filePath, title=None, mode='bvec'):
        """Export the base vectors and spectral data.

        Parameters
        ----------
        filePath : str
            The output file path.
        title : str, optional
            A brief description of the data collection
        mode : {'bvec', 'spec', 'all'}, optional
            The data which are exported. The option 'bvec' refers to the base
            vectors, while the option 'spec' refers to the spectral data, only.
            To export both types of data, use 'all'. Default is 'bvec'.
        """
        import datetime

        if not self.baseVectors and mode in ('bvec', 'all'):
            print("No base vectors available to export.")
            return
        if self.yData is None and mode in ('spec', 'all'):
            print("No spectral available to export.")
            return

        with HSComponentFile(filePath, format=self.format, title=title) as file:
            if mode in ('bvec', 'all'):
                for vec in self.baseVectors.values():
                    file.buffer(vec)
            if mode in ('spec', 'all'):
                file.buffer(self.yData, self.xData, label="spec",
                            format=self.format)
            file.write()


    def setData(self, y, x=None, format=None):
        """Set spectral data to be fitted.

        Parameters
        ----------
        y :  numpy.ndarray
            The spectral data.
        x :  numpy.ndarray, optional
            The wavelengths at which the spectral data are sampled. If not
            defined the internally stored wavelength data in
            :attr:`~.HSImageLSAnalysis.xData` are used. If no data are
            available an error is raised.
        """
        if format is None:
            format = self.format

        if not HSFormatFlag.hasFlag(format):
            raise Exception("Unknown format '{}'.".format(format))

        if isinstance(y, list):
            y = np.array(y)
        if not isinstance(y, np.ndarray) or y.ndim < 1:
            raise Exception("Spectral y data must be ndarray of at least one "
                            "dimension.")

        # ensure two- or higher-dimensional array
        if y.ndim < 2:
            y = y[:, np.newaxis]

        if x is not None:
            if isinstance(x, list):
                x = np.array(x)
            if not isinstance(x, np.ndarray) or x.ndim > 1:
                raise Exception("Spectral x data must be 1D ndarray.")
            if len(x) != len(y):
                raise Exception("Spectral x and y data must be of same length.")

            logger.debug("setData: Set spectral data. Update wavelength.")
            self.yData = convert(self.format, format, y, x)
            self.xData = x.view(np.ndarray)

            # update lower and upper bound indices for range of interest
            self.updateROIIndex()

            # update base vectors according to the new wavelength axis
            for vec in self.baseVectors.values():
                vec.setInterpPnts(self.xData)

        else:
            if self.xData is None: # set yData without
                raise Exception("Wavelength information is not available. "
                                "Cannot update spectral y data")
            elif len(self.xData) != len(y):
                raise Exception("Spectral x and y data must be of same length.")
            else:
                logger.debug("setData: Set spectral data. Preserve wavelength.")
                self.yData = convert(self.format, format, y, self.xData)


    def setFormat(self, format):

        if not HSFormatFlag.hasFlag(format):
            raise Exception("Unknown format '{}'.".format(format))

        if format != self.format:
            if self.yData is not None:
                self.yData[:] = convert(
                    format, self.format, self.yData, self.xData)
            for vec in self.baseVectors.values():
                vec.setFormat(format)
            self.format = format


    def setROI(self, bounds=None):
        """Set the bounds for the region of interest.

        Parameters
        ----------
        bounds :  list or tuple, optional
            The lower and upper bound for the wavelength region of interest.
        """
        if bounds is None:
            self.roi = [None, None]
        elif type(bounds) in [list, tuple, np.ndarray] and len(bounds) == 2:
            self.roi = [bounds[0], bounds[1]]
        else:
            raise ValueError("Argument bounds for ROI must be list, tuple or "
                             "1D ndarray of length 2. Got {}".format(bounds))

        logger.debug("Set ROI to {}.".format(self.roi))
        self.updateROIIndex()


    def setVarBounds(self, name, bounds):
        """Set the lower and upper bound of a variable.

        Parameters
        ----------
        name :  str
            The variable name associated with a base vector
        bounds : list, tuple
            The absolute lower and upper bounds for the variable.
        """
        if name in self.baseVectors.keys():
            logger.debug(
                "Change bounds of '{}' to {}.".format(name, bounds))
            self.baseVectors[name].setBounds(bounds)
            self.prepareLSProblem()



    @property
    def shape(self):
        if self.yData is None:
            return tuple([])
        else:
            return self.yData.shape[1:]


    def updateROIIndex(self):
        """Update lower and upper bound indices for range of interest."""
        if self.xData is None:
            return

        lbnd, ubnd = self.roi
        x = self.xData.view()
        llim, ulim = x[0], x[-1]

        if lbnd is None or lbnd <= llim:
            lbndIndex = 0
        elif lbnd > ulim:
            raise ValueError("Lower ROI limit {} exceeds available range {}."
                             .format(lbnd, (llim, ulim)))
        else:
            lbndIndex = (x > lbnd).argmax() - 1# if (x > lbnd).any() else 0

        if ubnd is None or ubnd >= ulim:
            ubndIndex = len(x) - 1
        elif ubnd < llim:
            raise ValueError("Upper ROI limit {} undercut available range {}."
                             .format(ubnd, (llim, ulim)))
        else:
            ubndIndex = (x > ubnd).argmax() - 1 # if (x > ubnd).any() else 0

        self.roiIndex = [lbndIndex, ubndIndex]
        logger.debug("Update ROI Indices to {}.".format(self.roiIndex))

