# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:30:26 2020

@author: kpapke
"""
import numpy as np
from scipy.optimize import minimize, nnls, lsq_linear

from .. import CONFIG_OPTIONS
from ..log import logmanager

from ..core.formats import HSFormatFlag, HSAbsorption, convert
from ..core.HSComponent import HSComponent
from ..core.HSComponentFile import HSComponentFile

from .HSBaseAnalysis import HSBaseAnalysis



logger = logmanager.getLogger(__name__)


__all__ = ['HSComponentFit']


if CONFIG_OPTIONS['enableBVLS']:
    import bvls
    def bvls_f(*args, **kwargs):
        return bvls.bvls(*args, **kwargs)
else:
    def bvls_f(*args, **kwargs):
        return None


class HSComponentFit(HSBaseAnalysis):
    """
    Class to approximate hyper spectral image data by a weighted sum of
    component spectra in order to analysize their individual contributions.

    Features:

    - load component spectra configuration
    - linear, nonlinear, constrained and unconstrained approximations

    Attributes
    ----------
    wavelen :  numpy.ndarray
        The wavelengths at which the spectral data are sampled.
    spectra :  numpy.ndarray
        The spectral data.
    components : dict of HSComponent
        A dictionary of component vector to represent the spectral data.
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

    keys : dict
        A dictionary of solution parameters
    """

    def __init__(self, spectra=None, wavelen=None, bounds=None,
                 format=HSAbsorption):
        """ Constructor

        Parameters
        ----------
        spectra :  numpy.ndarray, optional
            The spectral data.
        wavelen :  numpy.ndarray, optional
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
        super(HSComponentFit, self).__init__(spectra, wavelen, format)

        # dict of component vector to represent spectral data
        self.components = {}

        # list of bounds for the region of interest
        self.roi = [None, None]
        self.roiIndex = [None, None]

        # adopt roi bounds and corresponing indices
        self.setROI(bounds)


    def _func(self, x, a, b):
        # residual function to be minimized
        eps = np.einsum('ij,j->i', a, x) - b
        return np.einsum('i,i->', eps, eps) / len(eps)
        # return np.sum(eps ** 2) / len(eps)


    def addBaseVector(self, y, x, name=None, label=None, format=None,
                      weight=1., bounds=[None, None]):
        """Add component vector to represent spectral data

        Forwards all arguments to :class:`hsi.analysis.HSBaseVector`.
        The wavelengths at which the spectral information will be interpolated
        are taken from the internally stored array
        :attr:`~.HSImageLSAnalysis.xData` that is common to all component
        vectors.

        Parameters
        ----------
        y :  numpy.ndarray
            The component spectral data.
        x :  numpy.ndarray
            The wavelengths at which the component spectral data are sampled.
        name :  str, optional
            Name of the component spectrum.
        label :  str, optional
            Label of the component spectrum.
        weight :  float, optional
            The weight for the component spectrum.
        bounds :  list or tuple, optional
            The lower and upper bounds for the scaling factor.
        """
        if format is None:
            format = self.format

        if self.wavelen is None:
            raise Exception("Spectral x data are missing for component vector.")
        else:
            self.components[name] = HSComponent(
                y, x, self.wavelen, name=name, label=label, format=format,
                weight=weight, bounds=bounds)
            self.keys.append(name)


    def clear(self):
        """ Clear all spectral information including component vectors."""
        super(HSComponentFit, self).clear()
        self.roiIndex = [None, None]
        self.components.clear()


    def fit(self, method='gesv', **kwargs):
        """Fit spectral data

        A least square problem is solved to Approximate spectral data by a
        weighted sum of the component vectors

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
        if self.spectra is None or not self.components:
            return

        il, iu = self.roiIndex

        a = self._anaSysMatrix[il:iu]  # matrix of component vectors
        b = self._anaTrgVector[il:iu]  # spectra to be fitted
        x = self._anaVarVector.view()  # vector of unknowns
        r = self._anaResVector.view()  # residuals

        # m, n = b.shape  # number of wavelengths or spectra, respectively
        # k, n = x.shape  # number of wavelengths or spectra, respectively

        lbnd = self._anaVarBounds[:, 0]
        ubnd = self._anaVarBounds[:, 1]

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
            _lbnd = lbnd.copy()
            _ubnd = ubnd.copy()
            bounds = [_lbnd, _ubnd]
            # print(bounds)
            _lbnd[np.isnan(_lbnd)] = -np.inf
            _ubnd[np.isnan(_ubnd)] = np.inf
            bounds = [_lbnd, _ubnd]
            # print(bounds)
            for i in index_mask:
                x[:, i] = bvls_f(ap, bp[:, i], bounds=bounds)

            # print(x)
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
        if self.spectra is None or not self.components:
            return

        il, iu = self.roiIndex

        a = self._anaSysMatrix[il:iu]  # matrix of component vectors
        b = self._anaTrgVector[il:iu]  # spectra to be fitted
        x = self._anaVarVector.view()  # vector of unknowns
        r = self._anaResVector.view()  # residuals

        m, n = b.shape  # number of wavelengths or spectra, respectively
        k, n = x.shape  # number of wavelengths or spectra, respectively

        x0 = 1. / self._anaVarScales  # normalized starting vector

        lbnd = self._anaVarBounds[:, 0]  # lower bounds
        ubnd = self._anaVarBounds[:, 1]  # upper bounds
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


    def loadtxt(self, filePath, mode='bvec'):
        """Load the component vectors and spectral data from a file.

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
            return self.components

        # Set spectral test data
        logger.debug("Set spectral test data.")

        if mode in ('spec', 'all') and 'spec' in spectra:
            self.wavelen = wavelen
            self.spectra = convert(self.format, format, spectra['spec'], wavelen)

        # set component vectors
        if mode in ('bvec', 'all'):
            logger.debug("Set loaded component vectors.")
            if self.wavelen is None:
                self.wavelen = wavelen
            for vec in vectors.values():
                vec.setFormat(self.format)  # adopt format
                vec.set_interp_points(self.wavelen)
            self.components.update(vectors)
            self.keys = [key for key in vectors.keys()]

        return self.components # return dict of component vectors if no errors


    def model(self, format=None):
        a = self._anaSysMatrix.view()
        x = self._anaVarVector.view()
        b = np.einsum('ij,jk->ik', a, x)
        mspectra = b.reshape(self.spectra.shape)

        if format is None:
            return mspectra
        else:
            if not HSFormatFlag.hasFlag(format):
                raise Exception("Unknown format '{}'.".format(format))
            return convert(format, self.format, mspectra, self.wavelen)


    def prepareLSProblem(self):
        """Prepare least square problem for fitting procedure"""

        if self.spectra is None or not self.components:
            self._anaTrgVector = None
            self._anaVarVector = None
            self._anaResVector = None
            self._anaSysMatrix = None
            self._anaVarScales = None
            self._anaVarBounds = None

        else:

            k = len(self.components)  # number of unknown variables
            m = len(self.spectra)  # number of wavelengths

            # target vector: spectral data in a reshaped 2-dimensional format
            # (wavelen, spectrum)
            self._anaTrgVector = self.spectra.reshape(m, -1)
            # number of wavelengths, spectra
            m, n = self._anaTrgVector.shape

            # vector of unknowns for each spectrum
            self._anaVarVector = np.zeros((k + 1, n))  # (variable, spectrum)

            # vector of residuals for each spectrum
            self._anaResVector = np.zeros(n)  # (spectrum, )

            # (wavelen, component vector)
            self._anaSysMatrix = np.zeros((m, k + 1))
            # (variable, )
            self._anaVarScales = np.zeros((k + 1, 1))
            # (variable, bound)
            self._anaVarBounds = np.zeros((k + 1, 2))

            for i, vec in enumerate(self.components.values()):
                # print(f"{vec.name} {vec.bounds} {vec.getScaledBounds()}")
                self._anaSysMatrix[:, i] = vec.getScaledValue()
                self._anaVarScales[i, :] = vec.weight * vec.scale
                self._anaVarBounds[i, :] = vec.getScaledBounds()
                logger.debug("bounds: {}".format(self._anaVarBounds[i, :]))

            # variable to correct offset
            self._anaSysMatrix[:, -1] = 1.
            self._anaVarScales[-1, :] = 1.
            self._anaVarBounds[-1, :] = [-1e9, 1e9]


    def removeComponent(self, name):
        """Remove a component vector.

        Parameters
        ----------
        name :  str
            The variable name associated with a component vector
        """
        if name in self.components.keys():
            logger.debug(
                "Remove component '{}'.".format(name))
            self.components.pop(name)
            self.keys.remove(name)

            self.prepareLSProblem()


    def savetxt(self, filePath, title=None, mode='bvec'):
        """Export the component vectors and spectral data.

        Parameters
        ----------
        filePath : str
            The output file path.
        title : str, optional
            A brief description of the data collection
        mode : {'bvec', 'spec', 'all'}, optional
            The data which are exported. The option 'bvec' refers to the
            component vectors, while the option 'spec' refers to the spectral
            data. To export both types of data, use 'all'. Default is 'bvec'.
        """

        if not self.components and mode in ('bvec', 'all'):
            print("No component vectors available to export.")
            return
        if self.spectra is None and mode in ('spec', 'all'):
            print("No spectral available to export.")
            return

        with HSComponentFile(filePath, format=self.format, title=title) as file:
            if mode in ('bvec', 'all'):
                for vec in self.components.values():
                    file.buffer(vec)
            if mode in ('spec', 'all'):
                file.buffer(self.spectra, self.wavelen, label="spec",
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
        super(HSComponentFit, self).setData(y, x, format)
        if x is not None:
            # update lower and upper bound indices for range of interest
            self.updateROIIndex()
            # update component vectors according to the new wavelength axis
            for vec in self.components.values():
                vec.set_interp_points(self.wavelen)


    def setFormat(self, format):
        """Set the format for the hyperspectral data.

        Parameters
        ----------
        format :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The format for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`

        """
        super(HSComponentFit, self).setFormat(format)
        for vec in self.components.values():
            vec.setFormat(self.format)


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
            The variable name associated with a component vector
        bounds : list, tuple
            The absolute lower and upper bounds for the variable.
        """
        if name in self.components.keys():
            logger.debug(
                "Change bounds of '{}' to {}.".format(name, bounds))
            self.components[name].setBounds(bounds)
            self.prepareLSProblem()


    def updateROIIndex(self):
        """Update lower and upper bound indices for range of interest."""
        if self.wavelen is None:
            return

        lbnd, ubnd = self.roi
        x = self.wavelen.view()
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

