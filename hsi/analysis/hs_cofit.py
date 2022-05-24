# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:30:26 2020

@author: kpapke
"""
import numpy
from scipy.optimize import minimize, nnls, lsq_linear

from .. import CONFIG_OPTIONS
from ..log import logmanager

from ..core.hs_formats import HSFormatFlag, HSAbsorption, convert
from ..core.hs_component import HSComponent
from ..core.hs_component_file import HSComponentFile

from .hs_base_analysis import HSBaseAnalysis


logger = logmanager.getLogger(__name__)


__all__ = ['HSCoFit']


if CONFIG_OPTIONS['enableBVLS']:
    import bvls

    def bvls_f(*args, **kwargs):
        return bvls.bvls(*args, **kwargs)
else:
    def bvls_f(*args, **kwargs):
        return None


class HSCoFit(HSBaseAnalysis):
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
    nroi : int
        The number of regions of interest.
    roi : list of float
        The lower and upper bounds for the wavelength region of interest.
    roiIndex : list of int
        The lower and upper bound indices for the wavelength region of interest.
    hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
        The hsformat for the hyperspectral data. Should be one of:

            - :class:`HSIntensity<hsi.HSIntensity>`
            - :class:`HSAbsorption<hsi.HSAbsorption>`
            - :class:`HSExtinction<hsi.HSExtinction>`
            - :class:`HSRefraction<hsi.HSRefraction>`

    keys : dict
        A dictionary of solution parameters
    """
    def __init__(self, spectra=None, wavelen=None, bounds=None,
                 hsformat=HSAbsorption):
        """ Constructor

        Parameters
        ----------
        spectra :  numpy.ndarray, optional
            The spectral data.
        wavelen :  numpy.ndarray, optional
            The wavelengths at which the spectral data are sampled.
        bounds :  list or tuple, optional
            The lower and upper bounds for the region of interest.
        hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The hsformat for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`
        """
        # dict of component vector to represent spectral data
        self.components = {}

        # list of bounds for the region of interest
        self.nroi = 1
        self.roi = [None, None]
        self.roiIndex = [None, None]

        super(HSCoFit, self).__init__(spectra, wavelen, hsformat)

        # adopt roi bounds and corresponing indices
        self.set_roi(bounds)

    @staticmethod
    def _func(x, a, b):
        # residual function to be minimized
        eps = numpy.einsum('ij,j->i', a, x) - b
        return numpy.einsum('i,i->', eps, eps) / len(eps)
        # return numpy.sum(eps ** 2) / len(eps)

    def add_component(self, y, x, name=None, label=None, hsformat=None,
                      weight=1., bounds=None):
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
        hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The output hsformat for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`


        weight :  float, optional
            The weight for the component spectrum.
        bounds :  list or tuple, optional
            The lower and upper bounds for the scaling factor.
        """
        if hsformat is None:
            hsformat = self.hsformat

        if bounds is None:
            bounds = [None, None]

        if self.wavelen is None:
            raise Exception("Spectral x data are missing for component vector.")
        else:
            self.components[name] = HSComponent(
                y, x, self.wavelen, name=name, label=label, hsformat=hsformat,
                weight=weight, bounds=bounds)
            self.keys.append(name)

    def clear(self, mode='all'):
        """ Clear all spectral information including component vectors."""
        if mode == 'all':
            super(HSCoFit, self).clear()
            self.components.clear()
            self.roiIndex = [None, None]

        elif mode == 'spec':
            self.spectra = None  # image data flatten to 2D ndarray
            self._anaTrgVector = None
            self._anaVarVector = None
            self._anaResVector = None

        elif mode == 'bvec':
            self.components.clear()
            self._anaSysMatrix = None
            self._anaVarScales = None
            self._anaVarBounds = None

    def fit(self, method='gesv', **kwargs):
        """Fit spectral data

        A least square problem is solved to Approximate spectral data by a
        weighted sum of the component vectors

        Parameters
        ----------
        method : str
            The least square method. Should be one of the method supported by
            :func:`HSImageLSAnalysis.fit_linear` or
            :func:`HSImageLSAnalysis.fitNonLinear`.
        **kwargs : dict
            Key word arguments serving as options for the fitting methods.
            Depending whether the method is linear or not, the arguments are
            forwarded to :func:`HSImageLSAnalysis.fit_linear` or
            :func:`HSImageLSAnalysis.fitNonLinear`.
        """
        meth = method.lower()
        if meth in ('gesv', 'lstsq', 'bvls', 'bvls_f', 'nnls', 'trf'):
            self.fit_linear(method, **kwargs)
        elif meth in ('cg', 'nelder-mead', 'l-bfgs-b', 'powell', 'slsqp',
                      'tnc', 'trust-constr'):
            self.fit_nonlinear(method, **kwargs)
        else:
            raise ValueError("Method {} not supported.".format(method))

    def fit_linear(self, method='gesv', normal=True, mask=None,
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

        # indices for variable and fixed solutions
        ivar = [i for i, c in enumerate(self.components.values())
                if not c.is_frozen()]
        ifix = [i for i, c in enumerate(self.components.values())
                if c.is_frozen()]
        ivar.append(len(self.components))  # keep offset variable

        # spectra to be fitted
        if len(ifix):
            a = self._anaSysMatrix[il:iu, ivar]
            b = self._anaTrgVector[il:iu] - \
                self._anaSysMatrix[il:iu, ifix] @ self._anaVarVector[ifix, :]
            x = self._anaVarVector[ivar, :]  # vector of unknowns

        else:
            a = self._anaSysMatrix[il:iu]  # matrix of component vectors
            b = self._anaTrgVector[il:iu]  # spectra to be fitted
            x = self._anaVarVector.view()  # vector of unknowns

        r = self._anaResVector.view()  # residuals

        # m, n = b.shape  # number of wavelengths or spectra, respectively
        # k, n = x.shape  # number of wavelengths or spectra, respectively

        lbnd = self._anaVarBounds[ivar, 0]
        ubnd = self._anaVarBounds[ivar, 1]

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
            ap = numpy.einsum('ji,jk->ki', a, a)  # transp(A) x A
            bp = numpy.einsum('ji,jk', a, b)  # transp(A) x b
        else:
            ap = a.copy()
            bp = b.copy()

        # retrieve the selected spectra
        index_mask = self._ravel_mask(mask)

        # linear matrix equation (using LAPACK routine _gesv).
        if meth == 'gesv':  # solves linear problem trans(A)A x = trans(A)b
            if len(index_mask) < 10:
                for i in index_mask:
                    x[:, i] = numpy.linalg.solve(ap, bp[:, i])
            else:
                ap_inv = numpy.linalg.inv(ap)
                x[:, index_mask] = ap_inv @ bp[:, index_mask]

        # least square equation
        elif meth == 'lstsq':
            logger.debug("Mask {}".format(index_mask))
            x[:, index_mask], res, rank, sigma = numpy.linalg.lstsq(
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
            # bounds = [_lbnd, _ubnd]
            # print(bounds)
            _lbnd[numpy.isnan(_lbnd)] = -numpy.inf
            _ubnd[numpy.isnan(_ubnd)] = numpy.inf
            bounds = [_lbnd, _ubnd]
            # print(bounds)
            for i in index_mask:
                x[:, i] = bvls_f(ap, bp[:, i], bounds=bounds)

        # store solution vector
        self._anaVarVector[ivar, :] = x

        # store residual vector (squared Euclidean 2-norm)
        # r[:] = numpy.sum((a @ x - b) ** 2, axis=0)
        r[index_mask] = numpy.sum(
            (a @ x[:, index_mask] - b[:, index_mask]) ** 2, axis=0)

    def fit_nonlinear(self, method='slsqp', normal=False, mask=None,
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
            :math:`(x^k-x^{k+1})/\max\{|x^k|,|x^{k+1}|,1\} \leq \mathrm{xtol}`.
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

        # indices for variable and fixed solutions
        ivar = [i for i, c in enumerate(self.components.values())
                if not c.is_frozen()]
        ifix = [i for i, c in enumerate(self.components.values())
                if c.is_frozen()]
        ivar.append(len(self.components))  # keep offset variable

        # spectra to be fitted
        if len(ifix):
            a = self._anaSysMatrix[il:iu, ivar]
            b = self._anaTrgVector[il:iu] - \
                self._anaSysMatrix[il:iu, ifix] @ self._anaVarVector[ifix, :]
            x = self._anaVarVector[ivar, :]  # vector of unknowns
            x0 = 1. / self._anaVarScales[ivar]  # normalized starting vector

        else:
            a = self._anaSysMatrix[il:iu]  # matrix of component vectors
            b = self._anaTrgVector[il:iu]  # spectra to be fitted
            x = self._anaVarVector.view()  # vector of unknowns
            x0 = 1. / self._anaVarScales  # normalized starting vector
        r = self._anaResVector.view()  # residuals

        # m, n = b.shape  # number of wavelengths or spectra, respectively
        # k, n = x.shape  # number of wavelengths or spectra, respectively

        lbnd = self._anaVarBounds[ivar, 0]  # lower bounds
        ubnd = self._anaVarBounds[ivar, 1]  # upper bounds
        bounds = list(zip(lbnd, ubnd))

        # normal equations
        if normal:
            ap = numpy.einsum('ji,jk->ki', a, a)  # transp(A) x A
            bp = numpy.einsum('ji,jk', a, b)  # transp(A) x b
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
        index_mask = self._ravel_mask(mask)

        # rst = OptimizeResult()
        for i in index_mask:
            rst = minimize(self._func, x0, args=(ap, bp[:, i]), method=meth,
                           jac=jac, bounds=bounds, options=options)
            x[:, i] = rst.x

        # store solution vector
        self._anaVarVector[ivar, :] = x
        ibuf = self._buffer_index
        self._anaVarBuffer[ibuf, ...] = self._anaVarVector.copy()
        ibuf = (ibuf + 1) % self._buffer_maxcount

        # store residual vector (squared Euclidean 2-norm)
        # r[:] = numpy.sum((a @ x - b) ** 2, axis=0)
        r[index_mask] = numpy.sum(
            (a @ x[:, index_mask] - b[:, index_mask]) ** 2, axis=0)

    def freeze_component(self, name):
        """Freeze the solution of a variable for a component.

        Parameters
        ----------
        name :  str
            The component's name
        """
        if name not in self.components.keys():
            logger.debug(
                "Could not freeze solution for variable '{}'. "
                "Variable not found.".format(name))
            return

        self.components[name].freeze()
        logger.debug("Freeze solution for variable '{}'.".format(name))

    def loadtxt(self, file_path, mode='bvec'):
        """Load the component vectors and spectral data from a file.

        Parameters
        ----------
        file_path : str
            The intput file path.
        mode : {"bvec", "spec", "all"}
            Specify which kind of spectra to read.
        """
        self.clear(mode)
        with HSComponentFile(file_path) as file:
            vectors, spectra, wavelen = file.read()
            hsformat = file.hsformat

        # return empty dictionary if no wavelength information is provided
        if wavelen is None:
            return self.components

        # Set spectral test data
        logger.debug("Set spectral test data.")

        if mode in ('spec', 'all') and 'spec' in spectra:
            self.wavelen = wavelen
            self.spectra = convert(
                self.hsformat, hsformat, spectra['spec'], wavelen)

        # set component vectors
        if mode in ('bvec', 'all'):
            logger.debug("Set loaded component vectors.")
            if self.wavelen is None:
                self.wavelen = wavelen
            for vec in vectors.values():
                vec.set_format(self.hsformat)  # adopt hsformat
                vec.set_interp_points(self.wavelen)
            self.components.update(vectors)
            self.keys = [key for key in vectors.keys()]

        return self.components  # return dict of component vectors if no errors

    def model(self, hsformat=None):
        a = self._anaSysMatrix.view()
        x = self._anaVarVector.view()
        b = numpy.einsum('ij,jk->ik', a, x)
        mspectra = b.reshape(self.spectra.shape)

        if hsformat is None:
            return mspectra
        else:
            if not HSFormatFlag.has_flag(hsformat):
                raise Exception("Unknown hsformat '{}'.".format(hsformat))
            return convert(hsformat, self.hsformat, mspectra, self.wavelen)

    def prepare_ls_problem(self):
        """Prepare least square problem for fitting procedure"""

        if self.spectra is None or not self.components:
            self._anaTrgVector = None
            self._anaVarVector = None
            self._anaResVector = None
            self._anaSysMatrix = None
            self._anaVarScales = None
            self._anaVarBounds = None
            self._anaVarBuffer = None

        else:
            k = len(self.components)  # number of unknown variables
            m = len(self.spectra)  # number of wavelengths

            # target vector: spectral data in a reshaped 2-dimensional hsformat
            # (wavelen, spectrum)
            self._anaTrgVector = self.spectra.reshape(m, -1)
            # number of wavelengths, spectra
            m, n = self._anaTrgVector.shape

            # vector of unknowns for each spectrum
            self._anaVarVector = numpy.zeros((k + 1, n))  # (variable, spectrum)
            self._anaVarBuffer = numpy.zeros((self._buffer_maxcount, k + 1, n))

            # vector of residuals for each spectrum
            self._anaResVector = numpy.zeros(n)  # (spectrum, )

            # (wavelen, component vector)
            self._anaSysMatrix = numpy.zeros((m, k + 1))
            # (variable, )
            self._anaVarScales = numpy.zeros((k + 1, 1))
            # (variable, bound)
            self._anaVarBounds = numpy.zeros((k + 1, 2))

            for i, vec in enumerate(self.components.values()):
                # print(f"{vec.name} {vec.bounds} {vec.get_scaled_bounds()}")
                self._anaSysMatrix[:, i] = vec.get_scaled_value()
                self._anaVarScales[i, :] = vec.weight * vec.scale
                self._anaVarBounds[i, :] = vec.get_scaled_bounds()
                logger.debug("bounds: {}".format(self._anaVarBounds[i, :]))

            # variable to correct offset
            self._anaSysMatrix[:, -1] = 1.
            self._anaVarScales[-1, :] = 1.
            self._anaVarBounds[-1, :] = [-1e9, 1e9]

    def remove_component(self, name):
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

    def savetxt(self, file_path, title=None, mode='bvec'):
        """Export the component vectors and spectral data.

        Parameters
        ----------
        file_path : str
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

        with HSComponentFile(
                file_path, hsformat=self.hsformat, title=title) as file:
            if mode in ('bvec', 'all'):
                for vec in self.components.values():
                    file.buffer(vec)
            if mode in ('spec', 'all'):
                file.buffer(self.spectra, self.wavelen, label="spec",
                            format=self.hsformat)
            file.write()

    def set_data(self, y, x=None, hsformat=None):
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
        hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The output hsformat for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`


        """
        super(HSCoFit, self).set_data(y, x, hsformat)
        if x is not None:
            # update lower and upper bound indices for range of interest
            self.update_roi_index()
            # update component vectors according to the new wavelength axis
            for vec in self.components.values():
                vec.set_interp_points(self.wavelen)

    def set_format(self, hsformat):
        """Set the hsformat for the hyperspectral data.

        Parameters
        ----------
        hsformat :  :obj:`HSFormatFlag<hsi.HSFormatFlag>`, optional
            The hsformat for the hyperspectral data. Should be one of:

                - :class:`HSIntensity<hsi.HSIntensity>`
                - :class:`HSAbsorption<hsi.HSAbsorption>`
                - :class:`HSExtinction<hsi.HSExtinction>`
                - :class:`HSRefraction<hsi.HSRefraction>`

        """
        super(HSCoFit, self).set_format(hsformat)
        for vec in self.components.values():
            vec.set_format(self.hsformat)

    def set_roi(self, bounds=None):
        """Set the bounds for the region of interest.

        Parameters
        ----------
        bounds :  list or tuple, optional
            The lower and upper bound for the wavelength region of interest.
        """
        if bounds is None:
            self.roi = [None, None]
        elif type(bounds) in [list, tuple, numpy.ndarray] and len(bounds) == 2:
            self.roi = [bounds[0], bounds[1]]
        else:
            raise ValueError("Argument bounds for ROI must be list, tuple or "
                             "1D ndarray of length 2. Got {}".format(bounds))

        logger.debug("Set ROI to {}.".format(self.roi))
        self.update_roi_index()

    def set_var_bounds(self, name, bounds):
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
            self.components[name].set_bounds(bounds)

            if self._anaVarBounds is not None:
                for i, vec in enumerate(self.components.values()):
                    self._anaVarBounds[i, :] = vec.get_scaled_bounds()

    def unfreeze_component(self, name):
        """Unfreeze the solution of a variable for a component.

        Parameters
        ----------
        name :  str
            The component's name
        """
        if name not in self.components.keys():
            logger.debug(
                "Could not release solution for variable '{}'. "
                "Variable not found.".format(name))
            return

        self.components[name].unfreeze()
        logger.debug("Release solution for variable '{}'.".format(name))

    def update_roi_index(self):
        """Update lower and upper bound indices for range of interest."""
        if self.wavelen is None:
            return

        lbnd, ubnd = self.roi
        x = self.wavelen.view()
        llim, ulim = x[0], x[-1]

        if lbnd is None or lbnd <= llim:
            lbnd_index = 0
        elif lbnd > ulim:
            raise ValueError("Lower ROI limit {} exceeds available range {}."
                             .format(lbnd, (llim, ulim)))
        else:
            lbnd_index = (x > lbnd).argmax() - 1  # if (x > lbnd).any() else 0

        if ubnd is None or ubnd >= ulim:
            ubnd_index = len(x) - 1
        elif ubnd < llim:
            raise ValueError("Upper ROI limit {} undercut available range {}."
                             .format(ubnd, (llim, ulim)))
        else:
            ubnd_index = (x > ubnd).argmax() - 1  # if (x > ubnd).any() else 0

        self.roiIndex = [lbnd_index, ubnd_index]
        logger.debug("Update ROI Indices to {}.".format(self.roiIndex))
