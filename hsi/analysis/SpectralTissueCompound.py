# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:30:26 2020

@author: papkai
"""

import numpy as np
from scipy.optimize import minimize, OptimizeResult, nnls, lsq_linear
from pyswarm import pso
# import cvxpy as cp
from scipy.sparse import linalg

# import bvls


__all__ = ['SpectralTissueCompound']

class SpectralTissueCompound:
    
    def __init__(self, wavelen, spec, base, weight, **kwargs):

        # compound descriptor: normalized lower bound, upper bound, init value
        
        # normalize basis spectra to 1% blood, 70% water, 70% fat, and 1% melanin
        self.keys = ['blo', 'oxy', 'wat', 'fat', 'mel']
        self.compound = {} # lower bound, upper bound, initial value, lower limit, upper limit
        self.compound['blo'] = np.array([0., 2., 0.5, -0.2, 5.0])  # initially 0.5% blood
        self.compound['oxy'] = np.array([0., 1., 0.5, -0.2, 1.2])  # initially 50% cO2Hb
        self.compound['wat'] = np.array([0., 2., 0.5, -0.2, 1.2])  # initially 50% H2O
        self.compound['fat'] = np.array([0., 2., 0.7, -0.2, 1.2])  # initially 70% fat
        self.compound['mel'] = np.array([0., 6., 2.0, -0.2, 12.])  # initially  2% melanin
        self.compound['off'] = np.array([-10., 10., 0., -20, 20])  # initially 0 background

        self.wavelen = wavelen # wavelength
        
        # basis spectra
        m, n = base.shape  # number of wavelengths or basis spectra, resp
        self.base = np.hstack([base, np.ones([m, 1])])  # set of basis spectra
        
        # spectra to be fitted and unknows
        self.spec = np.array([[]])
        self.x = np.array([[]])  # vector of unknowns (fitting parameters)
        self.res = np.array([])  # vector of sum of residuals (squared Euclidean 2-norm)
        self.setSpectra(spec)
        
        # weights for the basis spectra
        self.weight = weight # parameters of the basis spectra used to rescale
        self.weight['off'] = 1.  # add background component with norm factor 1.

        # normalize basis spectra according to weights of basis for the compound
        self.baseNorm = self.base
        self.normalize()
        
        # lower and upper limit for fitting
        self.wavelenBounded = self.wavelen
        self.baseBounded = self.base
        self.specBounded = self.spec
        self.baseNormBounded = self.baseNorm
        self.lowerBound = kwargs.get('lower_bound', self.wavelen[0])
        self.upperBound = kwargs.get('upper_bound', self.wavelen[-1])

        self.setFittingRange()

        self.methods = [
            'gesv',    # linear matrix equation (unconstrained, linear)
            'lstsq',   # least square algorithm (unconstrained, linear)
            'trf',     # trust region reflective algorithm (linear)
            'bvls',    # bounded value least square algorithm (linear)
            'bvls_f',  # bounded value least square algorithm, fortran (linear)
            'slsqp',   # sequential least squares Programming (non-linear)
            'bfgs',    # constrained BFGS algorithm (non-linear)
            'cg'       # conjugate gradient algorithm (non-linear)
            ]

        

    def setSpectra(self, spec):
    
        if not isinstance(spec, np.ndarray):
            spec = np.array(spec)
        # spectrum to be fitted (ensure column format)
        
        if spec.ndim == 2:
            self.spec = spec
        elif spec.ndim == 1:
            self.spec = spec[:, np.newaxis] 
        elif spec.ndim == 0:
            self.spec = spec[np.newaxis, np.newaxis] 
        else:
            print("%s - Error: %d dimension not supported" % 
                  (self.__class__, spec.ndim))
            
        m, n = self.base.shape
        m, k = self.spec.shape  # number of wavelengths or spectra to be fitted, resp
            
        self.x = np.zeros((n+1, k))  # vector of unknowns (fitting parameters)
        self.res = np.zeros(k)  # vector of sum of residuals (squared Euclidean 2-norm)


    def normalize(self):
        """
        normalize basis spectra to initial values specified in the compound
        dictionary. Default is 1% blood, 70% water, 70% fat, and 1% melanin.
        """
        w = np.array([1. / (self.weight[key])
                      for key in self.keys])
        w = np.append(w, 1.)  # for background

        self.baseNorm = np.einsum("ij,j->ij", self.base, w)
        print(w)


    def setVariable(self, key, lower_bound, upper_bound, value):
        """
        set the start and bound values for the optimizer variables

        :param key: parameter name
        :param lower_bound: lower bound
        :param upper_bound: upper bound
        :param value: initial value
        """
        self.compound[key] = np.array([lower_bound, upper_bound, value])
        
        
    def setFittingRange(self, lower_bound=-1, upper_bound=-1):
        """
        Define lower and upper bound for the fitting algorithm
        """
        if lower_bound < self.wavelen[0] or lower_bound == -1:
            self.lowerBound = self.wavelen[0]
        else:
            self.lowerBound = lower_bound

        if upper_bound > self.wavelen[-1] or upper_bound == -1:
            self.upperBound = self.wavelen[-1]
        else:
            self.upperBound = upper_bound

        imin = np.argwhere(self.wavelen >= self.lowerBound)
        if len(imin):
            imin = imin[0, 0]
        else:
            imin = 0

        imax = np.argwhere(self.wavelen >= self.upperBound)
        if len(imax):
            imax = imax[0, 0] + 1
        else:
            imax = 0

        self.wavelenBounded = self.wavelen[imin:imax]
        self.baseBounded = self.base[imin:imax, :]
        self.baseNormBounded = self.baseNorm[imin:imax, :]

        m, k = self.spec.shape
        if imax <= m:
            self.specBounded = self.spec[imin:imax, :]    
            
      
    def functional(self, x, icol):
        """
        Residual between fitting model and data.

        The model is based on weighted basis spectra
        x0*(1-x1)*O2Hb + x0*x1*Hb + x2*H2O + x3*Fat + x4*Mel + x5

        x0 - blood concentration [%]
        x1 - percentage of oxygenated hemoglobin [%]
        x2 - percentage of water []
        x3 - percentage of fat []
        x4 - percentage of melanin [%]
        x5 - background []
        """

        # weights for blood 0% cO2Hb, blood 100% cO2Hb, water, fat, melanin
        w = np.array([x[0] * (1. - x[1]), x[0] * x[1], x[2], x[3], x[4], x[5]])

        # offset = x[5]    # account for background
        eps = self.specBounded[:, icol] - np.einsum('ij,j->i', self.baseNormBounded, w)
        
        return np.einsum('i,i->', eps, eps) / len(eps)
        # return np.sum(eps ** 2) / len(eps)


    def fit(self, method='gesv', quadratic=False, jac='3-point', verbose=False):
        if method == 'gesv':
            self.fitLinear(method)
        elif method == 'lstsq':
            self.fitLinear(method)
        elif method == 'nnls':
            self.fitLinearConstraint(method, quadratic)
        elif method == 'trf':
            self.fitLinearConstraint(method, quadratic)
        elif method == 'bvls':
            self.fitLinearConstraint(method, quadratic)
        elif method == 'bvls_f':
            self.fitLinearConstraint(method, quadratic)
        elif method == 'slsqp':
            self.fitNonLinearConstraint(method, jac, verbose)
        elif method == 'bfgs':
            self.fitNonLinearConstraint(method, jac, verbose)
        elif method == 'cg':
            self.fitNonLinearConstraint(method, jac, verbose)
        else:
            print("Error: Method %s not supported" % method)


    def fitLinear(self, method='gesv'):
        '''
        Parameter fitting using linear problem formuation
        :param method: 'gesv' or 'lstsq'
        '''
        a = self.baseNormBounded.copy()  # coefficient matrix.
        b = self.specBounded.copy()  # Ordinate or 'dependent variable' values.

        m, n = b.shape  # number of wavelengths or spectra, respectively
        aha = np.einsum('ji,jk->ki', a, a)  # transp(A) x A
        ahb = np.einsum('ji,jk', a, b)  # transp(A) x b
        x = np.zeros(ahb.shape)  # vector of matrix of unknowns
        res = np.zeros(n)

        # linear matrix equation (using LAPACK routine _gesv).
        if method == 'gesv': # solves linear problem trans(A)A x = trans(A)b
            if n < 10:
                for i in range(n):
                    x[:, i] = np.linalg.solve(aha, ahb[:, i])
            else:
                aha_inv = np.linalg.inv(aha)
                x = aha_inv @ ahb

                # x[:, i] = scipy.linalg.solve(aha, ahb[:, i], assume_a='pos')
                # x[:, i], res = scipy.sparse.linalg.cg(aha, ahb[:, i], x0=x0, tol=1e-12,
                #                                        maxiter=None, atol=None)

            res = np.sum((a @ x - b) ** 2, axis=0)# residual
            # res = np.sum((aha @ x - ahb) ** 2, axis=0)  # residual

        # least square equation
        elif method == 'lstsq':
            # x, res, rank, sigma = np.linalg.lstsq(aha, ahb, rcond=None)
            x, res, rank, sigma = np.linalg.lstsq(a, b, rcond=None)


        # transform coefficients
        # x = np.einsum('ij->ji', x)
        x[0, :] += x[1, :]
        x[1, :] /= x[0, :]

        self.x = x
        self.res = res


    def fitLinearConstraint(self, method='bvls_f', quadratic=False):
        '''
        Parameter fitting using linear problem formuation with constraints
        :param method: 'nnls', 'trf', 'bvls' or 'bvls_f'
        '''
        a = self.baseNormBounded.copy()  # coefficient matrix.
        b = self.specBounded.copy()  # Ordinate or 'dependent variable' values.

        m, n = b.shape  # number of wavelengths or spectra, respectively
        aha = np.einsum('ji,jk->ki', a, a)  # transp(A) x A
        ahb = np.einsum('ji,jk', a, b)  # transp(A) x b
        x = np.zeros(ahb.shape)  # vector of matrix of unknowns
        res = np.zeros(n)  # residual

        # use quadratic formulation
        if quadratic:
            a = aha
            b = ahb

        # constraints
        lower_bound = [val[0] for (key, val) in self.compound.items()]
        upper_bound = [val[1] for (key, val) in self.compound.items()]
        lower_bound[0] = -1.
        lower_bound[1] = -1.
        upper_bound[0] = 5.
        upper_bound[1] = 5.
        bounds = (lower_bound, upper_bound)

            
        # linear matrix equation (using LAPACK routine _gesv).
        if method == 'nnls':
            for i in range(n):
                x[:, i], res[i] = nnls(a, b[:, i])

        # trust region reflective algorithm
        elif method == 'trf':
            for i in range(n):
                state = lsq_linear(a, b[:, i], bounds=bounds,
                                   method='trf', lsmr_tol='auto')
                x[:, i] = state.x
                res[i] = state.cost

        # bounded-variable least-squares algorithm.
        elif method == 'bvls':
            for i in range(n):
                state = lsq_linear(a, b[:, i], bounds=(lower_bound, upper_bound),
                                   method='bvls', lsmr_tol='auto')
                                   # max_iter=100, tol=1e-16, lsq_solver='exact')
                x[:, i] = state.x
                res[i] = state.cost

        # bounded-variable least-squares algorithm (fortran implementation).
        elif method == 'bvls_f':
            # for i in range(n):
            #     x[:, i] = bvls.bvls(a, b[:, i], bounds=bounds)
            res = np.sum((a @ x - b) ** 2, axis=0)
            
            
        # transform coefficients
        x[0, :] += x[1, :]
        x[1, :] /= x[0, :]

        self.x = x
        self.res = res
        

    def fitNonLinearConstraint(self, method='L-BFGS-B', jac='3-point', verbose=False):
        '''
        Parameter constrained fitting by minimizing approximation error in
        least square sense
        :param method: Type of solver.
        :param jac: Method for computing the gradient vector (optional).
        :return: OptimizeResult.
        '''

        # collect variables in dictionary
        var = self.compound  # tissue compound
        x0 = [val[2] for (key, val) in var.items()] # vector of initial values
        bounds = tuple([(val[0], val[1]) # tuple of lower and upper bounds
                        for key, val in var.items()])

        # global options used for individual minimization schemes
        gl_options = {
            'epsx':1e-15,
            'epsf': 1e-15,
            'epsg': 1e-15,
            'diffstep': 1e-2,
            'maxit': 1000,
            'maxfc': 10000,
        }

        a = self.baseNormBounded.copy()  # coefficient matrix.
        b = self.specBounded.copy()  # Ordinate or 'dependent variable' values.

        m, k = a.shape  # number of wavelengths or unknowns, respectively
        m, n = b.shape  # number of wavelengths or spectra, respectively

        x = np.zeros((k, n))  # vector of matrix of unknowns
        res = np.zeros(n)  # residual
        rst = OptimizeResult()

        # Nelder-Mead algorithm
        if method == 'nelder-mead':
            options = {
                # 'func': None, # to be identified
                'xatol': gl_options['epsx'],
                'fatol': gl_options['epsf'],
                'maxfev': gl_options['maxfc'],
                'maxiter': gl_options['maxit'],
                'disp': False, # print convergence messages
                'adaptive': True # adapt algorithm parameters to dimensionality
            }
            for i in range(n):
                rst = minimize(self.functional, x0, args=(i), method='Nelder-Mead',
                               options=options)
                x[:, i] = rst.x

        # Powell algorithm
        elif method == 'powell':
            options = {
                # 'func': None, # to be identified
                'xtol': gl_options['epsx'],
                'ftol': gl_options['epsf'],
                'maxfev': gl_options['maxfc'],
                'maxiter': gl_options['maxit'],
                'disp': False, # print convergence messages
                # 'direc': [] # initial set of direction vectors
            }
            for i in range(n):
                rst = minimize(self.functional, x0, args=(i), method='Powell',
                               bounds=bounds, options=options)
                x[:, i] = rst.x

        # conjugate gradient algorithm
        elif method == 'cg':
            options = {
                'gtol': gl_options['epsg'],
                'maxiter': gl_options['maxit'],
                'disp': False, # print convergence messages
            }
            for i in range(n):
                rst = minimize(self.functional, x0, args=(i), method='CG',
                               jac=jac, options=options)
                x[:, i] = rst.x

        # truncated Newton (TNC) algorithm
        elif method == 'tnc':
            options = {
                'xtol': gl_options['epsx'],
                'ftol': gl_options['epsf'],
                'gtol': gl_options['epsg'],
                'maxfun': gl_options['maxfc'],
                'maxiter': gl_options['maxit'],
                'disp': False, # print convergence messages
            }
            for i in range(n):
                rst = minimize(self.functional, x0, args=(i), method='TNC',
                               jac=jac, bounds=bounds, options=options)
                x[:, i] = rst.x

        # Sequential Least Squares Programming
        elif method == 'slsqp':
            options = {
                'ftol': gl_options['epsf'],
                'maxiter': gl_options['maxit'],
                'disp': False, # print convergence messages
            }

            for i in range(n):
                rst = minimize(self.functional, x0, args=(i), method='SLSQP',
                               jac=jac, bounds=bounds, options=options)
                x[:, i] = rst.x


        # conjugate gradient algorithm
        elif method == 'bfgs':
            options = {
                'gtol': gl_options['epsg'],
                'ftol': gl_options['epsf'],
                'maxfun': gl_options['maxfc'],
                'maxiter': gl_options['maxit']
            }
            for i in range(n):
                rst = minimize(self.functional, x0, args=(i), method='L-BFGS-B',
                               jac=jac, bounds=bounds, options=options)
                x[:, i] = rst.x

        # trust-region constrained algorithm
        elif method == 'trust-constr':
            options = {
                'xtol': gl_options['epsx'],
                'gtol': gl_options['epsg'],
                'maxiter': gl_options['maxit']
            }
            for i in range(n):
                rst = minimize(self.functional, x0, args=(i), method='trust-constr',
                               jac=jac, bounds=bounds, options=options)

        if verbose:
            print("Converged: %s" % rst.success)
            print(rst.message)
            print("Final value x0 = " + str(rst.x))
            print("Function value f(x0) = %e" % rst.fun)
            print("Number of Iterations: %d" % rst.nit)
            print("Number of function calls: %d" % rst.nfev)

        res = np.sum((a @ x - b) ** 2, axis=0)
        self.x = x
        self.res = res


    def fitPSO(self, ):
        '''

        :param logfile:
        :return:
        '''

        # collect variables in dictionary
        var = self.compound # tissue compound
        lower_bound = [val[0] for (key, val) in var.items()]
        upper_bound = [val[1] for (key, val) in var.items()]

        options = {
            'maxiter': 500,
            'minstep': 1e-15,
            'minfunc': 1e-15,
            'swarmsize': 100,
            'debug': False
        }

        xopt, fval = pso(self.functional, lower_bound, upper_bound, **options)
        self.x = xopt

        # print('Converged: %d' % self.optim.m_state['converged'])
        print('Final value x0 = ' + str(xopt))
        print('Function value f(x0) = ' + str(fval))
        # print('Number of Iterations: %d' % self.optim.m_state['it'])
        # print('Number of function calls: %d' % self.optim.m_state['fc'])

        # self.x = self.optim.m_state['x']


    # def getParamerTransformed(self, index=-1):
    #
    #     # w = np.array([self.x[0]*(1. - self.x[1]), self.x[0]*self.x[1],
    #     #       self.x[2], self.x[3], self.x[4], self.x[5]])
    #     w = self.x.copy()
    #     w[1, :] *= w[0, :]
    #     w[0, :] -= w[1, :]
    #
    #     m, n = w.shape
    #     if index < 0 or index >= n:
    #         return w
    #     else:
    #         return w[:, index]
    #
    #     return w


    def getParamerTransformed(self, index=-1, unpack=False):

        # w = np.array([self.x[0]*(1. - self.x[1]), self.x[0]*self.x[1],
        #       self.x[2], self.x[3], self.x[4], self.x[5]])

        m, n = self.x.shape
        if index < 0 or index >= n:
            w = self.x.copy()
        else:
            w = self.x[:, index]

        w[1, :] *= w[0, :]
        w[0, :] -= w[1, :]

        if unpack:
            keys = list(self.compound.keys())
            return dict(zip(keys, w))
        else:
            return w


    # def getParameter(self, index=-1):
    #
    #     m, n = self.x.shape
    #     if index < 0 or index >= n:
    #         return self.x
    #     else:
    #         return self.x[:, index]

    def getParameter(self, index=-1, unpack=False):

        m, n = self.x.shape
        if index < 0 or index >= n:
            x = self.x
        else:
            x = self.x[:, index]

        if unpack:
            keys = list(self.compound.keys())
            return dict(zip(keys, x))
        else:
            return x

    def getParameterConstrained(self, index=-1):

        m, n = self.x.shape
        xc = self.x.copy()

        (lbnd, ubnd, val, llim, ulim) = self.compound['blo']
        idx = np.argwhere(xc[0, :] < llim)
        xc[:, idx] = -1.
        idx = np.argwhere(xc[0, :] > ulim)
        xc[:, idx] = -1.

        (lbnd, ubnd, val, llim, ulim) = self.compound['oxy']
        idx = np.argwhere(xc[1, :] < llim)
        xc[:, idx] = -1.
        idx = np.argwhere(xc[1, :] > ulim)
        xc[:, idx] = -1.

        (lbnd, ubnd, val, llim, ulim) = self.compound['wat']
        idx = np.argwhere(xc[2, :] < llim)
        xc[:, idx] = -1.
        # idx = np.argwhere(xc[2, :] > ulim)
        # xc[:, idx] = -1.

        (lbnd, ubnd, val, llim, ulim) = self.compound['fat']
        # idx = np.argwhere(xc[3, :] < llim)
        # xc[:, idx] = -1.
        idx = np.argwhere(xc[3, :] > ulim)
        xc[:, idx] = -1.


        # (lbnd, ubnd, val, llim, ulim) = self.compound['mel']
        # idx = np.argwhere(xc[4, :] < llim)

        xc[:, idx] = -1.

        # for i, (key, (lbnd, ubnd, val, llim, ulim)) in enumerate(
        #         self.compound.items()):
        #     # idx = np.argwhere(np.logical_or(xc[i, :] < llim, xc[i, :] > ulim))
        #     idx = np.argwhere(xc[i, :] < llim)
        #     print(llim, ulim)
        #     xc[:, idx] = -1.

        if index < 0 or index >= n:
            return xc
        else:
            return xc


    def getResidual(self, index=-1):

        m = len(self.res)
        if index < 0 or index >= m:
            return self.res
        else:
            return self.res[index]


    def getFittedSpectra(self):
        
        w = self.getParamerTransformed()
        specFitted = np.einsum('ij,jk->ik', self.baseNorm, w)

        return specFitted


if __name__ == "__main__":

    import os

    proj_path = os.path.join(os.getcwd(), "..", "model")
    data_path = os.path.join(os.getcwd(), "..", "data")
    pict_path = os.path.join(os.getcwd(), "..", "pictures")
        