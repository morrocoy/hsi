# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:24:22 2020

@author: kpapke
"""
import os
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt

from hsi import HSAbsorption, HSIntensity, HSExtinction, HSRefraction
from hsi.analysis import HSComponentFit
# from hsi.analysis import HSComponentFit2 as HSComponentFit

import logging

LOGGING = True
logger = logging.getLogger(__name__)
logger.propagate = LOGGING


def main():

    data_path = os.path.join(os.getcwd(), "..", "data")
    pict_path = os.path.join(os.getcwd(), "..", "pictures")

    # load spectra and base vectors ..........................................

    tissue = HSComponentFit(format=HSAbsorption)
    # tissue = HSVectorAnalysis(format=HSIntensity)
    # tissue = HSVectorAnalysis(format=HSExtinction)
    # tissue = HSVectorAnalysis(format=HSRefraction)
    tissue.loadtxt("basevectors_1.txt", mode='all')

    # alternative approach to load data into the analysis object:
    # tissue = HSVectorAnalysis(y, x)
    # tissue.addBaseVector(y1, x1, name=name1, label=label1,
    #                      format=HSAbsorption, weight=w1, bounds=bnds1)
    # tissue.addBaseVector(y2, x2, name=name2, label=label2,
    #                      format=HSAbsorption, weight=w2, bounds=bnds2)
    # ...


    # plot normalized basis spectra ..........................................
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for key, vec in tissue.baseVectors.items():
        ax.plot(vec.xIntpData*1e9, vec.yIntpData, label=vec.label,
                marker='s', markersize=3, markeredgewidth=0.3,
                markerfacecolor='none', markevery=5)

    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("rel absorbtion")
    ax.legend()
    plt.show()


    # modify constraints .....................................................
    tissue.setROI([500e-9, 1000e-9])

    # tissue.setROI([500e-9, 600e-9])
    # tissue.setROI([700e-9, 800e-9])
    # tissue.setROI([800e-9, 1000e-9])

    # tissue.setVarBounds("blo", 0., 0., 0.)
    # tissue.setVarBounds("blo", 0.5, 0.5, 0.5)
    # tissue.setVarBounds("blo", 0.4, 0.4, 0.4)

    # tissue.setVarBounds("oxy", 0, 0, 0)
    # tissue.setVarBounds("oxy", 0.01, 0.01, 0.01)
    # tissue.setVarBounds("oxy", 1, 1, 1)
    # tissue.setVarBounds("oxy", 0.99, 0.99, 0.99)
    # tissue.setVarBounds("oxy", 0.8, 0.8, 0.8)
    # tissue.setVarBounds("oxy", 0.94, 0.94, 0.94)
    # tissue.setVarBounds("oxy", 0.97, 0.97, 0.97)
    # tissue.setVarBounds("oxy", 0.77, 0.77, 0.77)
    # tissue.setVarBounds("oxy", 0.76, 0.78, 0.77)

    # tissue.setVarBounds("fat", 0.0, 0.0, 0.0)
    # tissue.setVarBounds("fat", 0.1, 0.1, 0.1)

    # tissue.setVarBounds("wat", [0.0, 0.01])
    # tissue.setVarBounds("wat", 0.65, 0.65, 0.65)
    # tissue.setVarBounds("wat", 0.5, 0.5, 0.5)

    # tissue.setVarBounds("mel", 2.3, 2.3, 2.3)
    # tissue.setVarBounds("mel", 2.5, 2.5, 2.5)
    # tissue.setVarBounds("mel", 2.6, 2.6, 2.6)
    # tissue.setVarBounds("mel", 2., 2., 2.)
    # tissue.setVarBounds("mel", 1.95, 1.95, 1.95)
    # tissue.setVarBounds("mel", 5, 5, 5)


    # fit spectrum ...........................................................
    start = timer()

    tissue.prepareLSProblem()

    # linear fit
    # tissue.fit(method='gesv') # linear equation (unconstrained)
    # tissue.fit(method='bvls')  # bounded value least square
    tissue.fit(method='bvls_f')  # bounded value least square (fortran)
    # tissue.fit(method='CG')  # conjugate gradient algorithm (unconstrained)

    # non-linear
    # tissue.fit(method='l-bfgs-b')  # constrained BFGS algorithm (fast)
    # tissue.fit(method='slsqp')  # Sequential Least Squares Programming (fast)

    # tissue.fit(method='Nelder-Mead')  # Nelder-Mead algorithm (unconstrained)
    # tissue.fit(method='Powell')  # Powell algorithm (not suitable)
    # tissue.fit(method='TNC')  # truncated Newton (TNC) algorithm
    # tissue.fit(method='trust-constr')  # trust-region constrained algorithm

    print("Elapsed time: %f sec" % (timer() - start))

    # param = tissue.getVarVector(unpack=True)
    param = tissue.getSolution(unpack=True)
    param['blo'] = param['hhb'] + param['ohb']
    param['oxy'] = np.zeros(param['blo'].shape)
    idx = np.nonzero(param['blo'])
    param['oxy'][idx] = param['ohb'][idx] / param['blo'][idx]


    # plot normalized basis spectra ..........................................
    tissIdx = 15

    fileName = "testVectorAnalysis_Spec_%d" % tissIdx
    filePath = os.path.join(pict_path, fileName)

    info = ""
    for key in ['blo', 'oxy', 'wat', 'fat', 'mel']:
        info += "\n%s = %f" % (key, param[key][tissIdx])

    wavelen = tissue.wavelen
    spec = tissue.spectra[:, tissIdx]
    specFit = tissue.model()[:, tissIdx]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(wavelen*1e9, spec, label="spec %d" % tissIdx,
                marker='s', markersize=3, markeredgewidth=0.3,
                markerfacecolor='none', markevery=5)
    ax.plot(wavelen*1e9, specFit, label="fit",
                marker='s', markersize=3, markeredgewidth=0.3,
                markerfacecolor='none', markevery=5)

    ax.text(0.02, 0.02, info, transform=ax.transAxes, va='bottom', ha='left')
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("rel absorbtion")
    ax.legend()

    options = {
            'bbox_inches': 'tight',
            'pad_inches': 0.03,
            'dpi': 900, # high resolution png file
        }
    # plt.savefig(filePath + ".pdf", format="pdf", **options)
    plt.savefig(filePath + ".png", format="png", **options)

    plt.show()
    plt.close()



if __name__ == '__main__':

    # fmt = "%(asctime)s %(filename)35s: %(lineno)-4d: %(funcName)20s(): " \
    #       "%(levelname)-7s: %(message)s"
    # logging.basicConfig(level='DEBUG', format=fmt)

    requests_logger = logging.getLogger('hsi')
    requests_logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
            "%(asctime)s %(filename)35s: %(lineno)-4d: %(funcName)20s(): " \
              "%(levelname)-7s: %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    requests_logger.addHandler(handler)

    main()
