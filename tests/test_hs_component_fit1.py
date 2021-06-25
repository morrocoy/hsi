# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:24:22 2020

@author: kpapke
"""
import sys
import os
import logging
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt

import hsi
from hsi import HSAbsorption
# from hsi import HSIntensity, HSExtinction, HSRefraction

from hsi.analysis import HSComponentFit
# from hsi.analysis import HSComponentFit2 as HSComponentFit
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


def main():
    # data_path = os.path.join(os.getcwd(), "..", "data")
    pict_path = os.path.join(os.getcwd(), "..", "pictures")

    # load spectra and base vectors ..........................................
    analysis = HSComponentFit(hsformat=HSAbsorption)
    # analysis = HSVectorAnalysis(hsformat=HSIntensity)
    # analysis = HSVectorAnalysis(hsformat=HSExtinction)
    # analysis = HSVectorAnalysis(hsformat=HSRefraction)
    analysis.loadtxt("basevectors_1.txt", mode='all')

    # alternative approach to load data into the analysis object:
    # analysis = HSVectorAnalysis(y, x)
    # analysis.add_component(y1, x1, name=name1, label=label1,
    #                      hsformat=HSAbsorption, weight=w1, bounds=bnds1)
    # analysis.add_component(y2, x2, name=name2, label=label2,
    #                      hsformat=HSAbsorption, weight=w2, bounds=bnds2)
    # ...

    # plot normalized basis spectra ..........................................
    corr = 0.408

    fig = plt.figure()
    fig.set_size_inches(6, 5)
    ax = fig.add_subplot(1, 1, 1)

    for key, vec in analysis.components.items():
        ax.plot(vec.xIntpData*1e9, vec.yIntpData, marker='s', markersize=3,
                markeredgewidth=0.3, markerfacecolor='none', markevery=5,
                label="%s (%g%%)" % (vec.label, vec.weight*100))

    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("absorption [-lg(%.3f remission)]" % corr)
    ax.legend()

    plt.savefig(os.path.join(pict_path, "BaseSpectra.png"),
                format="png", dpi=300)
    plt.show()

    # modify constraints .....................................................
    analysis.set_roi([500e-9, 1000e-9])
    # analysis.set_roi([500e-9, 600e-9])
    # analysis.set_roi([700e-9, 800e-9])
    # analysis.set_roi([800e-9, 1000e-9])

    # print base component names
    print(analysis.keys)

    # modify bounds for component weights
    # analysis.set_var_bounds("hhb", [-np.inf, np.inf])
    analysis.set_var_bounds("hhb", [0, np.inf])
    # analysis.set_var_bounds("hhb", [0, 0.05])
    analysis.set_var_bounds("ohb", [0, 0.05])
    analysis.set_var_bounds("wat", [0, 2.00])
    analysis.set_var_bounds("fat", [0, 1.00])
    analysis.set_var_bounds("mel", [0, 0.05])

    # analysis.set_var_bounds("blo", 0., 0., 0.)
    # analysis.set_var_bounds("blo", 0.5, 0.5, 0.5)
    # analysis.set_var_bounds("blo", 0.4, 0.4, 0.4)

    # analysis.set_var_bounds("oxy", 0, 0, 0)
    # analysis.set_var_bounds("oxy", 0.01, 0.01, 0.01)
    # analysis.set_var_bounds("oxy", 1, 1, 1)
    # analysis.set_var_bounds("oxy", 0.99, 0.99, 0.99)
    # analysis.set_var_bounds("oxy", 0.8, 0.8, 0.8)
    # analysis.set_var_bounds("oxy", 0.94, 0.94, 0.94)
    # analysis.set_var_bounds("oxy", 0.97, 0.97, 0.97)
    # analysis.set_var_bounds("oxy", 0.77, 0.77, 0.77)
    # analysis.set_var_bounds("oxy", 0.76, 0.78, 0.77)

    # analysis.set_var_bounds("fat", 0.0, 0.0, 0.0)
    # analysis.set_var_bounds("fat", 0.1, 0.1, 0.1)

    # analysis.set_var_bounds("wat", [0.0, 0.01])
    # analysis.set_var_bounds("wat", 0.65, 0.65, 0.65)
    # analysis.set_var_bounds("wat", 0.5, 0.5, 0.5)

    # analysis.set_var_bounds("mel", 2.3, 2.3, 2.3)
    # analysis.set_var_bounds("mel", 2.5, 2.5, 2.5)
    # analysis.set_var_bounds("mel", 2.6, 2.6, 2.6)
    # analysis.set_var_bounds("mel", 2., 2., 2.)
    # analysis.set_var_bounds("mel", 1.95, 1.95, 1.95)
    # analysis.set_var_bounds("mel", 5, 5, 5)

    # fit spectrum ...........................................................
    start = timer()

    analysis.prepare_ls_problem()

    # linear fit
    # tissue.fit(method='gesv') # linear equation (unconstrained)
    # tissue.fit(method='bvls')  # bounded value least square
    analysis.fit(method='bvls_f')  # bounded value least square (fortran)
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
    param = analysis.get_solution(unpack=True)
    param['blo'] = param['hhb'] + param['ohb']
    param['oxy'] = np.zeros(param['blo'].shape)
    idx = np.nonzero(param['blo'])
    param['oxy'][idx] = param['ohb'][idx] / param['blo'][idx]

    # plot normalized basis spectra ..........................................
    item_index = 15

    file_name = "testVectorAnalysis_Spec_%d" % item_index
    file_path = os.path.join(pict_path, file_name)

    info = ""
    for key in ['blo', 'oxy', 'wat', 'fat', 'mel']:
        info += "\n%s = %f" % (key, param[key][item_index])

    wavelen = analysis.wavelen
    spectra = analysis.spectra[:, item_index]
    spectra_fitted = analysis.model()[:, item_index]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(wavelen*1e9, spectra, label="spec %d" % item_index,
            marker='s', markersize=3, markeredgewidth=0.3,
            markerfacecolor='none', markevery=5)
    ax.plot(wavelen*1e9, spectra_fitted, label="fit",
            marker='s', markersize=3, markeredgewidth=0.3,
            markerfacecolor='none', markevery=5)

    ax.text(0.02, 0.02, info, transform=ax.transAxes, va='bottom', ha='left')
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("rel absorbtion")
    ax.legend()

    options = {
            'bbox_inches': 'tight',
            'pad_inches': 0.03,
            'dpi': 900,  # high resolution png file
        }
    # plt.savefig(filePath + ".pdf", hsformat="pdf", **options)
    plt.savefig(file_path + ".png", format="png", **options)

    plt.show()
    plt.close()


if __name__ == '__main__':
    if __name__ == '__main__':
        logmanager.setLevel(logging.DEBUG)
        logger.info("Python executable: {}".format(sys.executable))
        logger.info("Python hsi version: {}".format(hsi.__version__))

        main()
