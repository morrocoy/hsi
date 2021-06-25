# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:34:02 2021

@author: kpapke
"""
import sys
import os.path
import logging
from timeit import default_timer as timer
from scipy import signal, ndimage

import numpy as np
import matplotlib.pyplot as plt

import hsi
from hsi import HSAbsorption, HSIntensity, HSExtinction, HSRefraction
from hsi import HSComponentFile
from hsi.analysis import HSOpenTivita as HSTivita
# from hsi.analysis import HSTivita as HSTivita
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


def main():

    data_path = os.path.join(os.getcwd(), "..", "..", "data")
    pict_path = os.path.join(os.getcwd(), "..", "..", "pictures")

    # load spectra from file .................................................
    with HSComponentFile("basevectors_1.txt") as file:
        vectors, spectra, wavelen = file.read()
        format = file.hsformat

    spectra = spectra['spec']
    print("Spectral hsformat: %s" % (format.key))
    tissue = HSTivita(spectra, wavelen, hsformat=format)

    # tissue = HSTivitaAnalysis(hsformat=HSIntensity)
    # tissue = HSTivitaAnalysis(hsformat=HSExtinction)
    # tissue = HSTivitaAnalysis(hsformat=HSRefraction)

    # verify oxixenation
    ddspectra = signal.savgol_filter(
        spectra, window_length=5, polyorder=3, deriv=2, axis=0)

    reg0 = [570e-9, 590e-9]
    reg1 = [740e-9, 780e-9]

    idx0 = np.where((wavelen >= reg0[0]) * (wavelen <= reg0[1]))[0]
    idx1 = np.where((wavelen >= reg1[0]) * (wavelen <= reg1[1]))[0]

    val0 = np.min(spectra[idx0], axis=0)
    val1 = np.min(spectra[idx1], axis=0)

    r0 = 1.
    r1 = 1.

    ratios = val0 / val1
    res =  val0 / r0 / (val0 / r0 + val1 / r1)

    tissIdx = 0

    calIndex = [0, 15, 1]
    calValues = np.array([0.01, 0.8, 0.99])
    measuredRatios = ratios[calIndex]
    scaledRatios = measuredRatios * (1 - calValues) / calValues

    print("calibration values: {}".format(calValues))
    print("measured ratios: {}".format(measuredRatios))
    print("scaled ratios: {}".format(scaledRatios))
    print("results: {}".format(res[calIndex]))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # ax.plot(wavelen * 1e9, spectra[:, tissIdx], label="f of set %s" % tissIdx,
    #         marker='s', markersize=3, markeredgewidth=0.3,
    #         markerfacecolor='none', markevery=5)
    ax.plot(wavelen * 1e9, spectra[:, 0],
            # label="f\'\' of HHb",
            label="f of HHb",
            marker='s', markersize=3, markeredgewidth=0.3,
            markerfacecolor='none', markevery=1)
    ax.plot(wavelen * 1e9, spectra[:, 1],
            # label="f\'\' of O2Hb",
            label="f of O2Hb",
            marker='s', markersize=3, markeredgewidth=0.3,
            markerfacecolor='none', markevery=1)

    # ax.set_xlim([700, 800])
    # ax.set_ylim([-0.005, 0.005])
    # ax.text(0.02, 0.02, info, transform=ax.transAxes, va='bottom', ha='left')
    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("rel absorbtion")
    ax.legend()

    options = {
        'bbox_inches': 'tight',
        'pad_inches': 0.03,
        'dpi': 900,  # high resolution png file
    }
    # plt.savefig(filePath + ".pdf", hsformat="pdf", **options)
    plt.savefig(os.path.join(pict_path, "HSTivitaAnalysis_OxyCalibration.png"),
                             format="png", **options)

    plt.show()
    plt.close()


    # evaluate spectral index values according to tivita algorithms ..........
    # start = timer()
    # tissue.evaluate()
    # print("Elapsed time: %f sec" % (timer() - start))

    # param = tissue.get_solution(unpack=True, clip=False)
    #
    # param['oxy']
    # idx = np.nonzero(param['blo'])
    # param['oxy'][idx] = param['ohb'][idx] / param['blo'][idx]


if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()