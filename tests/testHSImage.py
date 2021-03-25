# -*- coding: utf-8 -*-
"""
Created on Fri Feb  12  7:57:45 2021

@author: kpapke
"""
import sys
import os
import logging

import numpy as np
import matplotlib.pyplot as plt

import hsi
from hsi import HSImage
from hsi import HSAbsorption
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)

# import logging
#
# LOGGING = True
# logger = logging.getLogger(__name__)
# logger.propagate = LOGGING

def main():

    data_path = os.path.join(os.getcwd(), "..", "data")
    pict_path = os.path.join(os.getcwd(), "..", "pictures")

    # load hyper spectral image data .........................................
    subfolder = "thyroid"
    timestamp = "2019_11_14_08_59_25"
    imgFilePath = os.path.join(data_path, subfolder, timestamp,
                               timestamp + "_SpecCube.dat")

    hsImage = HSImage(imgFilePath)
    rgbImage1 = hsImage.getRGBValue()

    hsImage.setFormat(HSAbsorption)
    hsiRaw = hsImage.spectra

    # add gaussian image filter
    hsImage.addFilter(mode='image', type='gauss', sigma=1, truncate=4)
    rgbImage2 = hsImage.getRGBValue()
    hsiGauss = hsImage.fspectra

    # add polynomial filter to spectra
    hsImage.addFilter(mode='spectra', type='savgol', size=7, order=2, deriv=0)
    hsiSGF = hsImage.fspectra

    # apply tissue mask
    mask = hsImage.getTissueMask(thresholds=[0.2, 0.8])
    rgbImage3 = rgbImage2 * mask[..., np.newaxis]

    rgbImage3 = hsImage.getRGBValue()
    # image_gray = np.dot(image_rgb, [0.2989, 0.5870, 0.1140])  # rgb to gray
    mask = hsImage.getTissueMask()

    red = rgbImage3[:, :, 0]
    green = rgbImage3[:, :, 1]
    blue = rgbImage3[:, :, 2]

    idx = np.nonzero(mask == 0)
    gray = 0.2989 * red[idx] + 0.5870 * green[idx] + 0.1140 * blue[idx]
    red[idx] = gray
    green[idx] = gray
    blue[idx] = gray


    # plot rgb pictures .....................................................
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    plt.imshow(rgbImage1)
    plt.show()

    fig = plt.figure()
    plt.imshow(rgbImage2)
    plt.show()

    fig = plt.figure()
    plt.imshow(rgbImage3)
    plt.show()

    # plot spectrum at point .................................................
    wavelength = hsImage.wavelen
    pnt = [337, 453]  # coordinates by index
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(wavelength, hsiRaw[:, pnt[0], pnt[1]], label='raw')
    ax.plot(wavelength, hsiGauss[:, pnt[0], pnt[1]], label='gauss')
    # ax.plot(wavelength, att_snv[:, pnt[0], pnt[1]], label='snv')
    ax.plot(wavelength, hsiSGF[:, pnt[0], pnt[1]], label='sgf')

    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("attenuation []")
    ax.text(0.98, 0.02, "x = %d\ny = %d" % tuple(pnt), transform=ax.transAxes,
            va='bottom', ha='right')

    ax.legend(loc=1)
    plt.show()


if __name__ == '__main__':
    # fmt = "%(asctime)s %(filename)35s: %(lineno)-4d: %(funcName)20s(): " \
    #       "%(levelname)-7s: %(message)s"
    # logging.basicConfig(level='DEBUG', format=fmt)

    # requests_logger = logging.getLogger('hsi')
    # requests_logger = logging.getLogger(__file__)
    # requests_logger.setLevel(logging.DEBUG)
    #
    # handler = logging.StreamHandler()
    # formatter = logging.Formatter(
    #         "%(asctime)s %(filename)35s: %(lineno)-4d: %(funcName)20s(): " \
    #           "%(levelname)-7s: %(message)s")
    # handler.setFormatter(formatter)
    # handler.setLevel(logging.DEBUG)
    # logger.addHandler(handler)
    # requests_logger.addHandler(handler)

    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()