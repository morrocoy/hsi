# -*- coding: utf-8 -*-
"""
Created on Fri Feb  12  7:57:45 2021

@author: kpapke
"""
import sys
import os
import logging

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

    # load hyper spectral image data .........................................
    subfolder = "thyroid"
    timestamp = "2019_11_14_08_59_25"
    img_file_path = os.path.join(
        data_path, subfolder, timestamp, timestamp + "_SpecCube.dat")

    hsimage = HSImage(img_file_path)
    rgb_image1 = hsimage.as_rgb()

    hsimage.set_format(HSAbsorption)
    hsi_raw = hsimage.spectra

    # add gaussian image filter
    hsimage.add_filter(mode='image', filter_type='gauss', sigma=1, truncate=4)
    rgb_image2 = hsimage.as_rgb()
    hsi_gauss = hsimage.fspectra

    # add polynomial filter to spectra
    hsimage.add_filter(
        mode='spectra', filter_type='savgol', size=7, order=2, deriv=0)
    hsi_sgf = hsimage.fspectra

    # apply tissue mask
    # mask = hsimage.get_tissue_mask(thresholds=[0.2, 0.8])
    # rgb_image3 = rgb_image2 * mask[..., numpy.newaxis]

    rgb_image3 = hsimage.as_rgb()
    # image_gray = numpy.dot(image_rgb, [0.2989, 0.5870, 0.1140])  # rgb to gray
    mask = hsimage.get_tissue_mask()

    red = rgb_image3[:, :, 0]
    green = rgb_image3[:, :, 1]
    blue = rgb_image3[:, :, 2]

    idx = mask == 0  # gray out region out of mask
    gray = 0.2989 * red[idx] + 0.5870 * green[idx] + 0.1140 * blue[idx]
    red[idx] = gray
    green[idx] = gray
    blue[idx] = gray

    # plot rgb pictures .....................................................
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    plt.imshow(rgb_image1)
    plt.show()

    plt.imshow(rgb_image2)
    plt.show()

    plt.imshow(rgb_image3)
    plt.show()

    # plot spectrum at point .................................................
    wavelength = hsimage.wavelen
    pnt = [337, 453]  # coordinates by index
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(wavelength, hsi_raw[:, pnt[0], pnt[1]], label='raw')
    ax.plot(wavelength, hsi_gauss[:, pnt[0], pnt[1]], label='gauss')
    # ax.plot(wavelength, att_snv[:, pnt[0], pnt[1]], label='snv')
    ax.plot(wavelength, hsi_sgf[:, pnt[0], pnt[1]], label='sgf')

    ax.set_xlabel("wavelength [nm]")
    ax.set_ylabel("attenuation []")
    ax.text(0.98, 0.02, "x = %d\ny = %d" % tuple(pnt), transform=ax.transAxes,
            va='bottom', ha='right')

    ax.legend(loc=1)
    plt.show()


if __name__ == '__main__':
    # fmt = "%(asctime)s %(filename)35s: %(lineno)-4d: %(funcName)20s(): " \
    #       "%(levelname)-7s: %(message)s"
    # logging.basicConfig(level='DEBUG', hsformat=fmt)

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
