# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 19:01:33 2021

@author: kpapke
"""
import os.path
from timeit import default_timer as timer
from scipy import signal, ndimage

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

from hsi import HSImage, HSAbsorption, HSIntensity, HSExtinction, HSRefraction
from hsi.analysis import HSComponentFile, HSTivita
from hsi import cm

import logging

LOGGING = True
logger = logging.getLogger(__name__)
logger.propagate = LOGGING


def main():

    data_path = os.path.join(os.getcwd(), "..", "data")
    pict_path = os.path.join(os.getcwd(), "..", "pictures")

    # load hyper spectral image data .........................................
    subfolder = "occlusion"
    timestamp = "2016_11_02_16_48_30"
    imgFilePath = os.path.join(data_path, subfolder, timestamp,
                               timestamp + "_SpecCube.dat")

    hsImage = HSImage(imgFilePath)

    hsImage.setFormat(HSAbsorption)
    hsImage.addFilter(mode='image', type='mean', size=5)

    spectra = hsImage.fspectra
    wavelen = hsImage.wavelen
    mask = hsImage.getTissueMask([0.1, 0.9])
    # mask = hsImage.getTissueMask([0.25, 0.9])
    # mask = hsImage.getTissueMask([0.4, 0.9])

    tissue = HSTivita(format=HSAbsorption)
    tissue.setData(spectra, wavelen, format=HSAbsorption)


    # tissue = HSTivitaAnalysis(format=HSIntensity)
    # tissue = HSTivitaAnalysis(format=HSExtinction)
    # tissue = HSTivitaAnalysis(format=HSRefraction)

    # evaluate spectral index values according to tivita algorithms ..........
    start = timer()
    tissue.evaluate(mask=mask)
    print("Elapsed time: %f sec" % (timer() - start))

    # plot mask ..............................................................
    filePath = os.path.join(pict_path, "mask1.png")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pos = plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    fig.colorbar(pos, ax=ax)
    plt.savefig(filePath, format="png", dpi=300)
    plt.show()

    # plot reference tivita index values for the hyperspectral image .........
    keys = ['oxy', 'nir', 'thi']#, 'twi']
    labels = ["Oxygenation", "NIR-Perfusion", "THI"]

    fig = plt.figure()
    fig.set_size_inches(10, 8)
    for i, key in enumerate(keys):
        ax = fig.add_subplot(2, 2, i+1)
        imgFilePath = os.path.join(
            data_path, subfolder, timestamp, timestamp + "_%s.png" % labels[i])
        img = plt.imread(imgFilePath)
        pos = plt.imshow(img)
        # fig.colorbar(pos, ax=ax)
        ax.set_title(key.upper())

    options = {
        'bbox_inches': 'tight',
        'pad_inches': 0.03,
        'dpi': 300,  # high resolution png file
    }
    filePath = os.path.join(pict_path, "_tivita_reference_index_values.png")
    plt.savefig(filePath + ".png", format="png", **options)
    plt.show()

    # plot tivita index values for the hyperspectral image ...................
    cmap = cm.tivita()

    param = tissue.getVarVector(unpack=True, clip=False)
    keys = ['oxy', 'nir', 'thi', 'twi']

    fig = plt.figure()
    fig.set_size_inches(10, 8)
    for i, key in enumerate(keys):
        ax = fig.add_subplot(2, 2, i+1)
        pos = plt.imshow(param[key], cmap=cmap, vmin=0, vmax=1)
        fig.colorbar(pos, ax=ax)
        ax.set_title(key.upper())

    options = {
        'bbox_inches': 'tight',
        'pad_inches': 0.03,
        'dpi': 300,  # high resolution png file
    }
    filePath = os.path.join(pict_path, "_tivita_index_values.png")
    plt.savefig(filePath + ".png", format="png", **options)
    plt.show()


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