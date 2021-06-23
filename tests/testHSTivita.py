# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 19:01:33 2021

@author: kpapke
"""
import sys
import os.path
import logging
from timeit import default_timer as timer
from scipy import ndimage

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

import hsi
from hsi import cm
from hsi import HSImage
from hsi import HSAbsorption, HSIntensity, HSExtinction, HSRefraction

from hsi.analysis import HSOpenTivita
from hsi.analysis import HSTivita
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


def main():

    data_path = os.path.join(os.getcwd(), "..", "data")
    pict_path = os.path.join(os.getcwd(), "..", "pictures")

    # load hyper spectral image data .........................................
    subfolder = "occlusion"
    timestamp = "2016_11_02_16_48_30"
    imgFilePath = os.path.join(data_path, subfolder, timestamp,
                               timestamp + "_SpecCube.dat")

    hsImage = HSImage(imgFilePath)

    hsImage.setFormat(HSIntensity)
    hsImage.addFilter(mode='image', type='gauss', sigma=1, truncate=4)

    spectra = hsImage.spectra  # raw spectral data
    fspectra = hsImage.fspectra  # filtered spectral data
    wavelen = hsImage.wavelen

    mask = hsImage.getTissueMask([0.1, 0.9])
    # mask = hsImage.getTissueMask([0.25, 0.9])
    # mask = hsImage.getTissueMask([0.4, 0.9])

    # Tivita algorithms
    tissue = HSTivita(format=HSIntensity)
    tissue.setData(spectra, wavelen, format=HSIntensity)

    # open source Tivita algorithms
    # tissue = HSOpenTivita(format=HSAbsorption)
    # tissue.set_data(fspectra, wavelen, format=HSIntensity)

    # evaluate spectral index values according to tivita algorithms ..........
    start = timer()
    tissue.evaluate(mask=mask)
    print("Elapsed time: %f sec" % (timer() - start))

    # plot mask ..............................................................
    filePath = os.path.join(pict_path, "tivita_mask")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    pos = plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    fig.colorbar(pos, ax=ax)
    plt.savefig(filePath + ".png", format="png", dpi=300)
    plt.show()

    # plot reference tivita index values for the hyperspectral image .........
    keys = ['oxy', 'nir', 'thi', 'twi']
    labels = ["Oxygenation", "NIR-Perfusion", "THI", "TWI"]

    fig = plt.figure()
    fig.set_size_inches(10, 8)
    fig.patch.set_visible(False)
    for i, key in enumerate(keys):
        ax = fig.add_subplot(2, 2, i+1)
        ax.axis('off')
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
    filePath = os.path.join(pict_path, "tivita_reference_index_values")
    plt.savefig(filePath + ".png", format="png", **options)
    plt.show()

    # plot tivita index values for the hyperspectral image ...................
    cmap = cm.tivita()

    param = tissue.getSolution(unpack=True, clip=False)
    keys = ['oxy', 'nir', 'thi', 'twi']

    fig = plt.figure()
    fig.set_size_inches(12, 8)
    fig.patch.set_visible(False)
    for i, key in enumerate(keys):
        ax = fig.add_subplot(2, 2, i+1)
        ax.axis('off')
        pos = plt.imshow(param[key], cmap=cmap, vmin=0, vmax=1)
        fig.colorbar(pos, ax=ax)
        ax.set_title(key.upper())

    options = {
        'bbox_inches': 'tight',
        'pad_inches': 0.03,
        'dpi': 300,  # high resolution png file
    }
    filePath = os.path.join(pict_path, "tivita_index_values")
    plt.savefig(filePath + ".png", format="png", **options)
    plt.show()

if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()