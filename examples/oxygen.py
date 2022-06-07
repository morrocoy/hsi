# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 12:20:00 2022

@author: kpapke
"""
import sys
import os.path
import logging
from timeit import default_timer as timer

import matplotlib.pyplot as plt

import hsi
from hsi import cm
from hsi import HSImage

from hsi.analysis import HSOxygen
from hsi import HSIntensity, HSAbsorption

from hsi.log import logmanager


logger = logmanager.getLogger(__name__)


def main():
    data_path = os.path.join(os.getcwd(), "..", "data")
    pict_path = os.path.join(os.getcwd(), "..", "pictures")

    # load hyper spectral image data .........................................
    subfolder = "occlusion"
    timestamp = "2016_11_02_16_48_30"
    image_file_path = os.path.join(
        data_path, subfolder, timestamp, timestamp + "_SpecCube.dat")

    hsimage = HSImage(image_file_path)
    hsimage.set_format(HSIntensity)
    hsimage.add_filter(mode='image', filter_type='gauss', sigma=1, truncate=4)

    wavelen = hsimage.wavelen

    mask = hsimage.get_tissue_mask([0.1, 0.9])
    # mask = hsImage.get_tissue_mask([0.25, 0.9])
    # mask = hsImage.get_tissue_mask([0.4, 0.9])

    # Tivita algorithms
    spectra = hsimage.spectra  # raw spectral data
    tissue = HSOxygen(hsformat=HSIntensity)
    tissue.set_data(spectra, wavelen, hsformat=HSIntensity)


    # evaluate spectral index values according to tivita algorithms ..........
    start = timer()
    # mask=None
    tissue.evaluate(mask=mask)
    print("Elapsed time: %f sec" % (timer() - start))

    # plot mask ..............................................................
    # file_path = os.path.join(pict_path, "tivita_mask")
    # plt.imshow(mask, cmap='gray', vmin=0, vmax=1)
    # plt.savefig(file_path + ".png", format="png", dpi=300)
    # plt.show()


    # plot tivita index values for the hyperspectral image ...................
    cmap = cm.tivita()

    param = tissue.get_solution(unpack=True, clip=False)
    keys = tissue.keys
    labels = tissue.labels

    fig = plt.figure()
    # fig.set_size_inches(6, 8)
    # fig.patch.set_visible(False)
    for i, key in enumerate(keys):
        ax = fig.add_subplot(1, 1, i+1, xticks = [], yticks = [])
        pos = plt.imshow(param[key], cmap=cmap, vmin=0, vmax=1)

        # fig.colorbar(pos, ax=ax)
        # ax.set_title(key.upper())#, fontsize=14)
        ax.set_title(labels[i])  # , fontsize=14)

    options = {
        'bbox_inches': 'tight',
        'pad_inches': 0.03,
        'dpi': 300,  # high resolution png file
    }
    file_path = os.path.join(pict_path, "oxy_index_values")
    plt.savefig(file_path + ".png", format="png", **options)
    plt.show()


if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    # logmanager.setLevel(logging.ERROR)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
