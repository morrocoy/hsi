# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 14:45:53 2021

@author: kpapke

Example component fit analysis applied to hyperspectral image
"""
import sys
import os.path
import logging
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt

import hsi
from hsi import cm
from hsi import HSAbsorption
from hsi import HSIntensity, convert
from hsi import HSImage
from hsi.analysis import HSComponentFit

from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


def main():
    data_path = os.path.join(os.getcwd(), "../doc", "data")
    # pict_path = os.path.join(os.getcwd(), "../doc", "pictures")

    # load hyperspectral image
    # subfolder = "thyroid"
    # timestamp = "2019_11_14_08_59_25"
    subfolder = "occlusion"
    timestamp = "2016_11_02_16_48_30"
    file_path = os.path.join(data_path, subfolder, timestamp, timestamp)

    hsimage = HSImage(file_path + "_SpecCube.dat")
    hsimage.set_format(HSAbsorption)

    # apply filter
    # hsImage.add_filter(mode='image', type='mean', size=5)
    hsimage.add_filter(mode='image', filter_type='gauss', sigma=1, truncate=4)

    # nwavelen, rows, cols = hsImage.shape
    image = hsimage.as_rgb()
    spectra = hsimage.fspectra
    wavelen = hsimage.wavelen
    mask = hsimage.get_tissue_mask([0.1, 0.9])

    # plot rgb image and gray out non-tissue regions ..........................
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    idx = mask == 0  # gray out region out of mask
    gray = 0.2989 * red[idx] + 0.5870 * green[idx] + 0.1140 * blue[idx]
    red[idx] = gray
    green[idx] = gray
    blue[idx] = gray

    plt.imshow(image)
    plt.show()

    # component fit analysis ..................................................
    analysis = HSComponentFit(hsformat=HSAbsorption)

    analysis.loadtxt("basevectors_3.txt", mode='all')
    analysis.set_data(spectra, wavelen, hsformat=HSAbsorption)
    analysis.set_roi([500e-9, 995e-9])

    # print base component names
    print(analysis.keys)

    # modify bounds for component weights
    analysis.set_var_bounds("hhb", [0, 0.05])
    analysis.set_var_bounds("ohb", [0, 0.05])
    analysis.set_var_bounds("wat", [0, 2.00])
    analysis.set_var_bounds("fat", [0, 1.00])
    analysis.set_var_bounds("mel", [0, 0.05])

    # remove components
    # analysis.remove_component("mel")
    print(analysis.keys)

    analysis.prepare_ls_problem()

    a = analysis._anaSysMatrix
    start = timer()
    analysis.fit(method='bvls_f', mask=mask)
    print("Elapsed time: %f sec" % (timer() - start))

    # get dictionary of solutions (weights for each component
    param = analysis.get_solution(unpack=True, clip=True)

    # squared sum of residual over wavelength
    res = analysis.get_residual()

    # residual of hyperspectral image
    a = analysis._anaSysMatrix  # matrix of base vectors
    b = analysis._anaTrgVector  # spectra to be fitted
    x = analysis._anaVarVector  # vector of unknowns

    res = b - a[:, :-1] @ x[:-1, :]
    res = res.reshape(hsimage.shape)

    wavelen = analysis.components["hhb"].xIntpData
    spectrum = analysis.components["hhb"].yIntpData

    # residual conversion from absorption to intensity
    res_int = convert(HSIntensity, HSAbsorption, res, wavelen)

    # post processing - calculate bood and oxygenation from HHB and O2HB ......
    param['blo'] = param['hhb'] + param['ohb']
    param['oxy'] = np.zeros(param['blo'].shape)
    idx = np.nonzero(param['blo'])
    param['oxy'][idx] = param['ohb'][idx] / param['blo'][idx]

    # plot solution parameters ................................................
    cmap = cm.tivita()
    keys = ['oxy', 'blo', 'wat', 'fat']
    labels = ["Oxygenation", "Blood", "Water", "Fat"]

    fig = plt.figure()
    fig.set_size_inches(12, 8)
    for i, key in enumerate(keys):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.axis('off')
        if key in ('blo', 'ohb', 'hhb', 'mel'):
            pos = plt.imshow(param[key], cmap=cmap, vmin=0, vmax=0.05)
        elif key in ('oxy', 'fat'):
            pos = plt.imshow(param[key], cmap=cmap, vmin=0, vmax=1)
        elif key == 'wat':
            pos = plt.imshow(param[key], cmap=cmap, vmin=0, vmax=1.6)
        else:
            pos = plt.imshow(param[key], cmap=cmap, vmin=0, vmax=1.)
        fig.colorbar(pos, ax=ax)
        ax.set_title(labels[i])

    plt.show()


if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    # logmanager.setLevel(logging.ERROR)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
