# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:59:24 2021

@author: kpapke
"""
import os
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt

from hsi import cm
from hsi import HSAbsorption
from hsi import HSImage

from hsi.analysis import HSComponentFit
# from hsi.analysis import HSComponentFit2 as HSComponentFit

import logging

LOGGING = True
logger = logging.getLogger(__name__)
logger.propagate = LOGGING


fig_options = {
    'format': 'png',
    'dpi': 300, # resolution
    'pil_kwargs': {
        'quality': 10,  # for jpg
        'optimize': True,  # for jpg
    },
    'bbox_inches': 'tight',
    'pad_inches': 0.03,
}



def main():

    data_path = os.path.join(os.getcwd(), "..", "data")
    pict_path = os.path.join(os.getcwd(), "..", "pictures")

    # load hyperspectral data
    # subfolder = "thyroid"
    # timestamp = "2019_11_14_08_59_25"
    subfolder = "occlusion"
    timestamp = "2016_11_02_16_48_30"
    filePath = os.path.join(data_path, subfolder, timestamp, timestamp)

    hsImage = HSImage(filePath + "_SpecCube.dat")
    hsImage.setFormat(HSAbsorption)
    # hsImage.addFilter(mode='image', type='mean', size=5)
    hsImage.addFilter(mode='image', type='gauss', sigma=1, truncate=4)

    nwavelen, rows, cols = hsImage.shape
    image = hsImage.getRGBValue()
    spectra = hsImage.fspectra
    wavelen = hsImage.wavelen
    mask = hsImage.getTissueMask([0.1, 0.9])

    # plot rgb image .........................................................
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    idx = np.nonzero(mask == 0)  # gray out region out of mask
    gray = 0.2989 * red[idx] + 0.5870 * green[idx] + 0.1140 * blue[idx]
    red[idx] = gray * 0
    green[idx] = gray * 0
    blue[idx] = gray * 0

    plt.imshow(image)
    plt.show()

    # analysis and post processing ...........................................
    analysis = HSComponentFit(format=HSAbsorption)

    analysis.loadtxt("basevectors_2.txt", mode='all')
    analysis.setData(spectra, wavelen, format=HSAbsorption)
    analysis.setROI([500e-9, 995e-9])
    analysis.prepareLSProblem()

    start = timer()
    analysis.fit(method='bvls_f', mask=mask)
    print("Elapsed time: %f sec" % (timer() - start))

    param = analysis.getSolution(unpack=True, clip=True)
    param['blo'] = param['hhb'] + param['ohb']
    param['oxy'] = np.zeros(param['blo'].shape)
    idx = np.nonzero(param['blo'])
    param['oxy'][idx] = param['ohb'][idx] / param['blo'][idx]

    # plot solution parameters ...............................................
    cmap = cm.tivita()
    keys = ['oxy', 'blo', 'wat', 'fat']
    labels = ["Oxygenation", "Blood", "Water", "Fat"]

    fig = plt.figure()
    fig.set_size_inches(10, 8)
    for i, key in enumerate(keys):
        ax = fig.add_subplot(2, 2, i + 1)
        if key in ('blo', 'ohb', 'hhb', 'mel'):
            pos = plt.imshow(param[key], cmap=cmap, vmin=0, vmax=0.05)
        elif key in ('oxy', 'fat'):
            pos = plt.imshow(param[key], cmap=cmap, vmin=0, vmax=1)
        elif key in ('wat'):
            pos = plt.imshow(param[key], cmap=cmap, vmin=0, vmax=1.6)
        # fig.colorbar(pos, ax=ax)
        ax.set_title(labels[i])

    filePath = os.path.join(pict_path, "componentfit_param")
    fig.savefig("%s.%s" % (filePath, fig_options['format']), **fig_options)
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
