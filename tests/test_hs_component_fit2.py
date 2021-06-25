# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 16:59:24 2021

@author: kpapke
"""
import sys
import os
import logging
from timeit import default_timer as timer

import numpy as np
import matplotlib.pyplot as plt

import hsi
from hsi import cm
from hsi import HSAbsorption
from hsi import HSImage
from hsi.analysis import HSComponentFit
# from hsi.analysis import HSComponentFit2 as HSComponentFit
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


fig_options = {
    'format': 'png',
    'dpi': 300,  # resolution
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
    file_path = os.path.join(data_path, subfolder, timestamp, timestamp)

    hsimage = HSImage(file_path + "_SpecCube.dat")
    hsimage.set_format(HSAbsorption)
    # hsimage.add_filter(mode='image', type='mean', size=5)
    hsimage.add_filter(mode='image', filter_type='gauss', sigma=1, truncate=4)

    # nwavelen, rows, cols = hsimage.shape
    image = hsimage.as_rgb()
    spectra = hsimage.fspectra
    wavelen = hsimage.wavelen
    mask = hsimage.get_tissue_mask([0.1, 0.9])

    # plot rgb image .........................................................
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    idx = mask == 0  # gray out region out of mask
    gray = 0.2989 * red[idx] + 0.5870 * green[idx] + 0.1140 * blue[idx]
    red[idx] = gray * 0
    green[idx] = gray * 0
    blue[idx] = gray * 0

    plt.imshow(image)
    plt.show()

    # analysis and post processing ...........................................
    analysis = HSComponentFit(hsformat=HSAbsorption)

    analysis.loadtxt("basevectors_2.txt", mode='all')
    analysis.set_data(spectra, wavelen, hsformat=HSAbsorption)
    analysis.set_roi([500e-9, 995e-9])

    # print base component names
    print(analysis.keys)

    # modify bounds for component weights
    # analysis.set_var_bounds("hhb", [0, np.inf])
    analysis.set_var_bounds("hhb", [0, 0.05])
    analysis.set_var_bounds("hhb", [-np.inf, np.inf])
    analysis.set_var_bounds("ohb", [0, 0.05])
    analysis.set_var_bounds("wat", [0, 2.00])
    analysis.set_var_bounds("fat", [0, 1.00])
    analysis.set_var_bounds("mel", [0, 0.05])

    analysis.prepare_ls_problem()

    start = timer()
    analysis.fit(method='bvls_f', mask=mask)
    print("Elapsed time: %f sec" % (timer() - start))

    param = analysis.get_solution(unpack=True, clip=True)
    param['blo'] = param['hhb'] + param['ohb']
    param['oxy'] = np.zeros(param['blo'].shape)
    idx = np.nonzero(param['blo'])
    param['oxy'][idx] = param['ohb'][idx] / param['blo'][idx]

    # plot solution parameters ...............................................
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
            pos = plt.imshow(param[key], cmap=cmap, vmin=0, vmax=1)
        fig.colorbar(pos, ax=ax)
        ax.set_title(labels[i])

    file_path = os.path.join(pict_path, "componentfit_param")
    fig.savefig("%s.%s" % (file_path, fig_options['format']), **fig_options)
    plt.show()


if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
