# -*- coding: utf-8 -*-
"""
Created on Fri May 27 14:45:53 2022

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
from hsi.analysis import HSCoFit

from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


def plot_params(analysis, which="last"):
    cmap = cm.tivita()
    keys = ['hhb', 'ohb', 'wat', 'fat', 'mel']

    prefix = analysis.prefix
    params = analysis.get_solution(which=which, unpack=True, clip=True)

    if which == "last":
        fig = plt.figure()
        fig.set_size_inches(4, 12)
        for i, key in enumerate(keys):
            ax = fig.add_subplot(len(keys), 1, i + 1)
            ax.axis('off')
            if key in ('blo', 'ohb', 'hhb', 'mel'):
                pos = plt.imshow(params[prefix + key], cmap=cmap, vmin=0,
                                 vmax=0.05)
            elif key in ('oxy', 'fat'):
                pos = plt.imshow(params[prefix + key], cmap=cmap, vmin=0,
                                 vmax=1)
            elif key == 'wat':
                pos = plt.imshow(params[prefix + key], cmap=cmap, vmin=0,
                                 vmax=1.6)
            else:
                pos = plt.imshow(params[prefix + key], cmap=cmap, vmin=0,
                                 vmax=1.)
            fig.colorbar(pos, ax=ax)
            ax.set_title(keys[i % len(keys)])
        plt.show()

    elif which == "all":
        fig = plt.figure()
        fig.set_size_inches(12, 12)
        for j in range(3):
            for i, key in enumerate(keys):
                ax = fig.add_subplot(len(keys), 3, j + 3 * i + 1)
                ax.axis('off')
                if key in ('blo', 'ohb', 'hhb', 'mel'):
                    pos = plt.imshow(params["%s%s_%d" % (prefix, key, j)],
                                     cmap=cmap, vmin=0, vmax=0.05)
                elif key in ('oxy', 'fat'):
                    pos = plt.imshow(params["%s%s_%d" % (prefix, key, j)],
                                     cmap=cmap, vmin=0, vmax=1)
                elif key == 'wat':
                    pos = plt.imshow(params["%s%s_%d" % (prefix, key, j)],
                                     cmap=cmap, vmin=0, vmax=1.6)
                else:
                    pos = plt.imshow(params["%s%s_%d" % (prefix, key, j)],
                                     cmap=cmap, vmin=0, vmax=1.)
                fig.colorbar(pos, ax=ax)
                ax.set_title(keys[i % len(keys)])

        plt.show()


def test_mc_simulations():
    # component fit analysis ..................................................
    index = 13
    analysis = HSCoFit(hsformat=HSAbsorption)
    analysis.loadtxt("basevectors_1.txt", mode='all')

    # print base component names
    print(analysis.keys)

    # modify bounds for component weights
    analysis.set_var_bounds("hhb", [0, 0.1])
    analysis.set_var_bounds("ohb", [0, 0.1])
    analysis.set_var_bounds("wat", [0, 2.00])
    analysis.set_var_bounds("fat", [0, 1.00])
    analysis.set_var_bounds("mel", [0, 0.20])

    analysis.prepare_ls_problem()
    prefix = analysis.prefix

    analysis.set_roi([520e-9, 995e-9])
    analysis.fit(method='bvls_f')

    spec_r = analysis.spectra[:, index]
    wavelen = analysis.wavelen

    param = analysis.get_solution(which="last", unpack=True, clip=True)
    spec_0 = analysis.model()[:, index]
    print(' '.join(["%s = %f" % (key, param[prefix + key][index])
                    for key in ["hhb", "ohb", "wat", "fat", "mel"]]))

    analysis.set_roi([520e-9, 600e-9])
    analysis.fit(method='bvls_f')
    spec_1 = analysis.model()[:, index]
    param = analysis.get_solution(which="last", unpack=True, clip=True)
    print(' '.join(["%s = %f" % (key, param[prefix + key][index])
                    for key in ["hhb", "ohb", "wat", "fat", "mel"]]))

    analysis.set_roi([700e-9, 800e-9])
    analysis.fit(method='bvls_f')
    spec_2 = analysis.model()[:, index]
    param = analysis.get_solution(which="last", unpack=True, clip=True)
    print(' '.join(["%s = %f" % (key, param[prefix + key][index])
                    for key in ["hhb", "ohb", "wat", "fat", "mel"]]))


    param = analysis.get_solution(which="all", unpack=True, clip=True)
    specs = analysis.model("all")
    # print(param.keys())

    for i in range(3):
        print(' '.join(
            ["%s = %f" % (key, param["%s%s_%d" % (prefix, key, i)][index])
             for key in ["hhb", "ohb", "wat", "fat", "mel"]]))

    # test plots
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4.5, 7))
    plt.subplots_adjust(hspace=0.1)

    ax1.plot(wavelen * 1e9, spec_r, 'k', label="spec_r",
             marker='o', markersize=5, markeredgewidth=0.5,
             markerfacecolor='none', markeredgecolor='k', markevery=5
             )
    ax1.plot(wavelen * 1e9, spec_0, label="spec_0")
    ax1.plot(wavelen * 1e9, spec_1, label="spec_1")
    ax1.plot(wavelen * 1e9, spec_2, label="spec_2")

    ax2.plot(wavelen * 1e9, spec_r, 'k', label="spec_r",
             marker='o', markersize=5, markeredgewidth=0.5,
             markerfacecolor='none', markeredgecolor='k', markevery=5
             )
    ax2.plot(wavelen * 1e9, specs[0, :, index], label="spec_0")
    ax2.plot(wavelen * 1e9, specs[1, :, index], label="spec_1")
    ax2.plot(wavelen * 1e9, specs[2, :, index], label="spec_2")

    ax1.set_xlabel("wavelength [nm]")
    ax1.set_ylabel("absorbtion")
    ax1.legend()

    ax2.set_xlabel("wavelength [nm]")
    ax2.set_ylabel("absorbtion")
    ax2.legend()

    plt.show()
    plt.close()

    return


def test_hsimage():
    data_path = os.path.join(os.getcwd(), "..", "data")
    data_path = os.path.join("d:", os.path.sep, "packages", "hsi", "data")
    # pict_path = os.path.join(os.getcwd(), "..", "pictures")

    # load hyperspectral image
    # subfolder = "thyroid"
    # timestamp = "2019_11_14_08_59_25"
    subfolder = "occlusion"
    timestamp = "2016_11_02_16_48_30"
    # subfolder = "2021-06-29_aufnahmen_arm"
    # timestamp = "2021_07_06_16_04_59"

    # subfolder = "wetransfer_2021_09_20_15_13_41_2022-02-23_2049"
    # # timestamp = "2021_09_20_15_13_41"
    # timestamp = "2021_09_27_15_42_49"

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

    # plt.imshow(image)
    # plt.show()

    # component fit analysis ..................................................
    row, col = 220, 520
    analysis = HSCoFit(hsformat=HSAbsorption)
    analysis.loadtxt("basevectors_1.txt", mode='all')
    analysis.set_data(spectra, wavelen, hsformat=HSAbsorption)

    # print base component names
    print(analysis.keys)

    # modify bounds for component weights
    analysis.set_var_bounds("hhb", [0, 0.1])
    analysis.set_var_bounds("ohb", [0, 0.1])
    analysis.set_var_bounds("wat", [0, 2.00])
    analysis.set_var_bounds("fat", [0, 1.00])
    analysis.set_var_bounds("mel", [0, 0.20])

    analysis.prepare_ls_problem()
    prefix = analysis.prefix

    analysis.set_roi([520e-9, 995e-9])
    analysis.fit(method='bvls_f', mask=mask)
    plot_params(analysis, which="last")

    spec_r = analysis.spectra[:, row, col]
    wavelen = analysis.wavelen

    param_0 = analysis.get_solution(which="last", unpack=True, clip=True)
    spec_0 = analysis.model()[:, row, col]
    print(' '.join(["%s = %f" % (key, param_0[prefix + key][row, col])
                    for key in ["hhb", "ohb", "wat", "fat", "mel"]]))

    analysis.set_roi([520e-9, 600e-9])
    analysis.fit(method='bvls_f', mask=mask)
    plot_params(analysis, which="last")
    param_1 = analysis.get_solution(which="last", unpack=True, clip=True)
    spec_1 = analysis.model()[:, row, col]
    print(' '.join(["%s = %f" % (key, param_1[prefix + key][row, col])
                    for key in ["hhb", "ohb", "wat", "fat", "mel"]]))


    analysis.set_roi([700e-9, 800e-9])
    analysis.fit(method='bvls_f', mask=mask)
    plot_params(analysis, which="last")
    spec_2 = analysis.model()[:, row, col]
    param_2 = analysis.get_solution(which="last", unpack=True, clip=True)
    print(' '.join(["%s = %f" % (key, param_2[prefix + key][row, col])
                    for key in ["hhb", "ohb", "wat", "fat", "mel"]]))

    plot_params(analysis, which="all")
    params = analysis.get_solution(which="all", unpack=True, clip=True)
    specs = analysis.model("all")

    for i in range(3):
        print(' '.join(
            ["%s = %f" % (key, params["%s%s_%d" % (prefix, key, i)][row, col])
             for key in ["hhb", "ohb", "wat", "fat", "mel"]]))

    keys = ['hhb', 'ohb', 'wat', 'fat', 'mel']
    for i, key in enumerate(keys):
        diff_0 = np.max(np.abs(
            params["%s%s_%d" % (prefix, key, 2)] - param_0[prefix + key]))
        diff_1 = np.max(np.abs(
            params["%s%s_%d" % (prefix, key, 1)] - param_1[prefix + key]))
        diff_2 = np.max(np.abs(
            params["%s%s_%d" % (prefix, key, 0)] - param_2[prefix + key]))
        print ("diff %s: %f %f %f" % (key, diff_0, diff_1, diff_2))


def main():
    start = timer()
    test_mc_simulations()
    test_hsimage()
    print("Elapsed time: %f sec" % (timer() - start))



if __name__ == '__main__':
    # logmanager.setLevel(logging.DEBUG)
    logmanager.setLevel(logging.ERROR)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
