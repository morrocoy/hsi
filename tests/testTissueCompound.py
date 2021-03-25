# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 07:54:44 2021

@author: kpapke
"""
import sys
import os.path
import logging

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from utils import readExcelTbl

import hsi
from hsi import HSTissueCompound
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


data_path = os.path.join(os.getcwd(), "..", "data")
pict_path = os.path.join(os.getcwd(), "..", "pictures")

fig_options = {
    'format': 'jpg',
    'dpi': 150,  # resolution
    'pil_kwargs': {
        'quality': 20,  # for jpg
        'optimize': True,  # for jpg
    },
    'bbox_inches': 'tight',
    'pad_inches': 0.03,
}


def testComponents():
    # load default tissue components ..........................................
    compound = HSTissueCompound()
    wavelen = compound.wavelen
    components = compound.components

    # load reference parameter for validation .................................
    refparam = {}
    filePath = os.path.join(data_path,
                            "Phantom builder parameters 2.xlsx")
    metadata = {
        'wat': {'sheet': 0, 'cols': [2, 3], 'rows': range(6, 400)},
        'fat': {'sheet': 0, 'cols': [4, 5], 'rows': range(6, 600)},
        'ohb': {'sheet': 0, 'cols': [6, 7], 'rows': range(6, 400)},
        'hhb': {'sheet': 0, 'cols': [8, 9], 'rows': range(6, 400)},
        'methb': {'sheet': 0, 'cols': [10, 11], 'rows': range(6, 400)},
        'cohb': {'sheet': 0, 'cols': [12, 13], 'rows': range(6, 400)},
        'shb': {'sheet': 0, 'cols': [14, 15], 'rows': range(6, 400)},
        'gref': {'sheet': 0, 'cols': [20, 21], 'rows': range(6, 400)},
    }

    for key, val in metadata.items():
        print(key)
        data = readExcelTbl(filePath, sheet=val['sheet'], nanchars=['', '-'],
                            rows=val['rows'], cols=val['cols'])
        data = np.array(data)
        idx = np.where(np.isnan(data))[0]  # find possible nan entries
        data = np.delete(data, idx, axis=0)  # remove nans

        f = interp1d(data[:, 0], data[:, 1], kind='linear')
        refparam[key] = f(wavelen)
    # add melanin manually from excel file
    refparam['mel'] = 519 * (wavelen / 500) ** -3.75

    # plot optical parameters in comparison ...................................
    for key in refparam.keys():
        fig = plt.figure()
        fig.set_size_inches(5, 3)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_yscale('log')

        y0 = refparam[key]
        if key == 'gref':
            y1 = compound._anisotropy(wavelen)
        else:
            y1 = components[key].absorption

        ax.plot(wavelen, y0, label='%s ref' % key, linewidth=1,
                marker='s', markersize=3, markeredgewidth=0.3,
                markerfacecolor='none', markevery=5)
        ax.plot(wavelen, y1, label='%s' % key, linewidth=0.7)


        ax.set_xlabel("wavelength [nm]")
        ax.set_ylabel("absorbtion coefficient [cm-1]")
        ax.legend()

        filePath = os.path.join(
            pict_path, "test_tissue_compound_%s" % key)
        plt.savefig(filePath + ".%s" % fig_options['format'], **fig_options)
        plt.show()
        plt.close(fig)


def testCompound(fileName):
    portions = {}
    filePath = os.path.join(data_path, fileName)
    with open(filePath, 'r') as file:
        file.readline()  # skip first line
        portions['blo'] = float(file.readline().split()[2].strip(",.:\'"))
        portions['ohb'] = float(file.readline().split()[2].strip(",.:\'"))
        portions['hhb'] = float(file.readline().split()[2].strip(",.:\'"))
        portions['methb'] = float(file.readline().split()[2].strip(",.:\'"))
        portions['cohb'] = float(file.readline().split()[2].strip(",.:\'"))
        portions['shb'] = float(file.readline().split()[2].strip(",.:\'"))
        portions['wat'] = float(file.readline().split()[2].strip(",.:\'"))
        portions['fat'] = float(file.readline().split()[2].strip(",.:\'"))
        portions['mel'] = float(file.readline().split()[2].strip(",.:\'"))
        skintype = file.readline().split()[2].strip(",.:\'")

        # reference data
        refdata = np.loadtxt(file, skiprows=2)

    compound = HSTissueCompound(portions=portions, skintype=skintype)
    compound.evaluate()

    icut = fileName.rfind('.')

    datasets = {
        'absorption': compound.absorption,
        'rscattering': compound.rscattering,
        'anisotropy': compound.anisotropy,
    }
    for i, (key, data) in enumerate(datasets.items()):
        fig = plt.figure()
        fig.set_size_inches(5, 5)
        ax = fig.add_subplot(2, 1, 1)
        ax.set_yscale('log')

        ax.set_title("Validation with Phantom Builder 2.4")
        ax.plot(refdata[:, 0], refdata[:, i+1], label='%s ref' % key, linewidth=1,
                marker='s', markersize=3, markeredgewidth=0.3,
                markerfacecolor='none', markevery=10)
        ax.plot(compound.wavelen, data, label=key)
        ax.set_xlabel("wavelength [nm]")
        ax.set_ylabel("%s" % key)
        ax.legend()

        ax = fig.add_subplot(2, 1, 2)
        ax.set_yscale('log')
        f = interp1d(refdata[:, 0], refdata[:, i + 1])
        res = np.abs(f(compound.wavelen) - data)
        ax.plot(compound.wavelen, res, label="residual")

        ax.set_xlabel("wavelength [nm]")
        ax.set_ylabel("residual")

        filePath = os.path.join(pict_path, "%s_%s" % (fileName[:icut], key))
        plt.savefig(filePath + ".%s" % fig_options['format'], **fig_options)
        plt.show()
        plt.close(fig)



def main():
    logger.info("Python executable: {}".format(sys.executable))
    testComponents()

    testCompound("test_tissue_compound1.txt")
    testCompound("test_tissue_compound2.txt")
    testCompound("test_tissue_compound3.txt")
    testCompound("test_tissue_compound4.txt")
    testCompound("test_tissue_compound5.txt")

if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()