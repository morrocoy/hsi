# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:46:45 2021

@author: kpapke
"""
import sys
import os.path
import logging
from shutil import copyfile

import numpy as np

import hsi
from hsi import HSFile
from hsi import HSIntensity, HSAbsorption, HSRefraction
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)


def main():
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    data_path = os.path.join(os.getcwd(), "..", "data")

    nwavelen = 100
    wavelen = np.linspace(500e-9, 1000e-9, nwavelen, endpoint=False)

    # data = np.random.rand(nwavelen, 4)
    data = 2*np.ones((nwavelen, 4))

    # example of one-dimensional numpy array
    spec1 = data[:, 0]
    # example of list
    spec2 = list(data[:, 0])
    # example of two-dimensional numpy array
    spec3 = data[:, :3]
    # example of multi-dimensional numpy array
    spec4 = data[:, :4].reshape(nwavelen, 2, 2)

    # define datasets for hsfile

    fpath1 = os.path.join(data_path, "hsfile_test1.txt")
    fpath2 = os.path.join(data_path, "hsfile_test2.txt")
    fpath3 = os.path.join(data_path, "hsfile_test3.txt")
    fpath4 = os.path.join(data_path, "..", "hsi", "data", "hsfile_test1.txt")

    # write datasets to file
    with HSFile(fpath1) as file:
        file.buffer(spec1, wavelen, label="spec1")
        file.buffer(spec2, label="spec2", hsformat=HSAbsorption)
        file.buffer(spec3, label="spec3", hsformat=HSRefraction)
        file.write(spec4, label="spec4", hsformat=HSIntensity)

    copyfile(fpath1, fpath2)

    # load datasets from file
    with HSFile(fpath2) as file:
        file.load()
        file.set_format(HSAbsorption)
        file.write()

    copyfile(fpath2, fpath3)

    # extend a dataset
    with HSFile(fpath3) as file:
        file.load()
        file.write(2 * spec1, label="spec2", hsformat=HSAbsorption)

    copyfile(fpath1, fpath4)

    # load hsi file from package data folder
    with HSFile("hsfile_test1.txt") as file:
        rspec, rwavelen = file.read()
        # rspec = file.read()
        version = file.version

    # check whether containt of 'file' is lost after exit the with-statement
    print(file.version)
    print(file.wavelen)

    # check whether the data taken from file are still there
    print(version)
    for key, val in rspec.items():
        print(key, val[0])


if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
