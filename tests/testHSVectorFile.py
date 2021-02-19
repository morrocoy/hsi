# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 07:54:44 2021

@author: kai
"""
import sys
import os
from shutil import copyfile

import numpy as np

import hsi
from hsi.analysis import HSVector, HSVectorFile
from hsi import HSIntensity, HSAbsorption, HSRefraction

import logging

LOGGING = True
logger = logging.getLogger(__name__)
logger.propagate = LOGGING



def main():
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    data_path = os.path.join(os.getcwd(), "..", "data")

    nwavelen = 100
    wavelen = np.linspace(500e-9, 1000e-9, nwavelen, endpoint=False)

    data = np.random.rand(nwavelen, 10)
    data = 2*np.ones((nwavelen, 10))

    # example of one-dimensional numpy array
    spec1 = data[:, 0]
    # example of list
    spec2 = list(data[:, 0])
    # example of two-dimensional numpy array
    spec3 = data[:, :3]
    # example of multi-dimensional numpy array
    spec4 = data[:, :4].reshape(nwavelen, 2,2)

    # spectral base vectors
    names = ["hhb", "ohb", "wat", "fat", "mel"]
    labels = ["HHB", "O2Hb", "Water", "Fat", "Melanin"]
    formats = [HSAbsorption, HSAbsorption, HSIntensity, HSAbsorption, HSAbsorption]
    weights = [0.005, 0.005, 0.70, 0.70, 0.025]
    bounds = [(0, 0.05), (0, 0.05), (0., 1.), (0., 1.), (0., 0.05)]
    vectors = {}
    for i, name in enumerate(names):
        vectors[name] = HSVector(
            data[:, i], wavelen, wavelen, name=name, label=labels[i],
            format=formats[i], weight=weights[i], bounds=bounds[i])

    fpath1 = os.path.join(data_path, "hsvectorfile_test1.txt")
    fpath2 = os.path.join(data_path, "hsvectorfile_test2.txt")
    fpath3 = os.path.join(data_path, "hsvectorfile_test3.txt")
    fpath4 = os.path.join(data_path, "hsvectorfile_test4.txt")
    fpath5 = os.path.join(data_path, "..", "hsi", "data", "hsvectorfile_test3.txt")

    # write datasets without vectors to file
    with HSVectorFile(fpath1) as file:
        file.buffer(spec1, wavelen, label="spec1")
        file.buffer(spec2, label="spec2", format=HSAbsorption)
        file.buffer(spec3, label="spec3", format=HSRefraction)
        file.write(spec4, label="spec4", format=HSIntensity)

    copyfile(fpath1, fpath2)

    # load datasets without vectors from file
    with HSVectorFile(fpath2) as file:
        file.load()
        file.setFormat(HSAbsorption)
        file.write()
        # file.write()
        # file.write()

    # write datasets with vectors to file
    with HSVectorFile(fpath3) as file:
        # file.buffer(spec1, wavelen, label="spec1")
        file.buffer(vectors['hhb'])
        file.buffer(vectors['ohb'])
        file.buffer(spec4, label="spec4", format=HSAbsorption)
        file.buffer(vectors['mel'])
        file.write(vectors['wat'])

    copyfile(fpath3, fpath4)

    # extend a dataset
    with HSVectorFile(fpath4) as file:
        file.load()
        file.write(vectors['fat'])
        file.setFormat(HSAbsorption)
        file.write()
        file.write()

    copyfile(fpath3, fpath5)

    # load hsi file from package data folder
    # with HSVectorFile("hsvectorfile_test3.txt") as file:
    with HSVectorFile(fpath3) as file:
        rvectors, rspec, rwavelen = file.read()
        # rspec = file.read()
        version = file.version
        format = file.format

    # check whether containt of 'file' is lost after exit the with-statement
    print(file.version)
    print(file.wavelen)

    # check whether the data taken from file are still there
    print(version)
    print(format)

    for key, val in rspec.items():
        print(key, val[0])

    rvectors['mel'].setFormat(HSAbsorption)
    for key, vec in rvectors.items():
        print(key, vec.label, vec.weight, vec.bounds, vec.format.key, vec.xIntpData[0], vec.yIntpData[0])



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




