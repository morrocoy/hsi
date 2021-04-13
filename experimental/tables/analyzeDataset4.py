# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:32:15 2021

@author: kpapke
"""
import sys
import os.path
import logging

from timeit import default_timer as timer

import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import tables

from tables_utils import getDirPaths, plotMasks, plotParam

import hsi
from hsi import HSStore
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)



def plotMultipleXY(filePath, data, target, keys1, keys2):
    ipos = np.where(target == 1)[0]
    ineg = np.where(target == 0)[0]
    nmask = 1

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']


    fig = plt.figure()
    fig.set_size_inches(6*nmask, 15)

    for iset, (key1, key2) in enumerate(zip(keys1, keys2)):
        for imask in range(nmask):
            ax = fig.add_subplot(3, nmask, iset * (nmask) + imask + 1)
            # ax.set_title(masks[imask])
            x = data[key1]
            y = data[key2]
            ax.plot(
                x[ipos], y[ipos], label="healed", linestyle='None',
                marker='o',
                color=colors[1], markersize=5, markeredgewidth=0.3,
                markerfacecolor=colors[1])
            ax.plot(
                x[ineg], y[ineg], label="not healed", linestyle='None',
                marker='s',
                color=colors[0], markersize=5, markeredgewidth=0.3,
                markerfacecolor=colors[0])

            ax.set_xlabel(key1.upper())
            ax.set_ylabel(key2.upper())
            ax.legend()

    # baseName, ext = os.path.split(filePath)
    fig.savefig(filePath, format='png', dpi=300)
    plt.show()
    plt.close()


def main():

    dirPaths = getDirPaths()

    start = timer()

    # fileName = "rostock_suedstadt_2018-2020_4.h5"
    # fileName = "rostock_suedstadt_2018-2020_4_test.h5"

    dataset = "rostock_suedstadt_2018-2020_4"
    # dataset = "rostock_suedstadt_2018-2020_4_test"
    study = "Tivita_004"
    # study = "CoFit_001"

    filePath = os.path.join(dirPaths['data'], dataset + ".h5")

    config = {
        "oxy": {"mask": "wound"},
        "nir": {"mask": "wound"},
        "thi": {"mask": "proximity"},
        "twi": {"mask": "wound"},
    }

    with HSStore.open(filePath, mode="r", path="/records") as reader:
        # attach tables
        reader.attacheTable("patient")
        reader.attacheTable("masks")
        reader.attacheTable("tivita")

        print(f"Tables to read: {reader.getTableNames()}")
        print(f"Number of entries: {len(reader)}")

        # patient data
        patients = pd.DataFrame.from_records(reader.getTable("patient")[:])

        # features used for classifications
        features = pd.DataFrame(
            {key: np.zeros(len(reader)) for key in config.keys()})

        # fill features by evaluating tivita index values for each record
        for i, (patient, masks, params) in enumerate(reader):
            print("%8d | %8d | %-20s | %-20s | %3d |" % (
                patient["pn"],
                patient["pid"],
                patient["descr"].decode(),
                patient["timestamp"].decode(),
                patient["target"]
            ))

            fileName = "PN_%03d_PID_%07d_Date_%s_Masks_test.jpg" % (
                patient["pn"], patient["pid"], patient["timestamp"].decode())
            # plotMasks(fileName, masks)

            fileName = "PN_%03d_PID_%07d_Date_%s_Tivita_test.jpg" % (
                patient["pn"], patient["pid"], patient["timestamp"].decode())
            # plotParam(fileName, params)

            for key, val in config.items():
                p = params[key].reshape(-1)
                mask = masks[val["mask"]]
                values = p[mask.reshape(-1) == 1]
                # features[key][i] = np.mean(values)
                features[key][i] = np.percentile(values, q=50)

    print(patients)
    print(features)

    keys1 = ["oxy", "oxy", "oxy"]
    keys2 = ["nir", "thi", "twi"]
    target = patients["target"]

    filePath = os.path.join(dirPaths["pictures"], "%s_Study_%s_mean.png" % (
    dataset, study))
    plotMultipleXY(filePath, features, target, keys1, keys2)


    print("\nElapsed time: %f sec" % (timer() - start))





if __name__ == '__main__':
    # logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
