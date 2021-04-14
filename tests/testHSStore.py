# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:46:45 2021

@author: kpapke

Clinical data example
----------------------------------------------------------------------------

**Data Set Characteristics:**

    :Number of Instances: 3 (1 of healing and 2 of not healing)
    :Number of Attributes: 7 meta, 3 numeric, predictive and 2 class attributes
    :Attribute Information:
        - patient (patient info):
            - pn: Patient number
            - pid: Patient ID
            - descr: Description of the wound
            - timestamp: Timestamp as string
            - class (target):
                - not healed with target index 0
                - healed with target index 1
        - hsidata:
            - format: Spectral format
            - hsidata: Hyperspectral data
            - wavelen: Wavelengths at which the spectral information is sampled
        - masks (Selection masks applied on the hyperspectral images)
            - tissue
            - critical
            - wound
            - proximity
"""
import sys
import os.path
import logging
import multiprocessing
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import tables
import matplotlib.pyplot as plt

import hsi
from hsi import HSImage, HSFormatFlag, HSIntensity, HSAbsorption
from hsi import HSStore
from hsi.analysis import HSTivita, HSOpenTivita
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)

data_path = os.path.join(os.getcwd(), "..", "data")
pict_path = os.path.join(os.getcwd(), "..", "pictures")

# define columns for the patient info table
HSPatientInfo = np.dtype([
    ("pn", '<i8'),
    ("pid", '<i8'),
    ("name", 'S32'),
    ("descr", 'S64'),
    ("timestamp", 'S32'),
    ("target", '<i4'),
])


def createDataset(fileName, path, descr=None):

    patientData = pd.DataFrame({
        "pn" : [1,2,3],
        "pid": [100001, 100002, 100003],
        "name": ["Smith", "Jones", "Williams"],
        "descr": ["1st beam left", "2nd beam left", "3rd beam left"],
        "timestamp": ["2021-01-01-12-00-00", "2021-01-02-12-00-00",
                      "2021-01-03-12-00-00"],
        "target": [0, 0, 1],
    })
    wavelen = np.linspace(500e-9, 1000e-9, 100, endpoint=False)

    np.random.seed(29012392)  # for reproducible results

    start = timer()
    filePath = os.path.join(data_path, fileName)
    with HSStore.open(
            filePath, mode="w", path=path, descr=descr) as dataset:
        rows = len(patientData.index)

        tablePatient = dataset.createTable(
            name="patient",
            dtype=HSPatientInfo,
            title="Patient information",
            expectedrows=rows,
        )

        tableHSImage = dataset.createTable(
            name="hsimage",
            dtype=np.dtype([
                ("hsformat", "<S32"),
                ("wavelen", "<f8", (100,)),
                ("spectra", "<f4", (100, 480, 640))
            ]),
            title="Hyperspectral image data",
            expectedrows=rows,
        )

        tableMasks = dataset.createTable(
            name="masks",
            dtype=np.dtype([
                ("tissue", "<i1", (480, 640)),
                ("critical", "<i1", (480, 640)),
                ("wound", "<i1", (480, 640)),
                ("proximity", "<i1", (480, 640)),
            ]),
            title="Masks applied on image data",
            expectedrows=rows,

        )

        entryPatient = tablePatient.row
        entryHSImage = tableHSImage.row
        entryMasks = tableMasks.row
        for index, info in patientData.iterrows():

            print("Process index %d ..." % info["pn"])

            entryPatient["pn"] = info["pn"]  # i f'Particle: {i:6d}'
            entryPatient["pid"] = info["pid"]
            entryPatient["name"] = str.encode(info["name"])
            entryPatient["descr"] = str.encode(info["descr"])
            entryPatient["timestamp"] = str.encode(info["timestamp"])
            entryPatient["target"] = info["target"]
            entryPatient.append()


            entryHSImage["hsformat"] = str.encode(HSIntensity.key)
            entryHSImage["wavelen"] = wavelen.astype("<f8")
            entryHSImage["spectra"] = np.random.random(
                (100, 480, 640)).astype("<f4")
            entryHSImage.append()

            entryMasks["tissue"] = np.random.randint(
                2, size=(480, 640), dtype="<i1")
            entryMasks["critical"] = np.random.randint(
                2, size=(480, 640), dtype="<i1")
            entryMasks["wound"] = np.random.randint(
                2, size=(480, 640), dtype="<i1")
            entryMasks["proximity"] = np.random.randint(
                2, size=(480, 640), dtype="<i1")
            entryMasks.append()

        tablePatient.flush()
        tableHSImage.flush()
        tableMasks.flush()

        print("\nElapsed time for creating dataset: %f sec" % (timer() - start))



def task(args):
    """Example task applied on each entry.

    Parameters
    ----------
    patient : pd.Series
        Metadata of the record.
    spectra :  numpy.ndarray
        The spectral data.
    wavelen :  numpy.ndarray
        The wavelengths at which the spectral data are sampled.
    masks :  numpy.ndarray
        Masks to be applied on the hyperspectral image.

    Returns
    -------
    numpy.ndarray : Array of values for validation.

    """
    patient = args[0]
    hsidata = args[1]
    masks = args[2]

    # hsformat = HSFormatFlag.fromStr(patient["hsformat"].decode())
    hsformat = HSFormatFlag.fromStr(hsidata["hsformat"].decode())

    print("%8d | %8d | %-20s | %-20s | %-10s | %3d |" % (
        patient["pn"],
        patient["pid"],
        patient["descr"].decode(),
        patient["timestamp"].decode(),
        hsformat.key,
        patient["target"]
    ))

    hsImage = HSImage(
        spectra=hsidata["spectra"], wavelen=hsidata["wavelen"], format=hsformat)
    image = hsImage.getRGBValue()

    # analysis = HSOpenTivita(format=HSAbsorption)  # open source algorithms
    analysis = HSTivita(format=HSIntensity)  # true algorithms

    analysis.setData(hsImage.spectra, hsImage.wavelen, format=hsformat)
    analysis.evaluate(mask=masks["tissue"])
    param = analysis.getSolution(unpack=True, clip=True)

    return param


def processDataset(fileName):
    """ Evaluate the tivita index values for each record.

    The evaluation is defined in task(args) where args are the entries of the
    attached tables.
    """
    start = timer()
    filePath = os.path.join(data_path, fileName)
    with tables.open_file(filePath, "r+") as file:
        reader = HSStore(file, path="/records")
        writer = HSStore(file, path="/records")

        reader.attacheTable("patient")
        reader.attacheTable("hsimage")
        reader.attacheTable("masks")

        tableTivita = writer.createTable(
            name="tivita",
            dtype=np.dtype([
                ("oxy", "<f8", (480, 640)),
                ("nir", "<f8", (480, 640)),
                ("thi", "<f8", (480, 640)),
                ("twi", "<f8", (480, 640)),
            ]),
            title="Tivita Index values",
            expectedrows=len(reader),
        )
        entryTivita = tableTivita.row

        print(f"Tables to read: {reader.getTableNames()}")
        print(f"Tables to write: {writer.getTableNames()}")
        print(f"Number of entries: {len(reader)}")

        # serial evaluation
        # for args in iter(reader):
        #     param = task(args)
        #     entryTivita["nir"] = param["nir"]
        #     entryTivita["oxy"] = param["oxy"]
        #     entryTivita["thi"] = param["thi"]
        #     entryTivita["twi"] = param["twi"]
        #     entryTivita.append()
        #
        # tableTivita.flush()

        # parallel evaluation
        pool = multiprocessing.Pool(processes=7)
        for param in pool.imap(task, iter(reader)):  # , chunksize=1):
            entryTivita["oxy"] = param["oxy"]
            entryTivita["nir"] = param["nir"]
            entryTivita["thi"] = param["thi"]
            entryTivita["twi"] = param["twi"]
            entryTivita.append()
        pool.close()

        tableTivita.flush()

        print("\nElapsed time for processing dataset: %f sec" %
              (timer() - start))


def plotMultipleXY(fileName, data, target, keys1, keys2):
    """ Helper function for plotting. """

    ipos = np.where(target == 1)[0]
    ineg = np.where(target == 0)[0]
    nmask = 1

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']


    fig = plt.figure()
    fig.set_size_inches(3*nmask, 7)

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
    _, ext = fileName.rsplit(".")
    filePath = os.path.join(pict_path, fileName)
    fig.savefig(filePath, format=ext, dpi=300, bbox_inches='tight',
                pad_inches=0.03)
    plt.show()
    plt.close()


def postprocessDataset(fileName, config=None):
    """ Selective visualization of the results. """

    if config is None:
        config = {
            "oxy": {"mask": "wound"},
            "nir": {"mask": "wound"},
            "thi": {"mask": "proximity"},
            "twi": {"mask": "wound"},
        }

    start = timer()
    filePath = os.path.join(data_path, fileName)
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
                features[key][i] = np.mean(values)
                # features[key][i] = np.percentile(values, q=50)

        print(patients)
        print(features)

        keys1 = ["oxy", "oxy", "oxy"]
        keys2 = ["nir", "thi", "twi"]
        target = patients["target"]

        baseName, ext = fileName.rsplit(".")
        plotMultipleXY(baseName+".png", features, target, keys1, keys2)

        print("\nElapsed time for preparing results: %f sec" %
              (timer() - start))



def main():
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    fileName = "test.h5"
    filePath = os.path.join(data_path, fileName)

    createDataset(filePath, path="/records", descr=__doc__)
    processDataset(filePath)
    postprocessDataset(filePath)



if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
