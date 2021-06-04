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
import seaborn as sns

import hsi
from hsi import HSImage, HSFormatFlag, HSIntensity, HSAbsorption
from hsi import HSStore
from hsi.analysis import HSOpenTivita #, HSTivita
from hsi.log import logmanager

logger = logmanager.getLogger(__name__)

data_path = os.path.join(os.getcwd(), "..", "data")

# define columns for the patient info table
HSPatientInfo = np.dtype([
    ("pn", '<i8'),
    ("pid", '<i8'),
    ("name", 'S32'),
    ("descr", 'S64'),
    ("timestamp", 'S32'),
    ("target", '<i4'),
])


def createDataset(file_name, path, descr=None):
    """create dummy dataset consisting of three tables for patient data,
    hyperspectral image data and selection masks.

    Parameters
    ----------
    file_name : pd.Series
        The path to the hdf5 output file.
    path :  str
        The path within the hdf5 file.
    descr :  str, optional
        A description for the dataset.
    """
    patient_data = pd.DataFrame({
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
    filePath = os.path.join(data_path, file_name)
    with HSStore.open(
            filePath, mode="w", path=path, descr=descr) as dataset:
        rows = len(patient_data.index)

        patient_table = dataset.createTable(
            name="patient",
            dtype=HSPatientInfo,
            title="Patient information",
            expectedrows=rows,
        )

        hsidata_table = dataset.createTable(
            name="hsidata",
            dtype=np.dtype([
                ("hsformat", "<S32"),
                ("wavelen", "<f8", (100,)),
                ("spectra", "<f4", (100, 480, 640))
            ]),
            title="Hyperspectral image data",
            expectedrows=rows,
        )

        masks_table = dataset.createTable(
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

        patient_entry = patient_table.row
        hsidata_entry = hsidata_table.row
        masks_entry = masks_table.row
        for index, info in patient_data.iterrows():

            print("Process index %d ..." % info["pn"])

            patient_entry["pn"] = info["pn"]  # i f'Particle: {i:6d}'
            patient_entry["pid"] = info["pid"]
            patient_entry["name"] = str.encode(info["name"])
            patient_entry["descr"] = str.encode(info["descr"])
            patient_entry["timestamp"] = str.encode(info["timestamp"])
            patient_entry["target"] = info["target"]
            patient_entry.append()


            hsidata_entry["hsformat"] = str.encode(HSIntensity.key)
            hsidata_entry["wavelen"] = wavelen.astype("<f8")
            hsidata_entry["spectra"] = np.random.random(
                (100, 480, 640)).astype("<f4")
            hsidata_entry.append()

            masks_entry["tissue"] = np.random.randint(
                2, size=(480, 640), dtype="<i1")
            masks_entry["critical"] = np.random.randint(
                2, size=(480, 640), dtype="<i1")
            masks_entry["wound"] = np.random.randint(
                2, size=(480, 640), dtype="<i1")
            masks_entry["proximity"] = np.random.randint(
                2, size=(480, 640), dtype="<i1")
            masks_entry.append()

        patient_table.flush()
        hsidata_table.flush()
        masks_table.flush()

        print("\nElapsed time for creating dataset: %f sec" % (timer() - start))


def processDataset(file_name):
    """ Evaluate the tivita index values for each record.

    The evaluation is defined in task(args) where args are the entries of the
    attached tables.
    """
    start = timer()
    file_path = os.path.join(data_path, file_name)
    with tables.open_file(file_path, "r+") as file:
        reader = HSStore(file, path="/records")
        writer = HSStore(file, path="/records")

        reader.attacheTable("patient")
        reader.attacheTable("hsidata")
        reader.attacheTable("masks")

        tivita_table = writer.createTable(
            name="tivita",
            dtype=np.dtype([
                ("oxy", "<f8", (480, 640)),
                ("nir", "<f8", (480, 640)),
                ("thi", "<f8", (480, 640)),
                ("twi", "<f8", (480, 640)),
            ]),
            title="Tivita Index Values",
            expectedrows=len(reader),
        )
        tivita_entry = tivita_table.row

        print(f"Tables to read: {reader.getTableNames()}")
        print(f"Tables to write: {writer.getTableNames()}")
        print(f"Number of entries: {len(reader)}")

        # serial evaluation
        # for args in iter(reader):
        #     param = task(args)
        #     tivita_entry["nir"] = param["nir"]
        #     tivita_entry["oxy"] = param["oxy"]
        #     tivita_entry["thi"] = param["thi"]
        #     tivita_entry["twi"] = param["twi"]
        #     tivita_entry.append()
        #
        # tivita_table.flush()

        # parallel evaluation
        pool = multiprocessing.Pool(processes=7)
        for param in pool.imap(task, iter(reader)):  # , chunksize=1):
            tivita_entry["oxy"] = param["oxy"]
            tivita_entry["nir"] = param["nir"]
            tivita_entry["thi"] = param["thi"]
            tivita_entry["twi"] = param["twi"]
            tivita_entry.append()
        pool.close()

        tivita_table.flush()

        print("\nElapsed time for processing dataset: %f sec" %
              (timer() - start))


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

    analysis = HSOpenTivita(format=HSAbsorption)  # open source algorithms
    # analysis = HSTivita(format=HSIntensity)  # true algorithms

    analysis.setData(hsImage.spectra, hsImage.wavelen, format=hsformat)
    analysis.evaluate(mask=masks["tissue"])
    param = analysis.getSolution(unpack=True, clip=True)

    return param


def postprocessDataset(file_name, mask_config=None):
    """ Selective visualization of the results. """

    if mask_config is None:
        mask_config = {
            "oxy": {"mask": "wound"},
            "nir": {"mask": "wound"},
            "thi": {"mask": "proximity"},
            "twi": {"mask": "wound"},
        }

    start = timer()
    file_path = os.path.join(data_path, file_name)
    with HSStore.open(file_path, mode="r", path="/records") as reader:
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
            {key: np.zeros(len(reader)) for key in mask_config.keys()})

        # fill features by evaluating tivita index values for each record
        for i, (patient, masks, params) in enumerate(reader):
            print("%8d | %8d | %-20s | %-20s | %3d |" % (
                patient["pn"],
                patient["pid"],
                patient["descr"].decode(),
                patient["timestamp"].decode(),
                patient["target"]
            ))

            # evaluate average of parameters over selection mask
            for key, val in mask_config.items():
                p = params[key].reshape(-1)
                mask = masks[val["mask"]]
                values = p[mask.reshape(-1) == 1]
                features[key][i] = np.mean(values)

        print(patients)
        print(features)

        # plot analysis results
        data = pd.DataFrame.from_dict({**patients, **features})
        sns.pairplot(
            data, hue="target", x_vars=["oxy"], y_vars=["nir", "thi", "twi"],
            height=3, aspect=1.2,
        )
        plt.show()


        print("\nElapsed time for preparing results: %f sec" %
              (timer() - start))



def main():
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    file_name = "hsstore_test.h5"
    file_path = os.path.join(data_path, file_name)

    createDataset(file_path, path="/records", descr=__doc__)
    processDataset(file_path)
    postprocessDataset(file_path)

    os.unlink(file_path)



if __name__ == '__main__':
    logmanager.setLevel(logging.DEBUG)
    logger.info("Python executable: {}".format(sys.executable))
    logger.info("Python hsi version: {}".format(hsi.__version__))

    main()
