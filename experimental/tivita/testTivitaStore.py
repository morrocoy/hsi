# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:03:42 2021

@author: kai
"""
import sys
import os.path
import logging
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from tables_utils import getDirPaths, loadPatientData, loadHSData
import hsi
from hsi import HSIntensity, HSTivitaStore
from hsi.log import logmanager


# windows system
if sys.platform == "win32":
    _rootPath = os.path.join(
        "d:", os.sep, "projects", "hyperlimit", "amputation",
        "studies", "rostock_suedstadt_2018-2020")

# linux system
else:
    _rootPath = os.path.join(
        os.path.expanduser("~"), "projects", "hyperlimit", "amputation",
        "studies", "rostock_suedstadt_2018-2020")


def getDirPaths():
    return {
        'root': _rootPath,
        'data': os.path.join(_rootPath, "data"),
        'pictures': os.path.join(_rootPath, "pictures"),
        'results': os.path.join(_rootPath, "results"),
    }

# define columns for the patient info table
HSPatientInfo = np.dtype([
    ("pn", '<i8'),
    ("pid", '<i8'),
    ("name", 'S32'),
    ("descr", 'S64'),
    ("timestamp", 'S32'),
    ("target", '<i4'),
    # ("path", 'S32'),
])


logger = logmanager.getLogger(__name__)

dirPaths = getDirPaths()


# def main():

# fileName = "181022_rostock_suedstadt.xlsx"
# filePath = os.path.join(data_path, fileName)

# load metadata
# fileName = "181022_Resektionsgrenze_Südstadt_Auswertung_27.10.2020.xlsx"
# filePath = os.path.join(dirPaths['results'], fileName)
# project = "rostock"
# hsformat = HSIntensity
#
# patientData = loadPatientData(filePath, sheet_name=0, skiprows=1)
# patientData['hsformat'] = hsformat.key

# create output file
start = timer()

fileName = "rostock_suedstadt_2018-2020.xlsx"
# fileName = "rostock_suedstadt_2018-2020_4_test.h5"
filePath = os.path.join(dirPaths['data'], fileName)

print(filePath)
with HSTivitaStore.open(filePath, path="/") as store:
    store.attacheTable(
        name="patient",
        dtype=HSPatientInfo,
        sheet_name=0,
        usecols=[3,4,5,6,8,7],
        skiprows=1,
    )
    
    table = store.getTable("patient")
    print(len(store))
    
    patient, hsimage, masks = store[0]

# df.astype(HSPatientInfo)
# df.astype(HSPatientInfo)
# df[0]["pid"]
# df.iloc[0]["name"]

# df.astype(HSPatientInfo)
# df.to_numpy()
#
# xl = pd.ExcelFile(filePath)
# df = xl.parse(0)  # sheet name


# if __name__ == '__main__':
#     if __name__ == '__main__':
#         logmanager.setLevel(logging.DEBUG)
#         logger.info("Python executable: {}".format(sys.executable))
#         logger.info("Python hsi version: {}".format(hsi.__version__))
#         main()
