# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 12:03:35 2021

@author: kpapke

.. _rostock_suedstadt_2018-2020_dataset:

Amputation data from the hospital of Rostock Suedstadt between 2018 and 2020
----------------------------------------------------------------------------

**Data Set Characteristics:**

    :Number of Instances: 65 (44 of healing and 21 of not healing)
    :Number of Attributes: 7 meta, 3 numeric, predictive and 2 class attributes
    :Attribute Information:
        - metadata:
            - pn: Patient number
            - pid: Patient ID
            - descr: Description of the wound
            - timestamp: Timestamp as string
            - hsformat: Spectral hsformat
            - hash: checksum of the spectral data using md5 hash
        - numeric predictive:
            - hsidata: Hyperspectral data
            - wavelen: Wavelengths at which the spectral information is sampled
            - masks: Selection masks applied on the hyperspectral images
        - class (target):
            - not healed with target index 0
            - healed with target index 1
"""
import os.path
from timeit import default_timer as timer

import pandas as pd
import numpy as np
import tables

from tables_utils import getDirPaths, loadPatientData, loadHSData
from hsi import HSIntensity, HSAbsorption


class Patient(tables.IsDescription):
    pn = tables.Int64Col()  # Signed 64-bit integer
    pid = tables.Int64Col()  # Signed 64-bit integer
    name = tables.StringCol(32)  # 32-byte character string (utf-8)
    descr = tables.StringCol(64)  # 64-byte character String (utf-8)
    timestamp = tables.StringCol(32)  # 32-byte character String (utf-8)
    hsformat = tables.StringCol(32)  # 32-byte character String (utf-8)
    # wavelen = tables.Float64Col(shape=(100, ))  # array of 64-bit floats
    # hsidata = tables.Float32Col(shape=(100, 480, 640))  # array of 32-bit floats
    # mask = tables.Int8Col(shape=(5, 480, 640))  # array of bytes
    target = tables.Int32Col()  # Signed 32-bit integer

class HSImageData(tables.IsDescription):
    wavelen = tables.Float64Col(shape=(100, ))  # array of 64-bit floats
    hsidata = tables.Float32Col(shape=(100, 480, 640))  # array of 32-bit floats
    masks = tables.Int8Col(shape=(5, 480, 640))  # array of bytes

# Patient = np.dtype([
#     ("index", np.int64),  # Signed 64-bit integer
#     ("id", np.int64),  # Signed 64-bit integer
#     ("name", "S16"),  # 16-byte character string
#     ("descr", "S32"),  # 32-byte character string
#     ("timestamp", "S32"),  # 32-byte character string
#     ("hsformat", "S16"),  # 16-byte character string
#     ("wavelen", np.float64, (100, )),  # array of floats (single-precision)
#     ("hsi", np.float32, (100, 640, 480)),  # array of floats (single-precision)
#     ("mask", np.int8, (4, 640, 480))  # array of bytes
#     ("target", np.int32),  # Signed 32-bit integer
#     ])



dirPaths = getDirPaths()


# load metadata ..............................................................
fileName = "181022_Resektionsgrenze_Südstadt_Auswertung_27.10.2020.xlsx"
filePath = os.path.join(dirPaths['results'], fileName)
project = "rostock"
hsformat = HSIntensity

patientData = loadPatientData(filePath, sheet_name=0, skiprows=1)
patientData['hsformat'] = hsformat.key
# dfHSImages = dfMetadata.apply(lambda x: loadHSImage(x['timestamp']), axis=1)
# dfData = dfMetadata['timestamp'].apply(load_hsdata)

npatient = len(patientData.index)
# npatient = 100
nwavelen = 100
nmask = 5
nrow = 480
ncol = 640

# create output file .........................................................
fileName = "rostock_suedstadt_2018-2020_2.h5"
h5file = tables.open_file(os.path.join(dirPaths['data'], fileName), mode="w")

# Create a new group
grecords = h5file.create_group("/", "records", "Clinical records")
grecords._v_attrs.descr = __doc__  # description of dataset

# Creating a new table
patientTable = h5file.create_table(grecords, "patient", Patient, "Patient information")
hsimageTable = h5file.create_table(
    grecords,
    name="hsidata",
    description=HSImageData,
    title="Hyperspectral image data",
    expectedrows=npatient,
    # chunkshape=None,
)
# ghsidata = h5file.create_group("/records", "hsidata", "Hyperspectral image data")




# spectraArray = h5file.create_carray(
#     ghsidata,
#     name="spectra",
#     # atom=tables.Float32Atom(shape=(100, 480, 640)),
#     # atom=tables.Float32Atom(),
#     # atom=tables.Float32Col(),
#     # atom=tables.Float32Col(),
#     # atom=tables.Float32Col(shape=(nwavelen, nrow, ncol)),
#     atom=tables.Float32Atom(shape=(nrow, ncol)),
#     # shape=(npatient, nwavelen, nrow, ncol),
#     # shape=(npatient, ),
#     shape=(npatient, nwavelen),
#     title="Spectral data",
#     # expectedrows=100,
#     # chunkshape=(1, 100, 480, 640),
#     # chunkshape=(1, ),
#     chunkshape=(1, nwavelen),
#     byteorder="little",
# )
#
# wavelenArray = h5file.create_carray(
#     ghsidata,
#     name="wavelen",
#     # atom=tables.Float32Atom(shape=(100, 480, 640)),
#     atom=tables.Float64Atom(),
#     shape=(npatient, nwavelen),
#     title="Wavelength samples",
#     # chunkshape=(1, 100, 480, 640),
#     chunkshape=(1, nwavelen),
#     byteorder="little",
# )
#
# masksArray = h5file.create_carray(
#     ghsidata,
#     name="masks",
#     # atom=tables.Float32Atom(shape=(100, 480, 640)),
#     atom=tables.Int8Col(shape=(nrow, ncol)),
#     shape=(npatient, nmask),
#     title="Masks",
#     chunkshape=(1, nmask),
#     # chunkshape=(1, ),
#     byteorder="little",
# )

# spectra = h5file.create_earray(
#     ghsidata,
#     name="spectra",
#     # atom=tables.Float32Atom(shape=(100, 480, 640)),
#     atom=tables.Float32Atom(),
#     shape=(0, 100, 480, 640),
#     # shape=(0, ),
#     title="Name column selection",
#     expectedrows=100,
#     # chunkshape=(100, 480, 640),
#     # chunkshape=(10, ),
#     byteorder="little",
# )

# fill table
patientEntry = patientTable.row  # get a pointer to the Row
hsimageEntry = hsimageTable.row
for index, entry in patientData.iterrows():
    if index > 2:
        break

    print("Process index %d ..." % entry["pn"])

    patientEntry["pn"] = entry["pn"]  # i f'Particle: {i:6d}'
    patientEntry["pid"] = entry["pid"]
    patientEntry["name"] = str.encode("")
    patientEntry["descr"] = str.encode(entry["descr"])
    patientEntry["timestamp"] = str.encode(entry["timestamp"])
    patientEntry["hsformat"] = str.encode(hsformat.key)
    patientEntry["target"] = entry["target"]

    # writes new particle record to the table I/O buffer
    patientEntry.append()

    baseName = entry["timestamp"].replace('-', '_')
    pathName = os.path.join(dirPaths['data'], baseName)
    hsidata, masks = loadHSData(pathName, baseName, hsformat)

    # spectraArray[index] = hsidata.spectra.astype(np.float32)  # float32, little
    # wavelenArray[index] = hsidata.wavelen.astype(np.float64)  # float64, little
    # masksArray[index] = masks.astype(np.int8)  # singned byte, little

    hsimageEntry["hsidata"] = hsidata.spectra.astype(np.float32)  # float32, little
    hsimageEntry["wavelen"] = hsidata.wavelen.astype(np.float64)  # float64, little
    hsimageEntry["masks"] = masks.astype(np.int8)  # singned byte, little
    hsimageEntry.append()

# flush the table’s I/O buffer to write all this data to disk
patientTable.flush()
hsimageTable.flush()

# Finally, close the file (this also will flush all the remaining buffers!)
h5file.close()


# a = dfMetadata.iloc[0]["timestamp"]

