# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 10:38:20 2021

@author: kpapke
"""
import sys
import os.path
from timeit import default_timer as timer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from hsi import cm, HSIntensity, HSAbsorption
from hsi import HSImage
from hsi import genHash

# windows system
if sys.platform == "win32":
    _rootPath = os.path.join(
        "d:", os.sep, "projects", "hyperlimit", "amputation",
        "data", "Rostock_Suedstadt_2018-2020")

# linux system
else:
    _rootPath = os.path.join(
        os.path.expanduser("~"), "projects", "hyperlimit", "amputation",
        "data", "Rostock_Suedstadt_2018-2020")

cmap = cm.tivita()

fig_options = {
    'dpi': 300, # resolution
    'pil_kwargs': {
        'quality': 60,  # for jpg
        'optimize': True,  # for jpg
    },
    'bbox_inches': 'tight',
    'pad_inches': 0.03,
}


def getDirPaths():
    return {
        'root': _rootPath,
        'data': os.path.join(_rootPath, "data"),
        'pictures': os.path.join(_rootPath, "pictures"),
        'results': os.path.join(_rootPath, "results"),
    }


def loadHSData(pathName, baseName, hsformat):
    """ Load hyperspectral data for a record.
    """

    filePath = os.path.join(pathName, baseName + "_SpecCube.dat")
    hsImage = HSImage(filePath)
    nwavelen, rows, cols = hsImage.shape

    hsImage.setFormat(hsformat)

    # add gaussian image filter for a cleaner tissue selection mask
    hsImage.addFilter(mode='image', type='gauss', sigma=1, truncate=4)

    # tissue and selection masks
    mask_labels = [
        "tissue",
        "critical wound region",
        "wound region",
        "wound and proximity",
        "wound proximity"
    ]

    nmask = 5
    masks = np.zeros((nmask, rows, cols), dtype=np.int8)
    masks[0, ...] = hsImage.getTissueMask([0.1, 0.9])  # tissue mask

    filePath = os.path.join(pathName, baseName + "_Masks.npz")
    maskfile = np.load(filePath)
    for imask in range(1, 4):
        key = "mask%d" % imask
        masks[imask] = maskfile[key] * masks[0]
    masks[4] = masks[3] * (1 - masks[2])  # wound proximity

    return hsImage, masks


def loadPatientData(filePath, sheet_name=0, columns=None,
                    skiprows=None, nrows=None):
    """ Load meta data of all records.

    Parameters
    ----------
    filePath : str
        The path to the excel file.
    sheet_name : int or str, optional
        The index or name of the worksheet within the excel file.
    columns : list of str, optional
        A list of column names.
    skiprows : int, optional
        The number of rows to skip.
    nrows : int
        The number of rows to read.
    """
    columns = None
    if columns is None:
        columns = {
            "pn": {"index": 3, "dtype": np.int64},  # patient number
            "pid": {"index": 4, "dtype": np.int64},  # patient id
            "descr": {"index": 6, "dtype": str},  # details of the wound
            "target": {"index": 7, "dtype": int},  # healed or not
            "timestamp": {"index": 8, "dtype": str},  # ref path within h5 file
        }

    names = columns.keys()
    usecols = [col["index"] for col in columns.values()]
    converters = {key: col["dtype"] for key, col in columns.items()}

    df = pd.read_excel(
        filePath, sheet_name=sheet_name, names=names,
        usecols=usecols, skiprows=skiprows, nrows=nrows, converters=converters)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.astype(converters)

    # retrieve timestamp from group name
    # df['timestamp'] = pd.to_datetime(df['group'], format="%Y-%m-%d-%H-%M-%S")

    return df


def plotMasks(fileName, masks, image=None):
    """Create masked plots of the original rgb image.

    Parameters
    ----------
    filepath : str,
        The full path to the output file.
    image : np.ndarray
        The rgb image.
    mask :  dict
        A dictionarry of the mask arrays.
    """
    fig = plt.figure()
    fig.set_size_inches(10, 8)

    d, m =  divmod(len(masks), 2)
    cols = d
    rows = d + m

    if isinstance(masks, dict):
        keys = masks.keys()
    elif isinstance(masks, np.void):
        keys = masks.dtype.names
    else:
        raise ValueError("Mask must be either dict or numpy.void")

    if image is None:
        for i, key in enumerate(keys):
            ax = fig.add_subplot(cols, rows, i + 1)
            plt.imshow(masks[key], cmap="gray", vmin=0, vmax=1)
            ax.set_title(key)

    else:
        for i, key in enumerate(keys):
            mimage = image.copy()
            red = mimage[:, :, 0]
            green = mimage[:, :, 1]
            blue = mimage[:, :, 2]

            idx = np.nonzero(masks[key] == 0)  # gray out region out of mask
            gray = 0.2989 * red[idx] + 0.5870 * green[idx] + 0.1140 * blue[idx]
            red[idx] = gray
            green[idx] = gray
            blue[idx] = gray

            ax = fig.add_subplot(cols, rows, i+1)
            plt.imshow(mimage)
            ax.set_title(key)

    ext = fileName[fileName.rfind('.')+1:]
    filepath = os.path.join(_rootPath, "pictures", fileName)

    fig.savefig(filepath, format=ext, **fig_options)
    # plt.show()
    plt.close()



def plotParam(fileName, param):
    """Create masked plots of the original rgb image.

    Parameters
    ----------
    filepath : str,
        The full path to the output file.
    param :  dict
        A dictionarry of the parameter arrays.
    """
    if isinstance(param, dict):
        keys = param.keys()
    elif isinstance(param, np.void):
        keys = param.dtype.names
    else:
        raise ValueError("Mask must be either dict or numpy.void")

    fig = plt.figure()
    fig.set_size_inches(10, 8)

    d, m =  divmod(len(param), 2)
    cols = d
    rows = d + m
    for i, key in enumerate(keys):
        ax = fig.add_subplot(cols, rows, i + 1)

        pos = plt.imshow(param[key], cmap=cmap, vmin=0, vmax=1)
        fig.colorbar(pos, ax=ax)
        ax.set_title(key)

    ext = fileName[fileName.rfind('.')+1:]
    filepath = os.path.join(_rootPath, "pictures", fileName)

    fig.savefig(filepath, format=ext, **fig_options)
    # plt.show()
    plt.close()
