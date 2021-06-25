""" Miscellaneous routines
"""
# from __future__ import print_function, division

import os.path
# import xlrd
import numpy as np
import hashlib

import logging

DEBUGGING = False
LOG_FMT = '%(asctime)s %(filename)35s:%(lineno)-4d : %(levelname)-7s: %(message)s'

logger = logging.getLogger(__name__)


def versionStrToTuple(versionStr):
    """ Converts a version string to tuple

        E.g. 'x.y.z' to (x, y, x)
    """
    versionInfo = []
    for elem in versionStr.split('.'):
        try:
            versionInfo.append(int(elem))
        except:
            versionInfo.append(elem)
    return tuple(versionInfo)


def is_an_array(var, allow_none=False):
    """ Returns True if var is a numpy array.
    """
    return isinstance(var, np.ndarray) or (var is None and allow_none)


def check_is_an_array(var, allow_none=False):
    """ Calls is_an_array and raises a TypeError if the check fails.
    """
    if not is_an_array(var, allow_none=allow_none):
        raise TypeError("var must be a NumPy array, however type(var) is {}"
                        .format(type(var)))


def check_class(var, cls, allowNone=False):
    """ Checks if a variable is an instance of the cls class, raises TypeError if the check fails.
    """
    if not isinstance(var, cls) and not (allowNone and var is None):
        raise TypeError("Unexpected type {}, was expecting {}".format(type(var), cls))


def getPkgDir():
    # return os.path.join(os.path.dirname(__file__), os.pardir)
    return os.path.dirname(__file__)


def genHash(obj):
    m = hashlib.md5()
    # m = hashlib.sha1()

    if isinstance(obj, np.ndarray):
        ndim = obj.ndim
        shape = obj.shape
        if ndim == 1:
            arr = obj[:]
        elif ndim == 2:
            arr = np.sum(obj, axis=0)
        else:
            arr = np.sum(obj.reshape((shape[0], -1)), axis=0)

        m.update(arr.tostring())

    elif isinstance(obj, str):
        m.update(obj.encode('utf-8'))

    else:
        m.update(obj)

    return m.hexdigest()


# excel stuff .................................................................
# def read_excel(file, sheet, rows, cols, zerochars=[], nanchars=[]):
#
#     workbook = xlrd.open_workbook(file)
#
#     if isinstance(sheet, int):
#         worksheet = workbook.sheet_by_index(sheet)
#     else:
#         worksheet = workbook.sheet_by_name(sheet)
#
#     data = []
#     for i in (range(len(rows))):
#         line = []
#         if worksheet.nrows <= rows[i]:
#             break
#         for j in range(len(cols)):
#             readout = worksheet.cell_value(rows[i], cols[j])
#             # check for number
#             if isinstance(readout, (int, float)):
#                 if readout == int(readout):
#                     line.append(int(readout))
#                 else:
#                     line.append(readout)
#
#             # check for characters which shall be associated with zero
#             elif readout in zerochars:
#                 line.append(0)
#
#             # check for characters which shall be associated with zero
#             elif readout in nanchars:
#                 line.append(np.nan)
#
#             # append readout as string
#             else:
#                 line.append(readout)
#         data.append(line)
#     return data


# QT stuff ...................................................................
def clearLayout(layout):
    """Clear all widgets in a layout.

    """
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()
            # child.widget().setParent(None)