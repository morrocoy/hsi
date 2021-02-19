""" Miscellaneous routines
"""
# from __future__ import print_function, division

import os.path
import sys
import re



import numpy as np

import logging


DEBUGGING = False
LOG_FMT = '%(asctime)s %(filename)35s:%(lineno)-4d : %(levelname)-7s: %(message)s'

logger = logging.getLogger(__name__)


# # Reads the version of the program from the first line of version.txt
# try:
#     if getattr(sys, 'frozen', False):
#         # If the application is run as a bundle, the pyInstaller bootloader
#         # extends the sys module by a flag frozen=True and sets the app
#         # path into variable _MEIPASS'.
#         MODULE_DIR = os.path.join(sys._MEIPASS, 'cmlib')
#         if DEBUGGING:
#             print("Module dir from meipass: {}".format(MODULE_DIR), file=sys.stderr)
#     else:
#         MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
#         if DEBUGGING:
#             print("Module dir from __file__: {}".format(MODULE_DIR), file=sys.stderr)
#
#     VERSION_FILE = os.path.join(MODULE_DIR, 'version.txt')
#     if DEBUGGING:
#         print("Reading version from: {}".format(VERSION_FILE), file=sys.stderr)
#     logger.debug("Reading version from: {}".format(VERSION_FILE))
#     with open(VERSION_FILE) as stream:
#         __version__ = stream.readline().strip()
# except Exception as ex:
#     __version__ = "?.?.?"
#     logger.error("Unable to read version number: {}".format(ex))
#     raise

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



# QT stuff ...................................................................
def clearLayout(layout):
    """Clear all widgets in a layout.

    """
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()
            # child.widget().setParent(None)