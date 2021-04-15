""" hsi. A hyper spectral image analysis package
"""
# from __future__ import absolute_import

import os.path
import sys

# __all__ = [""]
# print(f'Invoking __init__.py for {__name__}')

    
# Reads the version of the program from the first line of version.txt
try:
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the pyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app
        # path into variable _MEIPASS'.
        MODULE_DIR = os.path.join(sys._MEIPASS, 'cmlib')
    else:
        MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

    VERSION_FILE = os.path.join(MODULE_DIR, 'version.txt')
    with open(VERSION_FILE) as stream:
        __version__ = stream.readline().strip()
except Exception as ex:
    __version__ = "?.?.?"
    raise
              
# check python version (Allow anything >= 3.6)
if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
    raise Exception("Pyqtgraph requires Python version 2.6 or greater (this is %d.%d)" % (sys.version_info[0], sys.version_info[1]))
    

CONFIG_OPTIONS = {
    'enableGUI': True,  # provide all gui interfaces
    'enableBVLS': True,  # provide fortran bvls implementation
    'enableExperimental': True,  # Enable experimental features
    'imageAxisOrder': 'row-major',  # For 'row-major', image data is expected
                                    # in the standard (row, col) order.
                                    # For 'col-major', image data is expected
                                    # in reversed (col, row) order.
}


# requires packages (import here to check if they are available)
import numpy

try:
    import pyqtgraph
except ModuleNotFoundError as err:
    CONFIG_OPTIONS['enableGUI'] = False
    print("Module pyqtgraph was not found. Graphics items and widgets are disabled.")

try:
    import bvls
except ModuleNotFoundError as err:
    CONFIG_OPTIONS['enableBVLS'] = False
    print("Module bvls was not found. Using instead scipy.optimize.lsq_linear().")


def systemInfo():
    print("sys.platform: %s" % sys.platform)
    print("sys.version: %s" % sys.version)
    from pyqtgraph.Qt import VERSION_INFO
    print("qt bindings: %s" % VERSION_INFO)
    
    global __version__
    rev = None
    if __version__ is None:  ## this code was probably checked out from bzr; look up the last-revision file
        lastRevFile = os.path.join(os.path.dirname(__file__), '..', '.bzr', 'branch', 'last-revision')
        if os.path.exists(lastRevFile):
            with open(lastRevFile, 'r') as fd:
                rev = fd.read().strip()
    
    print("hsi: %s; %s" % (__version__, rev))
    print("config:")
    import pprint
    pprint.pprint(CONFIG_OPTIONS)


# color maps
from .core.cm import cm

# spectral formats
from .core.formats import HSFormatFlag, HSFormatDefault, convert
from .core.formats import HSIntensity, HSAbsorption, HSExtinction, HSRefraction

# hyperspectral data representation
from .core.HSImage import HSImage
from .core.HSComponent import HSComponent

# File IO
from .core.HSFile import HSFile
from .core.HSComponentFile import HSComponentFile
from .core.HSStore import HSStore, HSPatientInfo
from .core.HSTivitaStore import HSTivitaStore


# tissues
from .tissue.HSTissueComponent import HSTissueComponent
from .tissue.HSTissueCompound import HSTissueCompound

# miscellaneous
from .misc import genHash

__all__ = [
    # color maps
    "cm",
    # spectral formats
    "HSFormatFlag", "HSFormatDefault", "convert",
    "HSIntensity", "HSAbsorption", "HSExtinction", "HSRefraction",
    # hyperspectral data representation
    "HSImage", "HSComponent",
    # File IO
    "HSFile", "HSComponentFile", "HSStore", "HSTivitaStore", "HSPatientInfo",
    # tissues
    "HSTissueComponent", "HSTissueCompound",
    # miscellaneous
    "genHash",
]