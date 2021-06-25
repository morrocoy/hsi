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
if sys.version_info[0] < 3 or (
        sys.version_info[0] == 3 and sys.version_info[1] < 6):
    raise Exception(
        "Pyqtgraph requires Python version 2.6 or greater (this is %d.%d)" % (
            sys.version_info[0], sys.version_info[1]))

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
# import numpy

try:
    import pyqtgraph
except ModuleNotFoundError as err:
    CONFIG_OPTIONS['enableGUI'] = False
    print("Module pyqtgraph was not found. Graphics items and widgets are "
          "disabled.")

try:
    import bvls
except ModuleNotFoundError as err:
    CONFIG_OPTIONS['enableBVLS'] = False
    print("Module bvls was not found. "
          "Using instead scipy.optimize.lsq_linear().")

# color maps
from .core.hs_cm import cm

# spectral formats
from .core.hs_formats import HSFormatFlag, HSFormatDefault, convert
from .core.hs_formats import HSIntensity, HSAbsorption
from .core.hs_formats import HSExtinction, HSRefraction

# hyperspectral data representation
from .core.hs_image import HSImage
from .core.hs_component import HSComponent

# file io
from .core.hs_file import HSFile
from .core.hs_component_file import HSComponentFile
from .core.hs_store import HSStore, HSPatientInfo
from .core.hs_tivita_store import HSTivitaStore

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
    # miscellaneous
    "genHash",
]


def system_info():
    print("sys.platform: %s" % sys.platform)
    print("sys.version: %s" % sys.version)
    from pyqtgraph.Qt import VERSION_INFO
    print("qt bindings: %s" % VERSION_INFO)

    global __version__
    rev = None
    if __version__ is None:
        # this code was probably checked out from bzr; look up the last
        # revision file
        last_rev_file = os.path.join(
            os.path.dirname(__file__), '..', '.bzr', 'branch', 'last-revision')
        if os.path.exists(last_rev_file):
            with open(last_rev_file, 'r') as fd:
                rev = fd.read().strip()

    print("hsi: %s; %s" % (__version__, rev))
    print("config:")
    import pprint
    pprint.pprint(CONFIG_OPTIONS)
