""" hsi. A hyper spectral image analysis package
"""

# from ..misc import __version__

# from .SpectralTissueCompound import SpectralTissueCompound

from .HSVector import HSVector
from .HSVectorAnalysis import HSVectorAnalysis
from .HSVectorFile import HSVectorFile
from .HSTivitaAnalysis import HSTivitaAnalysis

__all__ = [
    "HSVector", "HSVectorAnalysis", "HSVectorFile",
    "HSTivitaAnalysis",
    # "SpectralTissueCompound",
]



