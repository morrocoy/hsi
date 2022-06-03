""" hsi. A hyper spectral image analysis package
"""
import os.path
import sys

from .hs_base_analysis import HSBaseAnalysis
from .hs_cofit import HSCoFit
# from .hs_cofit_sequence import HSCoFitSequence
from .hs_open_tivita import HSOpenTivita
from .hs_lipids import HSLipids
from .hs_blood_vessel import HSBloodVessel

__all__ = [
    "HSBaseAnalysis",
    "HSCoFit",
    "HSLipids",
    "HSBloodVessel",
    "HSOpenTivita",
]

if getattr(sys, 'frozen', True) or os.path.isfile(
        os.path.join(os.path.dirname(__file__), "hs_tivita.py")):
    from .hs_tivita import HSTivita
    __all__.append("HSTivita")

