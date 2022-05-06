""" hsi. A hyper spectral image analysis package
"""
import os.path
import sys

from .hs_base_analysis import HSBaseAnalysis
from .hs_component_fit import HSComponentFit
from .hs_open_tivita import HSOpenTivita

# TODO clean up Moussa's files before consider freezing
if not getattr(sys, 'frozen', False):
    from .hs_lipids import HSLipids
    from .hs_blood_vessel import HSBloodVessel

__all__ = [
    "HSBaseAnalysis",
    "HSComponentFit",
    "HSOpenTivita",
]

# TODO clean up Moussa's files before consider freezing
if not getattr(sys, 'frozen', False):
    __all__ += ["HSLipids", "HSBloodVessel"]

if getattr(sys, 'frozen', True) or os.path.isfile(
        os.path.join(os.path.dirname(__file__), "hs_tivita.py")):
    from .hs_tivita import HSTivita
    __all__.append("HSTivita")

