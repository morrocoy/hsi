""" hsi. A hyper spectral image analysis package
"""
import os.path

from .hs_base_analysis import HSBaseAnalysis
from .hs_component_fit import HSComponentFit
from .hs_open_tivita import HSOpenTivita

__all__ = [
    "HSBaseAnalysis",
    "HSComponentFit",
    "HSOpenTivita",
]

if os.path.isfile(os.path.join(os.path.dirname(__file__),
                               "hs_tivita.py")):
    from .hs_tivita import HSTivita
    __all__.append("HSTivita")

