""" hsi. A hyper spectral image analysis package
"""
import os.path

from .HSComponentFit import HSComponentFit
from .HSOpenTivita import HSOpenTivita
from .HSBaseStudy import HSBaseStudy

__all__ = [
    "HSBaseStudy",
    "HSComponentFit",
    "HSOpenTivita",
]

if os.path.join(os.path.dirname(__file__), "HSTivita.py"):
    from .HSTivita import HSTivita
    __all__.append("HSTivita")

