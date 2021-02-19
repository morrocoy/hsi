""" hsi. A hyper spectral image analysis package
"""

# from ..misc import __version__

from .. import CONFIG_OPTIONS

if CONFIG_OPTIONS['enableGUI']:

    from .graphicsItems.InfiniteLine import InfiniteLine, InfLineLabel
    from .graphicsItems.ColorBarItem import ColorBarItem

    from .graphicsItems.RegnPlotCtrlItem import RegnPlotCtrlItem
    from .graphicsItems.BaseImagCtrlItem import BaseImagCtrlItem
    from .graphicsItems.HistImagCtrlItem import HistImagCtrlItem
    from .graphicsItems.PosnImagCtrlItem import PosnImagCtrlItem

    from .widgets.QHSImageConfigWidget import QHSImageConfigWidget
    from .widgets.QHSVectorConfigWidget import QHSVectorConfigWidget

    from .widgets.QHSImageFitWidget import QHSImageFitWidget

    __all__ = [
        "InfiniteLine",
        "InfLineLabel",
        "RegnPlotCtrlItem",
        "BaseImagCtrlItem",
        "HistImagCtrlItem",
        "PosnImagCtrlItem",
        "QHSImageConfigWidget",
        "QHSVectorConfigWidget",
        "QHSImageFitWidget"
    ]

else:
    __all__ = []

# print(f'Invoking __init__.py for {__name__}')

