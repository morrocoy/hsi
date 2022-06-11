""" hsi. A hyper spectral image analysis package
"""

# from ..misc import __version__

from .. import CONFIG_OPTIONS

if CONFIG_OPTIONS['enableGUI']:

    from .graphicsItems.ImageItem import ImageItem
    from .graphicsItems.InfiniteLine import InfiniteLine, InfLineLabel
    from .graphicsItems.ColorBarItem import ColorBarItem

    from .graphicsItems.RegnPlotCtrlItem import RegnPlotCtrlItem
    from .graphicsItems.BaseImagCtrlItem import BaseImagCtrlItem
    from .graphicsItems.HistImagCtrlItem import HistImagCtrlItem
    from .graphicsItems.PosnImagCtrlItem import PosnImagCtrlItem
    from .graphicsItems.RegnImagCtrlItem import RegnImagCtrlItem

    from .widgets.QHSImageConfigWidget import QHSImageConfigWidget
    from .widgets.QHSCoFitConfigWidget import QHSCoFitConfigWidget

    __all__ = [
        "ImageItem",
        "InfiniteLine",
        "InfLineLabel",
        "ColorBarItem",
        "RegnPlotCtrlItem",
        "BaseImagCtrlItem",
        "HistImagCtrlItem",
        "PosnImagCtrlItem",
        "RegnImagCtrlItem",
        "QHSImageConfigWidget",
        "QHSCoFitConfigWidget",
    ]

else:
    __all__ = []

# print(f'Invoking __init__.py for {__name__}')

