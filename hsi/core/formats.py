import numpy as np

class HSFormatFlag(object):
    _counter = -1
    _flags = []

    def __init__(self, key, id):
        """Constructor

        Parameters
        ----------
        key : str
            The string identifier for the format flag.
        id : int
            The id for the format flag.
        """
        self._key = key
        self._id = id

    @property
    def key(self):
        """str: The flag label."""
        return self._key

    @property
    def id(self):
        """str: The flag id."""
        return self._id

    @classmethod
    def fromStr(cls, key):
        """Get the format flag from a string identifier."""
        for flag in cls._flags:
            if key == flag.key:
                return flag
        return None

    @classmethod
    def hasFlag(cls, flag):
        """list: A list of available flags."""
        if flag in cls._flags:
            return True
        else:
            return False

    @classmethod
    def getFlags(cls):
        """Get a list of available flags."""
        return cls._flags

    @classmethod
    def set(cls, key):
        """Adds a new format flag.

        Parameters
        ----------
        key : str
            The string identifier for the format flag.
        """
        cls._counter += 1
        flag = cls(key, id)
        cls._flags.append(flag)
        return flag


# HSFormatIntensity = HSFormatFlag.fromKey("INTENSITY")
# HSFormatAbsorption = HSFormatFlag.fromKey("ABSORPTION")
# HSFormatExtinction = HSFormatFlag.fromKey("EXTINCTION")
# HSFormatRefraction = HSFormatFlag.fromKey("REFRACTION")

HSIntensity = HSFormatFlag.set("Intensity")
"""static hsi.HSFormatFlag :
        Intensity 
"""

HSAbsorption = HSFormatFlag.set("Absorption")
"""static hsi.HSFormatFlag :
        Absorption 
"""

HSExtinction = HSFormatFlag.set("Extinction")
"""static hsi.HSFormatFlag :
        Extinction 
"""

HSRefraction = HSFormatFlag.set("Refraction")
"""static hsi.HSFormatFlag :
        Refraction 
"""

HSFormatDefault = HSIntensity

# HSFormatIntensity is HSFormatAbsorption
# HSFormatIntensity.key
# HSFormatIntensity.id
#
# HSFormatAbsorption.id
#
# flag = HSFormatFlag.fromStr("Extinction")
# flag is HSFormatExtinction
# flag is HSFormatIntensity

def convert(target_format, source_format, spec, wavelen=None):
    """Convert spectral data between different formats.

    The formats may be one of

        - :class:`hsi.HSIntensity`
        - :class:`hsi.HSAbsorption`
        - :class:`hsi.HSExtinction`
        - :class:`hsi.HSRefraction`

    Parameters
    ----------
    target_format : HSFormatFlag
        The target format.
    source_format : HSFormatFlag
        The source format.
    spec : numpy.ndarray
            The spectral data.
    wavelen :  list or numpy.ndarray, optional
        The wavelengths at which the spectral data are sampled. Required for
        conversions which involve the format :class:`hsi.HSRefraction`.

    Returns
    -------
    numpy.ndarray
        The spectral data in the new format.

    """
    if spec is None:
        return None

    if isinstance(spec, list):
        spec = np.array(spec)
    if isinstance(wavelen, list):
        wavelen = np.array(spec)


    if not isinstance(spec, np.ndarray):
        raise Exception("convert: Argument 'spec' must be ndarray.")

    if wavelen is None:
        if target_format is HSRefraction or source_format is HSRefraction:
            raise Exception("convert: Require argument 'wavelen'.")
    else:
        if not isinstance(wavelen, np.ndarray) or wavelen.ndim > 1:
            raise Exception("convert: Argument 'wavelen' must be 1D ndarray.")

    # if available reshape wavelen for broadcasting
    if wavelen is None:
        rwavelen = None
    else:
        ndim = spec.ndim
        if ndim > 1:
            axes = tuple(range(1, ndim))
            rwavelen = np.expand_dims(wavelen, axis=axes)
        else:
            rwavelen = wavelen

    # TODO: verify refraction calculation. Temporary use of scale factor
    wscale = 1e9

    # from intensity
    if target_format is HSIntensity and source_format is HSIntensity:
        return spec
    if target_format is HSAbsorption and source_format is HSIntensity:
        return -np.log(np.abs(spec))
    if target_format is HSExtinction and source_format is HSIntensity:
        return -np.log10(np.abs(spec))
    if target_format is HSRefraction and source_format is HSIntensity:
        return - np.log10(np.abs(spec)) * (rwavelen * wscale) / (4 * np.pi)

    # from absorption
    if target_format is HSIntensity and source_format is HSAbsorption:
        return np.exp(-spec)
    if target_format is HSAbsorption and source_format is HSAbsorption:
        return spec
    if target_format is HSExtinction and source_format is HSAbsorption:
        return spec / np.log(10)
    if target_format is HSRefraction and source_format is HSAbsorption:
        return spec / np.log(10) * (rwavelen * wscale) / (4 * np.pi)

    # from extinction
    if target_format is HSIntensity and source_format is HSExtinction:
        return 10. ** (-spec)
    if target_format is HSAbsorption and source_format is HSExtinction:
        return spec * np.log(10)
    if target_format is HSExtinction and source_format is HSExtinction:
        return spec
    if target_format is HSRefraction and source_format is HSExtinction:
        return spec * (rwavelen * wscale) / (4 * np.pi)

    # from refraction
    if target_format is HSIntensity and source_format is HSRefraction:
        return 10. ** (-spec * (4 * np.pi) / (rwavelen * wscale))
    if target_format is HSAbsorption and source_format is HSRefraction:
        return spec * (4 * np.pi) / (rwavelen * wscale) * np.log(10)
    if target_format is HSExtinction and source_format is HSRefraction:
        return spec * (4 * np.pi) / (rwavelen * wscale)
    if target_format is HSRefraction and source_format is HSRefraction:
        return spec