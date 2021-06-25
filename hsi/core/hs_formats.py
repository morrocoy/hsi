import numpy


class HSFormatFlag(object):
    _counter = -1
    _flags = []

    def __init__(self, key, format_id):
        """Constructor

        Parameters
        ----------
        key : str
            The string identifier for the hsformat flag.
        format_id : int
            The id for the hsformat flag.
        """
        self._key = key
        self._id = format_id

    @property
    def key(self):
        """str: The flag label."""
        return self._key

    @property
    def id(self):
        """str: The flag id."""
        return self._id

    @classmethod
    def from_str(cls, key):
        """Get the hsformat flag from a string identifier."""
        for flag in cls._flags:
            if key == flag.key:
                return flag
        return None

    @classmethod
    def has_flag(cls, flag):
        """list: A list of available flags."""
        if flag in cls._flags:
            return True
        else:
            return False

    @classmethod
    def get_flags(cls):
        """Get a list of available flags."""
        return cls._flags

    @classmethod
    def set(cls, key):
        """Adds a new hsformat flag.

        Parameters
        ----------
        key : str
            The string identifier for the hsformat flag.
        """
        cls._counter += 1
        flag = cls(key, cls._counter)
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
# flag = HSFormatFlag.from_str("Extinction")
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
        The target hsformat.
    source_format : HSFormatFlag
        The source hsformat.
    spec : numpy.ndarray
            The spectral data.
    wavelen :  list or numpy.ndarray, optional
        The wavelengths at which the spectral data are sampled. Required for
        conversions which involve the hsformat :class:`hsi.HSRefraction`.

    Returns
    -------
    numpy.ndarray
        The spectral data in the new hsformat.

    """
    if spec is None:
        return None

    if isinstance(spec, list):
        spec = numpy.array(spec)
    if isinstance(wavelen, list):
        wavelen = numpy.array(spec)

    if not isinstance(spec, numpy.ndarray):
        raise Exception("convert: Argument 'spec' must be ndarray.")

    if wavelen is None:
        if target_format is HSRefraction or source_format is HSRefraction:
            raise Exception("convert: Require argument 'wavelen'.")
    else:
        if not isinstance(wavelen, numpy.ndarray) or wavelen.ndim > 1:
            raise Exception("convert: Argument 'wavelen' must be 1D ndarray.")

    # if available reshape wavelen for broadcasting
    if wavelen is None:
        rwavelen = None
    else:
        ndim = spec.ndim
        if ndim > 1:
            axes = tuple(range(1, ndim))
            rwavelen = numpy.expand_dims(wavelen, axis=axes)
        else:
            rwavelen = wavelen

    # Absorption coefficient (ua) and extinction coefficient (eps):
    # ua = eps * log(10)
    # Complex refractive Index :n = n_r + i n_i
    # Absorption coefficient and imaginary refractive index:
    # ua = 4 pi n_i/ lambda
    # Intensity: I = exp(-ua * l) = 10 ** (-eps * l) = exp(-4 pi n_i / lambda)
    wscale = 1e9

    # from intensity
    if target_format is HSIntensity and source_format is HSIntensity:
        return spec
    if target_format is HSAbsorption and source_format is HSIntensity:
        return -numpy.log(numpy.abs(spec))
    if target_format is HSExtinction and source_format is HSIntensity:
        return -numpy.log10(numpy.abs(spec))
    if target_format is HSRefraction and source_format is HSIntensity:
        return -numpy.log(numpy.abs(spec)) * (rwavelen * wscale) / (
                4 * numpy.pi)

    # from absorption
    if target_format is HSIntensity and source_format is HSAbsorption:
        return numpy.exp(-spec)
    if target_format is HSAbsorption and source_format is HSAbsorption:
        return spec
    if target_format is HSExtinction and source_format is HSAbsorption:
        return spec / numpy.log(10)
    if target_format is HSRefraction and source_format is HSAbsorption:
        return spec * (rwavelen * wscale) / (4 * numpy.pi)

    # from extinction
    if target_format is HSIntensity and source_format is HSExtinction:
        return 10. ** (-spec)
    if target_format is HSAbsorption and source_format is HSExtinction:
        return spec * numpy.log(10)
    if target_format is HSExtinction and source_format is HSExtinction:
        return spec
    if target_format is HSRefraction and source_format is HSExtinction:
        return spec * numpy.log(10) * (rwavelen * wscale) / (4 * numpy.pi)

    # from refraction
    if target_format is HSIntensity and source_format is HSRefraction:
        return numpy.exp(-spec * (4 * numpy.pi) / (rwavelen * wscale))
    if target_format is HSAbsorption and source_format is HSRefraction:
        return spec * (4 * numpy.pi) / (rwavelen * wscale)
    if target_format is HSExtinction and source_format is HSRefraction:
        return spec * (4 * numpy.pi) / (rwavelen * wscale) / numpy.log(10)
    if target_format is HSRefraction and source_format is HSRefraction:
        return spec
