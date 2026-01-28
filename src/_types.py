'''Project-wide type aliases and lazy imports for type checking.'''

# standard imports
import typing
# third-party imports
import rasterio.io
import rasterio.windows

# typing aliases
# generic
ConfigType: typing.TypeAlias = typing.Mapping[str, typing.Any]
# from rasterio
RasterReader: typing.TypeAlias = rasterio.io.DatasetReader
RasterWindow: typing.TypeAlias = rasterio.windows.Window
