# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
#         as represented by the Minister of Natural Resources, 2026.          #
#                                                                             #
#                      © King's Printer for Ontario, 2026.                    #
#                                                                             #
#       Licensed under the Apache License, Version 2.0 (the 'License');       #
#          you may not use this file except in compliance with the            #
#                                  License.                                   #
#                  You may obtain a copy of the License at:                   #
#                                                                             #
#                  http://www.apache.org/licenses/LICENSE-2.0                 #
#                                                                             #
#    Unless required by applicable law or agreed to in writing, software      #
#     distributed under the License is distributed on an 'AS IS' BASIS,       #
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or        #
#                                   implied.                                  #
#       See the License for the specific language governing permissions       #
#                       and limitations under the License.                    #
# =========================================================================== #

# pylint: disable=missing-function-docstring

'''
Tools for preparing and loading world grid layouts.

This module provides the public entry point to build or load a persisted
`GridLayout` from configuration. If the requested grid already exists on
disk, it is loaded; otherwise, the grid specification is derived from the
extent configuration and the grid is created, saved, and returned.

Supported extent modes:
- 'ref'   : derive geometry from a reference raster (bounds, pixel size)
- 'aoi'   : derive from explicit origin, pixel size, and grid extent
- 'tiles' : derive from explicit origin, pixel size, and grid shape
'''

# standard imports
import os
import typing
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.utils as geo_utils


# ------------------------------Public Dataclass------------------------------
class GridParameters(typing.Protocol):
    '''Container for grid generation configuration.'''
    @property
    def tile_size(self) -> tuple[int, int]: ...
    @property
    def tile_stride(self) -> tuple[int, int]: ...
    @property
    def ref_fpath(self) -> str | None: ...
    @property
    def crs_string(self) -> str | None: ...
    @property
    def origin(self) -> tuple[float, float] | None: ...
    @property
    def pixel_size(self) -> tuple[float, float] | None: ...
    @property
    def extent_in_crs_units(self) -> tuple[float, float] | None: ...

# -------------------------------Public Function-------------------------------
def build_grid(
    mode: typing.Literal['ref', 'manual'] | str,
    config: GridParameters
) -> geo_core.GridLayout:
    '''
    Build or load a persisted world grid.

    If a grid with the configured ID exists on disk, it is loaded.
    Otherwise, a new grid is constructed from the extent configuration
    and grid profile, saved to disk, and returned.
    '''
    # derive from reference raster
    if mode == 'ref':
        if not (config.ref_fpath and os.path.exists(config.ref_fpath)):
            raise ValueError(f'Invalid reference raster: {config.ref_fpath}')

        with geo_utils.open_rasters(config.ref_fpath) as (src,):
            assert src
            # get transform - pixel size
            transform = src.transform
            px, py = transform.a, abs(transform.e)
            # get bounding box - origin and extent
            l, b, r, t = src.bounds
            # assign to gridspec
            grid_spec = geo_core.GridSpec(
                crs=config.crs_string or str(src.crs),
                origin=(l, t),             # left, top as x, y
                pixel_size=(px, py),       # pixel size in x, y
                tile_size=config.tile_size,
                tile_stride=config.tile_stride,
                grid_extent=(t - b, r - l) # top-bottom as H, right-left as W
            )

    # manually define the extent
    elif mode == 'manual':
        if not config.crs_string:
            raise ValueError('CRS string not provided')
        if not config.origin:
            raise ValueError('Origin not provided')
        if not config.pixel_size:
            raise ValueError('Pixel size not provided')
        if not config.extent_in_crs_units:
            raise ValueError('Extent (in CRS units) not provided')

        grid_spec = geo_core.GridSpec(
            crs=config.crs_string,
            origin=config.origin,
            pixel_size=config.pixel_size,
            tile_size=config.tile_size,
            tile_stride=config.tile_stride,
            grid_extent=config.extent_in_crs_units
        )

    else:
        raise ValueError(f'Invalid extent mode: {mode}')

    _mode = 'bbox' # now constant
    output_grid = geo_core.GridLayout(_mode, grid_spec)
    return output_grid
