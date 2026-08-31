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

'''
Multi-raster channel composition and nodata mask unification operations.
'''

# third-party imports
import rasterio


# ----- public functions
def add_band_description_to_vrt(
    vrt_fpath: str,
    band_mapping: dict[int, str]
):
    '''Simple helper to add band description to a `.vrt` raster file.'''
    with rasterio.open(vrt_fpath, 'r+') as vrt:
        if len(band_mapping) != vrt.count:
            raise ValueError(
                f'Expected {vrt.count} band descriptions, '
                f'got {len(band_mapping)}'
            )
        for band, name in band_mapping.items():
            vrt.set_band_description(int(band), name)


def add_tag_to_vrt(vrt_fpath: str, **kwargs):
    '''Simple helper to add metadata to a `.vrt` raster file.'''
    with rasterio.open(vrt_fpath, 'r+') as vrt:
        vrt.update_tags(**kwargs)


# def validate_domain_raster_index(
#     input_path: str,
#     min_allowed: int = 1
# ) -> None:
#     '''
#     Validate that a domain raster contains 1-based indices.

#     Args:
#         input_path: Path to the input domain raster file.
#         min_allowed: Minimum allowed index value (default: 1).

#     Raises:
#         ValueError: If valid pixel values contain any values < min_allowed.
#     '''
#     with rasterio.open(input_path) as src:
#         data = src.read(1)
#         nodata = src.nodata
#         valid_data = data[data != nodata] if nodata is not None else data
#         if valid_data.size > 0 and int(valid_data.min()) < min_allowed:
#             raise ValueError(
#                 f'Domain raster [{input_path}] contains index values '
#                 f'< {min_allowed} (minimum found: {valid_data.min()}). '
#                 'Categorical domain rasters must use 1-based indexing.'
#             )