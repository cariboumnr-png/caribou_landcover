# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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
Top-level namespace for `landseg.geopipe.harmonize`.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'CanvasSpec',
    'HarmonizationLogger',
    'ProcessedRasters',
    # functions
    'create_canvas',
    'warp_to_canvas',
    'stack_canonical_raster',
    'unify_nodata_mask',
    'validate_domain_raster_index',
    'add_band_description_to_vrt',
    'add_tag_to_vrt',
    'compile_dataset_manifest',
    'process_source',
    # typing
    'HarmonizationReportSchema',
    'ProvenanceRecord',
    'WorldGridReport',
    'DatasetConfigItem',
]

# for static check
if typing.TYPE_CHECKING:
    from .common import (
        HarmonizationLogger,
        HarmonizationReportSchema,
        ProvenanceRecord,
        WorldGridReport,
    )
    from .rasters import (
        CanvasSpec,
        create_canvas,
        warp_to_canvas,
        stack_canonical_raster,
        unify_nodata_mask,
        validate_domain_raster_index,
        add_band_description_to_vrt,
        add_tag_to_vrt,
        ProcessedRasters,
        process_source,
        DatasetConfigItem,
        compile_dataset_manifest,
    )


def __getattr__(name: str):

    if name in {
        'HarmonizationLogger',
        'HarmonizationReportSchema',
        'ProvenanceRecord',
        'WorldGridReport',
    }:
        return getattr(importlib.import_module('.common', __package__), name)

    if name in {
        'CanvasSpec',
        'create_canvas',
        'warp_to_canvas',
        'stack_canonical_raster',
        'unify_nodata_mask',
        'validate_domain_raster_index',
        'add_band_description_to_vrt',
        'add_tag_to_vrt',
        'ProcessedRasters',
        'process_source',
        'DatasetConfigItem',
        'compile_dataset_manifest',
    }:
        return getattr(importlib.import_module('.rasters', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
