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
    'HarmonizationLogger',
    'ProcessedRasters',
    # functions
    'unify_nodata_mask',
    'compile_dataset_manifest',
    'process_source',
    'get_available_profiles',
    'validate_taxonomy_specs',
    # typing
    'HarmonizationReportSchema',
    'ProvenanceRecord',
    'WorldGridReport',
]

# for static check
if typing.TYPE_CHECKING:
    from .common import (
        HarmonizationLogger,
        HarmonizationReportSchema,
        ProvenanceRecord,
        WorldGridReport,
    )

    from .manifest import(
        compile_dataset_manifest,
    )

    from .pipeline import (
        ProcessedRasters,
        process_source,
    )

    from .rasters import (
        unify_nodata_mask,
    )

    from .taxonomy import(
        get_available_profiles,
        validate_taxonomy_specs,
    )


def __getattr__(name: str):

    if name in {
        'HarmonizationLogger',
        'HarmonizationReportSchema',
        'ProvenanceRecord',
        'WorldGridReport',
    }:
        return getattr(
            importlib.import_module('.common', __package__), name
        )

    if name in {
        'compile_dataset_manifest',
    }:
        return getattr(
            importlib.import_module('.manifest', __package__), name
        )

    if name in {
        'unify_nodata_mask',
    }:
        return getattr(
            importlib.import_module('.rasters', __package__), name
        )

    if name in {
        'ProcessedRasters',
        'process_source',
    }:
        return getattr(
            importlib.import_module('.pipeline', __package__), name
        )

    if name in {
        'get_available_profiles',
        'validate_taxonomy_specs',
    }:
        return getattr(
            importlib.import_module('.taxonomy', __package__), name
        )

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
