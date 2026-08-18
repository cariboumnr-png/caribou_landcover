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
Top-level namespace for `landseg.geopipe.prepare.data_partition.split`.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'AoiSplitsResult',
    'PartitionParameters',
    'PartitionResults',
    'SplitsResult',
    'HydrationResults',
    # functions
    'create_blocks_partition',
    'filter_safe_tiles',
    'hydrate_train_split',
    'intersect_aoi_raster',
    'resolve_aoi_partitions',
    'score_blocks',
    'stratified_splitter',
]


# for static check
if typing.TYPE_CHECKING:
    from .aoi import (
        AoiSplitsResult,
        intersect_aoi_raster,
        resolve_aoi_partitions,
    )
    from .filter import filter_safe_tiles
    from .hydrate import HydrationResults, hydrate_train_split
    from .pipeline import (
        PartitionParameters,
        PartitionResults,
        create_blocks_partition,
    )
    from .score import score_blocks
    from .stratify import SplitsResult, stratified_splitter


def __getattr__(name: str):

    if name in {
        'AoiSplitsResult',
        'intersect_aoi_raster',
        'resolve_aoi_partitions',
    }:
        return getattr(importlib.import_module('.aoi', __package__), name)

    if name in {'filter_safe_tiles'}:
        return getattr(importlib.import_module('.filter', __package__), name)

    if name in {'HydrationResults', 'hydrate_train_split'}:
        return getattr(importlib.import_module('.hydrate', __package__), name)

    if name in {
        'PartitionParameters',
        'PartitionResults',
        'create_blocks_partition',
    }:
        return getattr(importlib.import_module('.pipeline', __package__), name)

    if name in {'score_blocks'}:
        return getattr(importlib.import_module('.score', __package__), name)

    if name in {'SplitsResult', 'stratified_splitter'}:
        return getattr(importlib.import_module('.stratify', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
