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

'''World grid artifacts lifecycle management.'''

# standard imports
import os
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.grid as grid

# typing aliases
D = list[list[int]]
M = geo_core.GridMeta
CTRL = artifacts.PayloadController[D, M]


def prepare_world_grid(
    grid_dpath: str,
    mode: str,
    config: grid.GridParameters,
    *,
    policy: artifacts.LifecyclePolicy,
) -> tuple[bool, geo_core.GridLayout]:
    '''
    Build or load a persisted world grid.

    If a grid with the configured ID exists on disk, it is loaded.
    Otherwise, a new grid is constructed from the extent configuration
    and grid profile, saved to disk, and returned.
    '''
    row_size, col_size = config.tile_size
    row_stride, col_stride = config.tile_stride
    gid = f'grid_row_{row_size}_{row_stride}_col_{col_size}_{col_stride}'

    ctrl = CTRL(
        os.path.join(grid_dpath, f'{gid}.json'),
        schema_id=geo_core.GridLayout.SCHEMA_ID,
        policy=policy
    )
    payload = ctrl.load()

    loaded_from_disk = False
    # load if present
    if payload:
        _grid = geo_core.GridLayout.from_payload(payload)
        loaded_from_disk = True
    else:
        # build if absent
        _grid = grid.build_grid(mode, config)
        payload = _grid.to_payload()
        ctrl.save(payload)

    return loaded_from_disk, _grid
