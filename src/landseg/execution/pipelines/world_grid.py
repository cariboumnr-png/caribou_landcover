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
World grid pipeline command implementation.
'''

# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe.grid as grid
import landseg.utils as utils


# ----- public functions
def exec_world_grid(config: configs.RootConfig) -> None:
    '''
    Execute the world-grid pipeline.

    Args:
        config: Resolved root configuration object.
    '''
    root_paths = artifacts.ArtifactPaths.from_config(config)
    grid_cfg = config.data.world_grid

    logger = utils.Logger(name='world-grid', enable_file_log=False)
    logger.log_sep()
    logger.log('INFO', 'Building/loading canonical world grid')
    is_loaded, _grid = grid.prepare_world_grid(
        root_paths.world_grid,
        grid_cfg.mode,
        grid_cfg.params,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
    )
    status_str = 'loaded' if is_loaded else 'created and persisted'
    logger.log('INFO', f'[COMPLETE] World grid {status_str}: {_grid.gid}')
    logger.log('INFO', f'CRS: {_grid.crs}, Total Tiles: {len(_grid)}')
    logger.log_sep()
