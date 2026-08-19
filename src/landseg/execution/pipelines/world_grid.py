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
    grid_cfg = config.data.world_grid

    logger = utils.Logger(name='world-grid', enable_file_log=False)

    logger.log_sep()
    logger.log('INFO', 'Building/loading canonical world grid')

    is_loaded, grid_fp, world_grid = grid.prepare_world_grid(grid_cfg)
    status_str = 'loaded' if is_loaded else 'created and persisted'

    logger.log('INFO', f'[COMPLETE] World grid {status_str}')
    logger.log('INFO', f'Grid ID: {world_grid.gid}')
    logger.log('INFO', f'Grid artifact file path: {grid_fp}')
    logger.log('INFO', f'CRS: {world_grid.crs}')
    logger.log('INFO', f'Total Tiles: {len(world_grid)}')
    logger.log_sep()
