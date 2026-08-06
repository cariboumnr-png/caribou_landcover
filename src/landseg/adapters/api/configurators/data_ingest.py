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
Data ingestion configurator
'''

# standard imports
import typing
# local imports
import landseg.adapters.api.configurators as configurators

class DataIngestionConfigurator(configurators.BaseConfigurator):
    '''Configure data ingestion.'''

    def __init__(
        self,
        experiment_root: str,
        dataset_name: str = 'sample_data',
    ):
        super().__init__(experiment_root, 'data-ingest', dataset_name)

    def set_grid(
        self,
        tile_size: int = 256,
        tile_overlap: int = 0,
        crs: str = ''
    ) -> typing.Self:
        '''Set study extent and grid specs.'''
        self._cfg.data.ingestion.grid.mode = 'ref'
        self._cfg.data.ingestion.grid.crs = crs
        self._cfg.data.ingestion.grid.tile_specs.size_row = tile_size
        self._cfg.data.ingestion.grid.tile_specs.size_col = tile_size
        self._cfg.data.ingestion.grid.tile_specs.overlap_row = tile_overlap
        self._cfg.data.ingestion.grid.tile_specs.overlap_col = tile_overlap
        return self

    def set_rebuild(self, rebuild: bool) -> typing.Self:
        '''Set whether to force rebuild ingestion artifacts.'''
        self._cfg.data.ingestion.rebuild = rebuild
        return self

    def set_harmonization_run(
        self,
        target_run: int | str | None = None
    ) -> typing.Self:
        '''Set targeted harmonization run index, folder name, or path.'''
        self._cfg.data.ingestion.harmonization_run = target_run
        return self
