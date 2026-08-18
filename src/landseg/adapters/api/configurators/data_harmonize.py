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
Data harmonization ETL configurator
'''

# standard imports
import typing
# local imports
import landseg.adapters.api.configurators as configurators


class DataHarmonizationConfigurator(configurators.BaseConfigurator):
    '''Configure data harmonization ETL.'''

    def __init__(
        self,
        experiment_root: str,
        dataset_name: str = 'sample_data',
    ):
        super().__init__(experiment_root, 'data-harmonize', dataset_name)

    def set_canvas(
        self,
        target_crs: str,
        target_resolution: float,
        reference_raster: str | None = None,
    ) -> typing.Self:
        '''Set canvas spatial reference specs.'''
        self._cfg.data.harmonization.canvas.target_crs = target_crs
        self._cfg.data.harmonization.canvas.target_resolution = (
            target_resolution
        )
        if reference_raster:
            self._cfg.data.harmonization.canvas.reference_raster = (
                reference_raster
            )
        return self

    def set_dataset_manifest(
        self,
        dataset_manifest: str,
        dataset_name: str = 'sample_data',
    ) -> typing.Self:
        '''Set dataset metadata manifest path and dataset name.'''
        self._cfg.data.harmonization.dataset_manifest = dataset_manifest
        self._cfg.data.harmonization.dataset_name = dataset_name
        self._cfg.data.ingestion.datablocks.name = dataset_name
        return self

    def set_resampling(
        self,
        continuous: str = 'bilinear',
        categorical: str = 'nearest',
    ) -> typing.Self:
        '''Set raster resampling methods.'''
        self._cfg.data.harmonization.resampling_continuous = continuous
        self._cfg.data.harmonization.resampling_categorical = categorical
        return self

    def set_grid(
        self,
        tile_size: int = 256,
        tile_overlap: int = 0,
        crs: str = '',
        mode: str = 'ref',
    ) -> typing.Self:
        '''Set study extent and grid specs.'''
        self._cfg.data.harmonization.grid.mode = mode
        self._cfg.data.harmonization.grid.crs = crs
        self._cfg.data.harmonization.grid.tile_specs.size_row = tile_size
        self._cfg.data.harmonization.grid.tile_specs.size_col = tile_size
        self._cfg.data.harmonization.grid.tile_specs.overlap_row = tile_overlap
        self._cfg.data.harmonization.grid.tile_specs.overlap_col = tile_overlap
        return self

    def set_output_dpath(self, output_dpath: str) -> typing.Self:
        '''Set output directory path for harmonized artifacts.'''
        self._cfg.data.harmonization.output_dpath = output_dpath
        return self
