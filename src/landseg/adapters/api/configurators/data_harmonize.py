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
        reference_raster: str | None = None
    ) -> typing.Self:
        '''Set canvas spatial reference specs.'''
        self._cfg.data.harmonization.canvas.target_crs = target_crs
        self._cfg.data.harmonization.canvas.target_resolution = target_resolution
        if reference_raster:
            self._cfg.data.harmonization.canvas.reference_raster = reference_raster
        return self

    def set_dev_features(
        self,
        dev_features: dict[str, str]
    ) -> typing.Self:
        '''Set continuous development feature rasters map.'''
        self._cfg.data.harmonization.raw_data.dev_features = dev_features
        return self

    def set_features(
        self,
        features: dict[str, str]
    ) -> typing.Self:
        '''Set continuous feature rasters map.'''
        return self.set_dev_features(features)

    def set_dev_labels(
        self,
        dev_labels: dict[str, str]
    ) -> typing.Self:
        '''Set categorical development label rasters map.'''
        self._cfg.data.harmonization.raw_data.dev_labels = dev_labels
        return self

    def set_labels(
        self,
        labels: dict[str, str]
    ) -> typing.Self:
        '''Set categorical label rasters map.'''
        return self.set_dev_labels(labels)

    def set_dataset_config(
        self,
        dataset_config: str,
        dataset_name: str = 'sample_data'
    ) -> typing.Self:
        '''Set dataset metadata configuration path and name.'''
        self._cfg.data.harmonization.dataset_config = dataset_config
        self._cfg.data.harmonization.dataset_name = dataset_name
        self._cfg.data.ingestion.datablocks.name = dataset_name
        return self

    def set_test_holdout(
        self,
        test_features: dict[str, str],
        test_labels: dict[str, str]
    ) -> typing.Self:
        '''Set test holdout rasters map.'''
        self._cfg.data.harmonization.raw_data.test_features = test_features
        self._cfg.data.harmonization.raw_data.test_labels = test_labels
        return self
