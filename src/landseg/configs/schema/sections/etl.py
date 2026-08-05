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

# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

'''
Data harmonization ETL schema
'''

# standard imports
import dataclasses
import re
# local imports
import landseg.configs.schema.utils as utils

# alias
field = dataclasses.field


# -------------------------------`ETLConfig` schema-------------------------------
@dataclasses.dataclass
class _Canvas:
    reference_raster: str = ''
    target_crs: str | None = None
    target_resolution: float | None = None


@dataclasses.dataclass
class _RawData:
    dev_features: dict[str, str] = field(default_factory=dict)
    domains: dict[str, str] = field(default_factory=dict)
    dev_labels: dict[str, str] = field(default_factory=dict)
    test_features: dict[str, str] = field(default_factory=dict)
    test_labels: dict[str, str] = field(default_factory=dict)


@dataclasses.dataclass
class ETLConfig:
    canvas: _Canvas = field(default_factory=_Canvas)
    raw_data: _RawData = field(default_factory=_RawData)
    dataset_name: str = 'sample_data'
    dataset_config: str = ''
    resampling_continuous: str = 'bilinear'
    resampling_categorical: str = 'nearest'
    output_dpath: str = 'experiment/harmonized'

    def validate(self) -> None:
        utils.must_exist(self.canvas.reference_raster, 'Reference raster')
        if self.dataset_config:
            utils.must_exist(self.dataset_config, 'Dataset configuration JSON')

        if (
            self.canvas.target_crs and
            not bool(re.fullmatch(r'epsg:\d+', self.canvas.target_crs, re.I))
        ):
            raise ValueError('Invalid CRS identifier. Must be "EPSG:...."')

        if (
            self.canvas.target_resolution and
            self.canvas.target_resolution <= 0.0
        ):
            raise ValueError('target_resolution must be positive.')
