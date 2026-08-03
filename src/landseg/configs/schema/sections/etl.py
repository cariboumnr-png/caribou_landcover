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

# alias
field = dataclasses.field


# -------------------------------`ETLConfig` schema-------------------------------
@dataclasses.dataclass
class ETLConfig:
    target_crs: str = 'EPSG:3161'
    target_resolution: float = 20.0
    reference_raster: str = ''
    resampling_continuous: str = 'bilinear'
    resampling_categorical: str = 'nearest'
    features: dict[str, str] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    output_dpath: str = 'experiment/harmonized'

    @property
    def sources(self) -> dict[str, str]:
        '''Return combined dictionary of all feature and label source rasters.'''
        combined = dict(self.features)
        combined.update(self.labels)
        return combined

    def validate(self) -> None:
        if self.target_resolution <= 0.0:
            raise ValueError('target_resolution must be positive.')
        if not self.target_crs:
            raise ValueError('target_crs cannot be empty.')
