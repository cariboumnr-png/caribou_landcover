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

# pylint: disable=missing-function-docstring

'''
Canonical filesystem paths for data harmonization (ETL) artifacts.
'''

# standard imports
import dataclasses
import os


# ----- `ETLPaths` definition
@dataclasses.dataclass
class ETLPaths:
    '''Paths for data harmonization ETL artifacts.'''
    root: str
    run_id: str = ''
    run_folder: str = ''

    def init(self, trace_to_last: bool = False):
        '''Initialize an ETL run folder tree.'''
        if not self.run_id:
            i = 1
            while True:
                candidate_id = f'run_{i:04d}'
                candidate_folder = os.path.join(self.root, candidate_id)
                if not os.path.exists(candidate_folder):
                    break
                i += 1
            if trace_to_last and i > 1:
                i -= 1
                self.run_id = f'run_{i:04d}'
            else:
                self.run_id = f'run_{i:04d}'
            self.run_folder = os.path.join(self.root, self.run_id)

        os.makedirs(self.effective_root, exist_ok=True)

    @property
    def effective_root(self) -> str:
        return self.run_folder if self.run_folder else self.root

    def harmonized_raster(self, name: str) -> str:
        return os.path.join(self.effective_root, f'harmonized_{name}.vrt')

    @property
    def composite_raster(self) -> str:
        return os.path.join(self.effective_root, 'harmonized_image_composite.vrt')

    @property
    def valid_mask_raster(self) -> str:
        return os.path.join(self.effective_root, 'valid_pixel_mask.vrt')

    @property
    def report(self) -> str:
        return os.path.join(self.effective_root, 'etl_report.json')

    @property
    def config(self) -> str:
        return os.path.join(self.effective_root, 'config.json')
