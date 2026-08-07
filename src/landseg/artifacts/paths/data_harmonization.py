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

# pylint: disable=missing-function-docstring

'''
Canonical filesystem paths for data harmonization (ETL) artifacts.
'''

# standard imports
import dataclasses
import os


# ----- `HarmonizationPaths` definition
@dataclasses.dataclass
class HarmonizationPaths:
    '''Paths for data harmonization ETL artifacts.'''
    root: str
    run_id: str = ''
    run_folder: str = ''
    _current_run_folder: str = ''

    @property
    def effective_root(self) -> str:
        return self._current_run_folder if self._current_run_folder else (
            self.run_folder if self.run_folder else self.root
        )

    @property
    def dev_feature_raster(self) -> str:
        return os.path.join(self.effective_root, 'stacked_images.vrt')

    @property
    def feature_raster(self) -> str:
        return self.dev_feature_raster

    @property
    def dev_label_raster(self) -> str:
        return os.path.join(self.effective_root, 'stacked_labels.vrt')

    @property
    def label_raster(self) -> str:
        return self.dev_label_raster

    @property
    def domain_raster(self) -> str:
        return os.path.join(self.effective_root, 'stacked_domains.vrt')

    @property
    def valid_mask_raster(self) -> str:
        return os.path.join(self.effective_root, 'valid_pixel_mask.vrt')

    @property
    def test_feature_raster(self) -> str:
        return os.path.join(
            self.effective_root, 'stacked_test_images.vrt'
        )

    @property
    def test_label_raster(self) -> str:
        return os.path.join(
            self.effective_root, 'stacked_test_labels.vrt'
        )

    @property
    def has_test_data(self) -> bool:
        return (
            os.path.exists(self.test_feature_raster) and
            os.path.exists(self.test_label_raster)
        )

    @property
    def dataset_config(self) -> str:
        return os.path.join(self.effective_root, 'dataset_config.json')

    @property
    def report(self) -> str:
        return os.path.join(self.effective_root, 'harmonize_report.json')

    @property
    def config(self) -> str:
        return os.path.join(self.effective_root, 'config.json')

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
            self._current_run_folder = self.run_folder

        os.makedirs(self.effective_root, exist_ok=True)

    def get_run_folder(self, run_id: int | str | None = None) -> str:
        '''
        Return the path to a run folder.

        Args:
            run_id:
                Integer run ID (e.g. 1 -> run_0001), string run folder
                name/ID (e.g. "run_0001" or "1"), or directory path. If
                None, returns the latest existing run folder.

        Raises:
            FileNotFoundError:
                If the requested run does not exist or no run folders
                exist.
            TypeError:
                If run_id is of an invalid type.
        '''
        if run_id is not None:
            if isinstance(run_id, int):
                folder = os.path.join(self.root, f'run_{run_id:04d}')
            elif isinstance(run_id, str):
                if run_id.isdigit():
                    folder = os.path.join(self.root, f'run_{int(run_id):04d}')
                elif os.path.isdir(run_id):
                    folder = run_id
                elif os.path.isdir(os.path.join(self.root, run_id)):
                    folder = os.path.join(self.root, run_id)
                else:
                    raise FileNotFoundError(
                        f'Run folder does not exist: {run_id}'
                    )
            else:
                raise TypeError(f'Invalid run_id type: {type(run_id)}')

            if not os.path.isdir(folder):
                raise FileNotFoundError(f'Run folder does not exist: {folder}')
            self._current_run_folder = folder
            return folder

        runs = sorted(
            d for d in os.listdir(self.root)
            if d.startswith('run_')
            and os.path.isdir(os.path.join(self.root, d))
        )

        if not runs:
            raise FileNotFoundError('No run folders found.')

        self._current_run_folder = os.path.join(self.root, runs[-1])
        return self._current_run_folder

    def harmonized_raster(self, name: str) -> str:
        return os.path.join(self.effective_root, f'harmonized_{name}.vrt')
