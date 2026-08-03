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
Canonical filesystem paths for session experiment training and evaluation results.
'''

# standard imports
import dataclasses
import os


# ----- `SessionPaths` definition
@dataclasses.dataclass
class SessionPaths:
    '''Root entry of a training run.'''

    results_root: str
    run_id: str = ''
    run_folder: str = ''

    @property
    def checkpoints(self) -> str:
        return os.path.join(self.run_folder, 'checkpoints')

    @property
    def phase_status(self) -> str:
        return os.path.join(self.checkpoints, 'status.json')

    @property
    def logs(self) -> str:
        return os.path.join(self.run_folder, 'logs')

    @property
    def plots(self) -> str:
        return os.path.join(self.run_folder, 'plots')

    @property
    def previews(self) -> str:
        return os.path.join(self.run_folder, 'previews')

    @property
    def config(self) -> str:
        return os.path.join(self.run_folder, 'config.json')

    @property
    def evaluation(self) -> str:
        return os.path.join(self.run_folder, 'evaluation.json')

    @property
    def summary(self) -> str:
        return os.path.join(self.run_folder, 'summary.json')

    @property
    def step_results(self) -> str:
        return os.path.join(self.run_folder, 'step_results.json')

    def init(self, trace_to_last: bool = False):
        '''Initialize a results folder.'''

        # starting id number
        i = 1
        # find the latest run number
        while True:
            self.run_id = f'run_{i:04d}'
            self.run_folder = os.path.join(self.results_root, self.run_id)
            if not os.path.exists(self.run_folder):
                break
            i += 1
        # if trace to the last folder
        if trace_to_last:
            self.run_id = f'run_{i - 1:04d}'
            self.run_folder = os.path.join(self.results_root, self.run_id)

        # create all subfolders if not already exist
        os.makedirs(self.checkpoints, exist_ok=True)
        os.makedirs(self.logs, exist_ok=True)
        os.makedirs(self.plots, exist_ok=True)
        os.makedirs(self.previews, exist_ok=True)

    def best_checkpoint(self, name: str) -> str:
        return os.path.join(self.checkpoints, f'{name}_best.pt')

    def last_checkpoint(self, name: str) -> str:
        return os.path.join(self.checkpoints, f'{name}_last.pt')
