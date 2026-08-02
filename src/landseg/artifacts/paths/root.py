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
Root entrypoint for all artifact path namespaces.
'''

# standard imports
import dataclasses
import os
# local imports
import landseg.artifacts.paths as paths


# ----- `ArtifactPaths` definition
@dataclasses.dataclass
class ArtifactPaths:
    '''Root entrypoint for all artifact path namespaces.'''
    root: str

    @property
    def foundation(self) -> paths.FoundationPaths:
        '''Return FoundationPaths container.'''
        return paths.FoundationPaths(os.path.join(self.root, 'foundation'))

    @property
    def transform(self) -> paths.TransformPaths:
        '''Return TransformPaths container.'''
        return paths.TransformPaths(os.path.join(self.root, 'transform'))

    @property
    def etl(self) -> paths.ETLPaths:
        '''Return ETLPaths container.'''
        return paths.ETLPaths(os.path.join(self.root, 'harmonized'))

    @property
    def session(self) -> paths.SessionPaths:
        '''Return SessionPaths container.'''
        return paths.SessionPaths(results_root=os.path.join(self.root, 'results'))
