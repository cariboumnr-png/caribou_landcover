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
import typing
# local imports
import landseg.artifacts.paths as paths

if typing.TYPE_CHECKING:
    import landseg.configs as configs


# ----- `ArtifactPaths` definition
@dataclasses.dataclass
class ArtifactPaths:
    '''Root entrypoint for all artifact path namespaces.'''
    root: str = './experiment'
    etl_root: str | None = None
    foundation_root: str | None = None
    transform_root: str | None = None
    session_root: str | None = None

    @classmethod
    def from_config(cls, config: 'configs.RootConfig') -> typing.Self:
        '''Construct `ArtifactPaths` from a `RootConfig` instance.'''
        return cls(
            root=config.execution.exp_root,
            etl_root=config.etl.output_dpath,
            foundation_root=config.foundation.output_dpath,
            transform_root=config.transform.output_dpath,
            session_root=config.session.output_dpath,
        )

    @property
    def foundation(self) -> paths.FoundationPaths:
        '''Return FoundationPaths container.'''
        r = (
            self.foundation_root
            if self.foundation_root
            else os.path.join(self.root, 'foundation')
        )
        return paths.FoundationPaths(r)

    @property
    def transform(self) -> paths.TransformPaths:
        '''Return TransformPaths container.'''
        r = (
            self.transform_root
            if self.transform_root
            else os.path.join(self.root, 'transform')
        )
        return paths.TransformPaths(r)

    @property
    def etl(self) -> paths.ETLPaths:
        '''Return ETLPaths container.'''
        r = (
            self.etl_root
            if self.etl_root
            else os.path.join(self.root, 'harmonized')
        )
        return paths.ETLPaths(r)

    @property
    def session(self) -> paths.SessionPaths:
        '''Return SessionPaths container.'''
        r = (
            self.session_root
            if self.session_root
            else os.path.join(self.root, 'results')
        )
        return paths.SessionPaths(root=r)
