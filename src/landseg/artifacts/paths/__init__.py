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

'''
Top-level namespace for `landseg.artifacts.paths`.

Exposes artifact and session path dataclasses for all pipelines.
'''

# standard imports
from __future__ import annotations
import importlib
import typing

__all__ = [
    'ArtifactPaths',
    'ETLPaths',
    'FoundationPaths',
    'SessionPaths',
    'TransformPaths',
]

# for static check
if typing.TYPE_CHECKING:
    from .etl import ETLPaths
    from .foundation import FoundationPaths
    from .root import ArtifactPaths
    from .session import SessionPaths
    from .transform import TransformPaths


def __getattr__(name: str):

    if name in {'ETLPaths'}:
        return getattr(importlib.import_module('.etl', __package__), name)

    if name in {'FoundationPaths'}:
        return getattr(importlib.import_module('.foundation', __package__), name)

    if name in {'SessionPaths'}:
        return getattr(importlib.import_module('.session', __package__), name)

    if name in {'ArtifactPaths'}:
        return getattr(importlib.import_module('.root', __package__), name)

    if name in {'TransformPaths'}:
        return getattr(importlib.import_module('.transform', __package__), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
