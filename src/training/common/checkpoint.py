# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods
'''Protocol for trainer checkpointing.'''

from __future__ import annotations
# standard imports
import typing

# checkpoint metadata
class CheckpointMetaLike(typing.TypedDict):
    '''Checkpont metadata'''
    best_value: float
    epoch: int
    step: int
