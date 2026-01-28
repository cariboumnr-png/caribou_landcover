'''
Top-level namespace for dataset.blocks.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    # classes
    'BlockCache',
    'BlockLayout',
    'DataBlock',
    'BlockCreationOptions',
    # functions
    'build_data_cache',
]

# for static check
if typing.TYPE_CHECKING:
    from ._types import BlockCreationOptions
    from .block import DataBlock
    from .cache import BlockCache, build_data_cache
    from .layout import BlockLayout

def __getattr__(name: str):

    if name == 'BlockCreationOptions':
        return importlib.import_module('._types', __package__).BlockCreationOptions
    if name == 'DataBlock':
        return importlib.import_module('.block', __package__).DataBlock
    if name in ['BlockCache', 'build_data_cache']:
        return getattr(importlib.import_module('.cache', __package__), name)
    if name == 'BlockLayout':
        return importlib.import_module('.layout', __package__).BlockLayout

    raise AttributeError(name)
