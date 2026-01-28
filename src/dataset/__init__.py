'''
Top-level namespace for dataset.

Exposes selected public functions via lazy resolution to keep import
order simple and circular-free.
'''

from __future__ import annotations
import importlib
import typing

__all__ = [
    'prepare_data',
    'DataSummary',
    'DataBlock',
    # typing
    'BlockCreationOptions',
    'BlockLayout',
]

# for static check
if typing.TYPE_CHECKING:
    from .blocks import BlockCreationOptions, DataBlock, BlockLayout
    from .prep import run as prepare_data
    from .summary import DataSummary

def __getattr__(name: str):

    if name in ['BlockCreationOptions','DataBlock', 'BlockLayout']:
        return getattr(importlib.import_module('.blocks', __package__), name)
    if name == 'prepare_data':
        return importlib.import_module('.prep', __package__).run
    if name == 'DataSummary':
        return importlib.import_module('.summary', __package__).DataSummary

    raise AttributeError(name)
