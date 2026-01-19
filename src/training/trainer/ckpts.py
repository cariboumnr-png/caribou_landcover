# pylint: disable=missing-function-docstring, too-few-public-methods
'''Persistence API.'''

from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    import torch

@typing.runtime_checkable
class Checkpointable(typing.Protocol):
    '''Protocol for a model that can save and load.'''
    def state_dict(self) -> typing.Mapping[str, 'torch.Tensor']: ...
    def load_state_dict(self, state_dict: typing.Mapping[str, typing.Any]) -> typing.Any: ...


# publich functions
def save(
        model: Checkpointable,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        best_value: float,
        fpath: str
    ) -> None:
    '''
    Save model/optimizer/scheduler states.
    '''

    # save states to file
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'best_value': best_value,
    }
    torch.save(state, fpath)

def load(
        model: Checkpointable,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        fpath: str,
        device: str
    ) -> float:
    '''
    Load previously saved state dicts for model, optimizer,
    scheduler, return best metric.
    '''

    # load states from file
    checkpoint = torch.load(fpath, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if checkpoint.get('optimizer'):
        optimizer.load_state_dict(checkpoint['optimizer'])
    if checkpoint.get('scheduler') and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint.get('best_value', -float('inf'))
