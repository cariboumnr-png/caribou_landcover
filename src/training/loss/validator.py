'''Loss config validation type guard.'''

# standard imports
import typing
# local imports
import training.loss

def is_loss_types(cfg: dict) -> typing.TypeGuard[training.loss.LossTypes]:
    '''Loss config validation. Grows with according TypedDict.'''

    # current types: focal, dice
    has_focal = False
    has_dice = False
    # check if each exists
    if _is_focal(cfg.get('focal')):
        has_focal = True
    elif _is_dice(cfg.get('dice')):
        has_dice = True
    # needs to contain at least one of the types
    return has_focal or has_dice

def _is_focal(d: dict | None) -> bool:
    if d is None:
        return False
    return (
        isinstance(d.get('weight'), float) and
        _is_alpha(d.get('alpha')) and
        isinstance(d.get('gamma'), float) and
        isinstance(d.get('reduction'), str)
    )

def _is_dice(d: dict | None) -> bool:
    if d is None:
        return False
    return (
        isinstance(d.get('weight'), float) and
        isinstance(d.get('smooth'), float)
    )

def _is_alpha(a: list | None) -> bool:
    # can be a list of float or just None
    return (
        a is None or
        (isinstance(a, list) and all(isinstance(x, float) for x in a))
    )
