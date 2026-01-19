# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=too-few-public-methods
'''
Callback-facing trainer class protocols. Mimics related behaviours.
'''

# standard imports
from __future__ import annotations
import typing
# local imports
import training.common

if typing.TYPE_CHECKING:
    import torch

# -------------------------------trainer class-------------------------------
@typing.runtime_checkable
class TrainerLike(typing.Protocol):
    comps: training.common.TrainerComponentsLike
    config: training.common.RuntimeConfigLike
    state: training.common.RuntimeStateLike
    device: str
    def _parse_batch(self) -> None: ...
    def _autocast_ctx(self) -> typing.ContextManager: ...
    def _val_ctx(self) -> typing.ContextManager: ...
    def _compute_loss(self) -> None: ...
    def _update_train_logs(self, bidx: int) -> bool: ...
    def _update_conf_matrix(self) -> None: ...
    def _compute_iou(self) -> None: ...
    def _clip_grad(self) -> None: ...

#
@typing.runtime_checkable
class Checkpointable(typing.Protocol):
    def state_dict(self) -> typing.Mapping[str, 'torch.Tensor']: ...
    def load_state_dict(self, state_dict: typing.Mapping[str, typing.Any]) -> typing.Any: ...
