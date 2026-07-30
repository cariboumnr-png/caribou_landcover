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
Unit tests for `landseg.artifacts.checkpoint`.
'''

# third-party imports
import torch
# local imports
import landseg.artifacts.checkpoint as ckpt_mod


# ----- Checkpoint save and load tests
def test_save_and_load_checkpoint_full(tmp_path):
    '''
    Given: Model, optimizer, scheduler instances, and checkpoint metadata.
    When: Calling `save_checkpoint` and restoring with `load_checkpoint`.
    Then: Restore state dicts and return matching metadata.
    '''
    model_src = _DummyModel()
    optim_src = torch.optim.SGD(model_src.parameters(), lr=0.01)
    sched_src = torch.optim.lr_scheduler.StepLR(optim_src, step_size=5)

    meta_in: ckpt_mod.CheckpointMeta = {'metric': 0.85, 'epoch': 10, 'step': 500}
    ckpt_file = str(tmp_path / 'model.pt')

    ckpt_mod.save_checkpoint(
        model=model_src,
        fpath=ckpt_file,
        ckpt_meta=meta_in,
        optimizer=optim_src,
        scheduler=sched_src,
    )

    model_dst = _DummyModel()
    optim_dst = torch.optim.SGD(model_dst.parameters(), lr=0.01)
    sched_dst = torch.optim.lr_scheduler.StepLR(optim_dst, step_size=5)

    meta_out = ckpt_mod.load_checkpoint(
        model=model_dst,
        fpath=ckpt_file,
        map_device='cpu',
        optimizer=optim_dst,
        scheduler=sched_dst,
    )

    assert meta_out['metric'] == 0.85
    assert meta_out['epoch'] == 10
    assert meta_out['step'] == 500

    # verify model state matches
    for p1, p2 in zip(model_src.parameters(), model_dst.parameters()):
        assert torch.equal(p1, p2)


def test_load_checkpoint_optional_components(tmp_path):
    '''
    Given: A saved model checkpoint file without scheduler.
    When: `load_checkpoint` is called in evaluation mode with model only.
    Then: Successfully load parameters without error and return metadata.
    '''
    model_src = _DummyModel()
    optim_src = torch.optim.SGD(model_src.parameters(), lr=0.01)
    meta_in: ckpt_mod.CheckpointMeta = {'metric': 0.92, 'epoch': 5, 'step': 250}
    ckpt_file = str(tmp_path / 'model_eval.pt')

    ckpt_mod.save_checkpoint(
        model=model_src,
        fpath=ckpt_file,
        ckpt_meta=meta_in,
        optimizer=optim_src,
        scheduler=None,
    )

    model_dst = _DummyModel()
    meta_out = ckpt_mod.load_checkpoint(
        model=model_dst,
        fpath=ckpt_file,
        map_device='cpu',
    )

    assert meta_out['metric'] == 0.92
    assert meta_out['epoch'] == 5
    assert meta_out['step'] == 250


# ----- Dummy PyTorch Model helper
class _DummyModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self):
        '''dummy forward'''
