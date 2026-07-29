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
Unit tests for `landseg.configs.schema.root`.
'''

# local imports
import landseg.configs.schema.root as root_mod
import landseg.configs.schema.sections as sec


# ----- `RootConfig` tests
def test_root_config_defaults_and_as_dict() -> None:
    '''verify `RootConfig` default initializations and dictionary conversion.'''
    root = root_mod.RootConfig()

    assert isinstance(root.execution, root_mod._ExecutionContext)
    assert isinstance(root.foundation, sec.DataFoundation)
    assert isinstance(root.transform, sec.DataTransform)
    assert isinstance(root.dataspecs, sec.DataSpecs)
    assert isinstance(root.models, sec.ModelsConfig)
    assert isinstance(root.session, sec.SessionConfig)
    assert isinstance(root.study, sec.StudyConfig)
    assert isinstance(root.pipeline, sec.PipelineConfig)

    # dictionary serialization test
    cfg_dict = root.as_dict
    assert isinstance(cfg_dict, dict)
    assert 'execution' in cfg_dict
    assert 'foundation' in cfg_dict
    assert 'models' in cfg_dict


def test_root_config_hyperparameter_setters() -> None:
    '''verify `RootConfig` hyperparameter setter methods.'''
    root = root_mod.RootConfig()

    root.set_data_patch_size(256)
    assert root.session.data_loader.patch_size == 256

    root.set_data_batch_size(32)
    assert root.session.data_loader.batch_size == 32

    root.set_optimizer_lr(5e-4)
    assert root.session.engine_optim.lr == 5e-4

    root.set_optimizer_weight_decay(1e-4)
    assert root.session.engine_optim.weight_decay == 1e-4

    root.set_optimizer_type('Adam')
    assert root.session.engine_optim.opt_cls == 'Adam'


def test_root_config_validate_all(tmp_path) -> None:
    '''verify `RootConfig.validate_all()` execution with valid file structure.'''
    dev_img = tmp_path / 'dev_img.tif'
    dev_lbl = tmp_path / 'dev_lbl.tif'
    cfg_json = tmp_path / 'cfg.json'

    for f in (dev_img, dev_lbl, cfg_json):
        f.write_text('data')

    root = root_mod.RootConfig()
    root.foundation.datablocks.name = 'test_blocks'
    root.foundation.datablocks.filepaths.dev_image = str(dev_img)
    root.foundation.datablocks.filepaths.dev_label = str(dev_lbl)
    root.foundation.datablocks.filepaths.config = str(cfg_json)
    root.foundation.grid.mode = 'ref'
    root.foundation.grid.crs = 'EPSG:32617'
    root.foundation.grid.extent.filepath = str(dev_img)

    root.session.orchestration.single_phase.num_epochs = 10
    root.validate_all()
