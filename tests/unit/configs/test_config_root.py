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

# pylint: disable=protected-access

'''
Unit tests for `landseg.configs.schema.root`.
'''

# local imports
import landseg.configs.schema.root as root_mod
import landseg.configs.schema.sections as sec


# ----- `RootConfig` tests
def test_root_config_defaults_and_as_dict():
    '''
    Given: Default `RootConfig` instantiation parameters.
    When: Instantiating `RootConfig` and calling `.as_dict`.
    Then: Initialize sub-sections and serialize config to dictionary.
    '''
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


def test_root_config_hyperparameter_setters():
    '''
    Given: A default `RootConfig` instance.
    When: Invoking hyperparameter setter helper methods.
    Then: Mutate nested session data loader and optimizer fields.
    '''
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


def test_root_config_validate_all(tmp_path):
    '''
    Given: A `RootConfig` with valid foundation files and session.
    When: `RootConfig.validate_all()` is executed.
    Then: Complete validation across all configuration sub-sections.
    '''
    cfg_json = tmp_path / 'cfg.json'
    cfg_json.write_text('data')

    ref_tif = tmp_path / 'ref.tif'
    ref_tif.write_text('data')

    root = root_mod.RootConfig()
    root.etl.canvas.reference_raster = str(ref_tif)
    root.etl.dataset_config = str(cfg_json)
    root.foundation.datablocks.name = 'test_blocks'
    root.foundation.grid.mode = 'ref'
    root.foundation.grid.crs = 'EPSG:32617'

    root.session.orchestration.single_phase.num_epochs = 10
    root.validate_all()
