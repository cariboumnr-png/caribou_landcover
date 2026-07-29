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
Unit tests for diagnose overfit pipeline (diagnose_overfit.py).
'''

# standard imports
import os
import shutil
import typing
# third-party imports
import omegaconf
# local imports
import landseg.configs as configs
import landseg.core as core
import landseg.execution.pipelines.diagnose_overfit as overfit_pipeline
import landseg.session as session


# ----- `_prepare_dataspecs` helper
def test_prepare_dataspecs(tmp_path, dummy_block):
    '''
    Given: A RootConfig and an existing block in output directory.
    When: `_prepare_dataspecs` is called.
    Then: Return a valid DataSpecs instance with correct shape and meta.
    '''
    block, fpath = dummy_block
    overfit_dpath = str(tmp_path / 'results' / 'overfit_test')
    os.makedirs(overfit_dpath, exist_ok=True)
    shutil.copy(fpath, os.path.join(overfit_dpath, 'test_block.npz'))

    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    cfg_schema.execution.exp_root = str(tmp_path)
    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    logger = session.SessionLogger(
        'test_overfit', log_file=f'{overfit_dpath}/test.json'
    )
    specs = overfit_pipeline._prepare_dataspecs(
        overfit_dpath, config, logger=logger
    )

    assert isinstance(specs, core.DataSpecs)
    assert specs.mode == 'single'
    assert specs.meta.image_specs.num_channels == block.data.image.shape[0]
    assert specs.meta.image_specs.height_width == block.data.image.shape[1]
    assert 'base' in specs.heads.class_counts
    assert specs.domains.ids_num == 0
    assert specs.domains.vec_dim == 0


# ----- `overfit` pipeline mock test
def test_overfit_pipeline_with_existing_block(tmp_path, dummy_block):
    '''
    Given: A RootConfig and an existing block in output directory.
    When: `overfit` pipeline executes.
    Then: Successfully load existing block, set up model and run overfit.
    '''
    _, fpath = dummy_block
    exp_root = str(tmp_path / 'exp')
    overfit_dpath = f'{exp_root}/results/overfit_test'
    os.makedirs(overfit_dpath, exist_ok=True)
    shutil.copy(fpath, os.path.join(overfit_dpath, 'test_block.npz'))

    # compose config with OmegaConf
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    cfg_schema.execution.exp_root = exp_root

    # override session & models for fast overfit test
    cfg_schema.session.orchestration.monitor.track_heads = {'base': 1.0}
    cfg_schema.session.data_loader.patch_size = 16
    cfg_schema.models.model_body_registry.unet.base_ch = 8

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    overfit_pipeline.overfit(config)

    assert os.path.exists(f'{overfit_dpath}/log/overfit.log')
    assert os.path.exists(f'{overfit_dpath}/log/overfit_summary.json')
