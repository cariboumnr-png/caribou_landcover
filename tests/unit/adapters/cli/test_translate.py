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
Unit tests for `landseg.adapters.cli.translate`.
'''

# third-party imports
import omegaconf
# local imports
import landseg.adapters.cli.translate as translate_mod


# ----- `translate_user_config` tests
def test_translate_user_config_data_harmonize():
    '''
    Given: A user configuration `DictConfig` with `data-harmonize` settings.
    When: `translate_user_config` is executed.
    Then: Map fields to harmonization structure including canvas and grid.
    '''
    user_cfg = omegaconf.OmegaConf.create({
        'data-harmonize': {
            'target_crs': 'EPSG:3161',
            'target_resolution': 20.0,
            'grid_crs': 'EPSG:32617',
            'tile_size': 256,
            'output_dpath': '/path/exp/harmonized',
        },
    })
    result = translate_mod.translate_user_config(user_cfg)

    assert result.data.harmonization.canvas.target_crs == 'EPSG:3161'
    assert result.data.harmonization.canvas.target_resolution == 20.0
    assert result.data.harmonization.grid.crs == 'EPSG:32617'
    assert result.data.harmonization.grid.tile_specs.size_row == 256
    assert result.data.harmonization.grid.tile_specs.size_col == 256
    assert result.data.harmonization.output_dpath == '/path/exp/harmonized'


def test_translate_user_config_data_ingest():
    '''
    Given: A user configuration `DictConfig` with `data-ingest` settings.
    When: `translate_user_config` is executed.
    Then: Map fields to `ingestion` datablocks and output_dpath structures.
    '''
    user_cfg = omegaconf.OmegaConf.create({
        'data-ingest': {
            'harmonization_run': 1,
            'output_dpath': '/path/exp/artifacts/foundation',
        },
    })
    result = translate_mod.translate_user_config(user_cfg)

    assert result.data.ingestion.harmonization_run == 1
    assert result.data.ingestion.output_dpath == '/path/exp/artifacts/foundation'


def test_translate_user_config_data_prepare():
    '''
    Given: A user configuration `DictConfig` with `data-prepare` settings.
    When: `translate_user_config` is called.
    Then: Map fields to `transform` partition, catalog, scoring, and output_dpath.
    '''
    user_cfg = omegaconf.OmegaConf.create({
        'data-prepare': {
            'val_ratio': 0.2,
            'test_ratio': 0.1,
            'target_head': 'cover',
            'rebuild': True,
            'output_dpath': '/path/exp/artifacts/transform',
        },
    })
    result = translate_mod.translate_user_config(user_cfg)

    assert result.data.preparation.partition.val_ratio == 0.2
    assert result.data.preparation.partition.test_ratio == 0.1
    assert result.data.preparation.catalog.focal_target == 'cover'
    assert result.data.preparation.rebuild is True
    assert result.data.preparation.output_dpath == '/path/exp/artifacts/transform'


def test_translate_user_config_model_train():
    '''
    Given: A user configuration `DictConfig` with `model-train` settings.
    When: `translate_user_config` is invoked with epochs and active tasks.
    Then: Map fields to models, session output_dpath, and curriculum orchestration phases.
    '''
    user_cfg = omegaconf.OmegaConf.create({
        'model-train': {
            'model_body': 'unet',
            'patch_size': 128,
            'batch_size': 32,
            'epochs': 25,
            'active_tasks': ['cover', 'canopy'],
            'output_dpath': '/path/exp/results',
        },
    })
    result = translate_mod.translate_user_config(user_cfg)

    assert result.models.model_body == 'unet'
    assert result.session.data_loader.patch_size == 128
    assert result.session.data_loader.batch_size == 32
    assert result.session.output_dpath == '/path/exp/results'
    phases = result.session.orchestration.curriculum.single.phases
    assert len(phases) == 1
    assert phases[0].num_epochs == 25
    assert phases[0].active_heads == ['cover', 'canopy']


def test_set_paths_helper():
    '''
    Given: A target dictionary and dot-separated property paths.
    When: Calling internal `_set_paths` helper function.
    Then: Create nested dictionary and assign value to target keys.
    '''
    target: dict = {}
    translate_mod._set_paths(target, ['a.b.c', 'x.y'], 42)

    assert target['a']['b']['c'] == 42
    assert target['x']['y'] == 42
