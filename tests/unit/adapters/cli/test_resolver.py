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
Unit tests for `landseg.adapters.cli.resolver`.
'''

# third-party imports
import omegaconf
import pytest
# local imports
import landseg.adapters.cli.resolver as resolver_mod


# ----- `resolve_configs` tests
def test_resolve_configs_base(tmp_path):
    '''
    Given: A minimal `DictConfig` with valid mock file paths.
    When: Calling `resolve_configs` with `use_additional_settings=False`.
    Then: Resolve OmegaConf structure, set `cli_mode=True`, and validate.
    '''
    dev_img = tmp_path / 'dev_img.tif'
    dev_lbl = tmp_path / 'dev_lbl.tif'
    cfg_json = tmp_path / 'cfg.json'
    for f in (dev_img, dev_lbl, cfg_json):
        f.write_text('data')

    cfg_dict = omegaconf.OmegaConf.create({
        'foundation': {
            'datablocks': {
                'name': 'test_ds',
                'filepaths': {
                    'dev_image': str(dev_img),
                    'dev_label': str(dev_lbl),
                    'config': str(cfg_json),
                },
            },
            'grid': {
                'mode': 'ref',
                'crs': 'EPSG:32617',
                'extent': {'filepath': str(dev_img)},
            },
        },
        'session': {
            'orchestration': {
                'single_phase': {'num_epochs': 5},
            },
        },
    })

    root = resolver_mod.resolve_configs(
        config=cfg_dict,
        use_additional_settings=False,
    )

    assert root.execution.cli_mode is True
    assert root.foundation.datablocks.name == 'test_ds'
    assert root.session.orchestration.single_phase.num_epochs == 5


def test_resolve_configs_missing_user_file():
    '''
    Given: A custom non-existent user config path specified in `execution.user_cfg`.
    When: `resolve_configs` is executed with `use_additional_settings=True`.
    Then: Raise a FileNotFoundError indicating missing user config file.
    '''
    cfg_dict = omegaconf.OmegaConf.create({
        'execution': {'user_cfg': '/path/missing_user.yaml'},
    })

    with pytest.raises(FileNotFoundError, match='configuration file not found'):
        resolver_mod.resolve_configs(
            config=cfg_dict,
            use_additional_settings=True,
        )
