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
Unit tests for `landseg.configs.schema.sections.models`.
'''

# third-party imports
import pytest
# local imports
import landseg.configs.schema.sections.models as models


# ----- `ModelsConfig` tests
def test_models_config_defaults_and_getters() -> None:
    '''verify `ModelsConfig` defaults and backbone properties.'''
    cfg = models.ModelsConfig()
    cfg.validate()

    assert cfg.model_body == 'unet'
    assert cfg.bottleneck == 'conv'

    # backbone property check
    backbone_cfg = cfg.unet_backbone_config
    assert isinstance(backbone_cfg, models._UNetBackboneConfig)
    assert backbone_cfg.body.body == 'unet'

    # base channel modification
    cfg.set_base_channel(64)
    assert cfg.model_body_registry['unet'].base_ch == 64

    # conditioners getter check
    cfg.conditioners = ['film', 'concat']
    cond_map = cfg.conditioning_config
    assert isinstance(cond_map['film'], models._FiLM)
    assert isinstance(cond_map['concat'], models._Concat)


def test_models_config_validation_failures() -> None:
    '''verify `ModelsConfig` validation error handling.'''
    # invalid model body
    invalid_body_cfg = models.ModelsConfig(model_body='invalid_body')
    with pytest.raises(ValueError, match='Invalid model body'):
        invalid_body_cfg.validate()

    # invalid conditioner
    invalid_cond_cfg = models.ModelsConfig(conditioners=['non_existent'])
    with pytest.raises(ValueError, match='Invalid conditioner'):
        invalid_cond_cfg.validate()

    # invalid clamp range ordering
    invalid_clamp = models.ModelsConfig(
        numeric_safety=models._NumericSafety(clamp_range=(10.0, 1.0)),
    )
    with pytest.raises(ValueError, match='Invalid clamp_range ordering'):
        invalid_clamp.validate()
