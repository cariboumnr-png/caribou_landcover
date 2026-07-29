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
Unit tests for `landseg.configs.schema.sections.study`.
'''

# local imports
import landseg.configs.schema.sections.study as study


# ----- `StudyConfig` tests
def test_study_config_default_instantiation() -> None:
    '''verify `StudyConfig` defaults and search space ranges.'''
    cfg = study.StudyConfig()

    assert isinstance(cfg.base, study._BaseObj)
    assert isinstance(cfg.optimizer, study._OptimizerObj)
    assert isinstance(cfg.architecture, study._ArchitectureObj)

    # range definitions
    assert cfg.base.learning_rate == (1e-5, 1e-1)
    assert cfg.optimizer.weight_decay == (1e-6, 1e-2)
    assert cfg.architecture.model_body == study.MODEL_BODIES
    assert cfg.architecture.bottleneck == study.BOTTLENECKS


def test_study_config_custom_objective() -> None:
    '''verify `StudyConfig` customization of sweep spaces.'''
    custom_arch = study._ArchitectureObj(
        model_body=['unet'],
        base_channel=(32, 64, 32),
    )
    cfg = study.StudyConfig(architecture=custom_arch)

    assert cfg.architecture.model_body == ['unet']
    assert cfg.architecture.base_channel == (32, 64, 32)
