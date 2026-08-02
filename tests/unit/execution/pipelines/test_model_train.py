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
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring

'''
Unit tests for model train pipeline (model_train.py).
'''

# standard imports
import typing
# third-party imports
import omegaconf
import pytest
# local imports
import landseg.configs as configs
import landseg.core as core
import landseg.execution.pipelines.model_train as train_pipeline


# ----- `_parse_verbosity` helper
@pytest.mark.parametrize('verbosity, expected', [
    ('full', 10),
    ('select', 20),
    ('silent', None),
])
def test_parse_verbosity_valid(verbosity: str, expected: int | None):
    '''
    Given: A valid verbosity string.
    When: `_parse_verbosity` is called.
    Then: Return the corresponding logging level integer or None.
    '''
    assert train_pipeline._parse_verbosity(verbosity) == expected


def test_parse_verbosity_invalid_raises_value_error():
    '''
    Given: An invalid verbosity string.
    When: `_parse_verbosity` is called.
    Then: Raise a ValueError.
    '''
    with pytest.raises(ValueError, match='Invalid option'):
        train_pipeline._parse_verbosity('invalid')


# ----- `_get_device_name` helper
def test_get_device_name_returns_string():
    '''
    Given: System execution device.
    When: `_get_device_name` is called.
    Then: Return a non-empty string representing device.
    '''
    device_name = train_pipeline._get_device_name()
    assert isinstance(device_name, str)
    assert len(device_name) > 0


# ----- `_build_session_runner` helper
def test_build_session_runner_invalid_mode_raises_value_error(
    tmp_path, dataspecs
):
    '''
    Given: A RootConfig with an invalid training session mode.
    When: `_build_session_runner` is called.
    Then: Raise a ValueError.
    '''
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = str(tmp_path)
    schema.session.mode = 'invalid'

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )

    class DummyModel:
        def parameters(self):
            return []

    mock_model = typing.cast(core.MultiheadModelLike, DummyModel())
    paths = train_pipeline.artifacts.SessionPaths(str(tmp_path / 'results'))
    paths.init()
    logger = train_pipeline.session.SessionLogger(
        'test', log_file=paths.summary
    )

    with pytest.raises(ValueError, match='Invalid training mode'):
        train_pipeline._build_session_runner(
            config, dataspecs, mock_model, paths, logger
        )


# ----- `train` pipeline test
@pytest.mark.parametrize('mode', ['continuous', 'curriculum'])
def test_train_pipeline_success(tmp_path, dataspecs, monkeypatch, mode: str):
    '''
    Given: A valid RootConfig and mock dependencies for continuous/curriculum.
    When: `train` is called.
    Then: Execute training runner and record results cleanly.
    '''
    exp_root = str(tmp_path / 'exp')
    schema = omegaconf.OmegaConf.structured(configs.RootConfig)
    schema.execution.exp_root = exp_root
    schema.session.mode = mode

    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(schema)
    )

    class DummyRunner:
        def execute(self):
            return 0.91

    class DummyModel:
        def parameters(self):
            return []

    mock_model = typing.cast(core.MultiheadModelLike, DummyModel())

    monkeypatch.setattr(
        train_pipeline.geopipe,
        'build_dataspec',
        lambda *a, **kw: dataspecs
    )
    monkeypatch.setattr(
        train_pipeline.models,
        'build_multihead_unet',
        lambda *a, **kw: mock_model,
    )
    monkeypatch.setattr(
        train_pipeline.session.factory,
        'build_continous_training_session',
        lambda *a, **kw: DummyRunner(),
    )
    monkeypatch.setattr(
        train_pipeline.session.factory,
        'build_curriculum_training_session',
        lambda *a, **kw: DummyRunner(),
    )

    train_pipeline.train(config)
