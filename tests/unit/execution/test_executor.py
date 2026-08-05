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

'''Unit tests for pipeline executor logic.'''

# pylint: disable=protected-access

# standard imports
import os
# third-party imports
import pytest
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.configs.schema.sections as secs
import landseg.execution.executor as executor


# ----- _normalize_val helper
def test_normalize_val_primitives():
    '''
    Given: Primitive values (int, bool, simple string).
    When: Normalizing values.
    Then: Return the values unmodified.
    '''
    assert executor._normalize_val(42) == 42
    assert executor._normalize_val(True) is True
    assert executor._normalize_val("hello") == "hello"


def test_normalize_val_paths():
    '''
    Given: String values that look like paths (contain slashes or
        start with dot).
    When: Normalizing values.
    Then: Return absolute paths with forward slashes.
    '''
    normalized = executor._normalize_val("./test/path")
    expected = os.path.abspath("./test/path").replace('\\', '/')
    assert normalized == expected

    normalized_backslash = executor._normalize_val("test\\path")
    expected_backslash = os.path.abspath("test\\path").replace('\\', '/')
    assert normalized_backslash == expected_backslash


def test_normalize_val_recursive():
    '''
    Given: A dictionary containing nested dicts and lists with
        path-like strings.
    When: Normalizing values recursively.
    Then: Paths are resolved inside all collections.
    '''
    input_data = {
        'path1': './test',
        'list': ['./test1', 42],
        'nested': {'p2': 'foo/bar'}
    }
    norm = executor._normalize_val(input_data)
    assert norm['path1'] == os.path.abspath('./test').replace('\\', '/')
    assert norm['list'][0] == os.path.abspath('./test1').replace('\\', '/')
    assert norm['list'][1] == 42
    assert norm['nested']['p2'] == os.path.abspath('foo/bar').replace('\\', '/')


# ----- _diff_configs helper
def test_diff_configs_identical():
    '''
    Given: Two identical configuration dictionaries.
    When: Diffing the configurations.
    Then: Return an empty dictionary indicating no differences.
    '''
    dict1 = {'a': 1, 'b': {'c': 2}}
    dict2 = {'a': 1, 'b': {'c': 2}}
    assert not executor._diff_configs(dict1, dict2)


def test_diff_configs_different_values():
    '''
    Given: Two dictionaries with overlapping keys but different values.
    When: Diffing the configurations.
    Then: Return a dictionary with paths pointing to tuples of
        (val1, val2).
    '''
    dict1 = {'a': 1, 'b': {'c': 2}}
    dict2 = {'a': 2, 'b': {'c': 3}}
    diff = executor._diff_configs(dict1, dict2)
    assert diff == {'a': (1, 2), 'b.c': (2, 3)}


def test_diff_configs_missing_keys():
    '''
    Given: Two dictionaries where some keys exist only in one
        dictionary.
    When: Diffing the configurations.
    Then: Missing keys appear as None for the dictionary that lacks them.
    '''
    dict1 = {'a': 1, 'b': 2}
    dict2 = {'b': 2, 'c': 3}
    diff = executor._diff_configs(dict1, dict2)
    assert diff == {'a': (1, None), 'c': (None, 3)}


def test_diff_configs_lists():
    '''
    Given: Dictionaries containing lists that differ in length or
        elements.
    When: Diffing the configurations.
    Then: Lists of different lengths are fully reported, while lists of
        the same length report item-by-item differences.
    '''
    dict1 = {'l1': [1, 2], 'l2': [3, 4]}
    dict2 = {'l1': [1, 2, 3], 'l2': [3, 5]}
    diff = executor._diff_configs(dict1, dict2)
    assert diff == {
        'l1': ([1, 2], [1, 2, 3]),
        'l2[1]': (4, 5)
    }


# ----- _validate_upstream_pipelines logic
def test_validate_upstream_pipelines_entrypoints():
    '''
    Given: Pipeline names that are standalone entrypoints.
    When: Validating upstream pipelines.
    Then: Pass silently without checking upstream reports.
    '''
    config = configs.RootConfig(
        pipeline=secs.PipelineConfig(name='default'),
        foundation=secs.DataFoundation(),
        transform=secs.DataTransform(),
    )
    config.session.orchestration.curriculum.single.phases[0].num_epochs = 1
    # should not raise
    executor._validate_upstream_pipelines(config, 'default')
    executor._validate_upstream_pipelines(config, 'data-harmonize')


def test_validate_upstream_pipelines_missing_etl(tmp_path):
    '''
    Given: A data-ingest pipeline run when no ETL report exists.
    When: Validating upstream pipelines.
    Then: Raise an ArtifactError about missing data-harmonize.
    '''
    config = configs.RootConfig(
        pipeline=secs.PipelineConfig(name='data-ingest'),
        etl=secs.ETLConfig(output_dpath=str(tmp_path)),
        foundation=secs.DataFoundation(output_dpath=str(tmp_path)),
    )
    config.session.orchestration.curriculum.single.phases[0].num_epochs = 1
    with pytest.raises(
        artifacts.ArtifactError,
        match='Upstream pipeline "data-harmonize" has not been executed yet'
    ):
        executor._validate_upstream_pipelines(config, 'data-ingest')


def test_validate_upstream_pipelines_missing_ingest(tmp_path):
    '''
    Given: A pipeline that requires data-ingest to be complete, but no
        report exists.
    When: Validating upstream pipelines.
    Then: Raise an ArtifactError.
    '''
    etl_paths = artifacts.ETLPaths(str(tmp_path))
    artifacts.Controller(etl_paths.report).persist({'status': 'SUCCESS'})

    config = configs.RootConfig(
        pipeline=secs.PipelineConfig(name='data-prepare'),
        etl=secs.ETLConfig(output_dpath=str(tmp_path)),
        foundation=secs.DataFoundation(output_dpath=str(tmp_path)),
        transform=secs.DataTransform(),
    )
    config.session.orchestration.curriculum.single.phases[0].num_epochs = 1
    with pytest.raises(
        artifacts.ArtifactError,
        match='Upstream pipeline "data-ingest" has not been executed yet'
    ):
        executor._validate_upstream_pipelines(config, 'data-prepare')


def test_validate_upstream_pipelines_failed_ingest(tmp_path):
    '''
    Given: A pipeline that requires data-ingest, and a failed ingest
        report exists.
    When: Validating upstream pipelines.
    Then: Raise an ArtifactError about the status.
    '''
    etl_paths = artifacts.ETLPaths(str(tmp_path))
    artifacts.Controller(etl_paths.report).persist({'status': 'SUCCESS'})
    foundation_paths = artifacts.FoundationPaths(str(tmp_path))
    ctrl = artifacts.Controller(foundation_paths.report)
    ctrl.persist({'status': 'FAILED'})

    config = configs.RootConfig(
        pipeline=secs.PipelineConfig(name='data-prepare'),
        etl=secs.ETLConfig(output_dpath=str(tmp_path)),
        foundation=secs.DataFoundation(output_dpath=str(tmp_path)),
        transform=secs.DataTransform(),
    )
    config.session.orchestration.curriculum.single.phases[0].num_epochs = 1
    with pytest.raises(
        artifacts.ArtifactError,
        match='Upstream pipeline "data-ingest" status is "FAILED"'
    ):
        executor._validate_upstream_pipelines(config, 'data-prepare')


def test_validate_upstream_pipelines_success(tmp_path):
    '''
    Given: Successful etl, ingest and prepare reports exist.
    When: Validating upstream pipelines for a downstream pipeline.
    Then: Pass silently.
    '''
    etl_paths = artifacts.ETLPaths(str(tmp_path))
    foundation_paths = artifacts.FoundationPaths(str(tmp_path))
    transform_paths = artifacts.TransformPaths(str(tmp_path))

    artifacts.Controller(etl_paths.report).persist({'status': 'SUCCESS'})
    artifacts.Controller(foundation_paths.report).persist({'status': 'SUCCESS'})
    artifacts.Controller(transform_paths.report).persist({'status': 'SUCCESS'})

    config = configs.RootConfig(
        pipeline=secs.PipelineConfig(name='model-train'),
        etl=secs.ETLConfig(output_dpath=str(tmp_path)),
        foundation=secs.DataFoundation(output_dpath=str(tmp_path)),
        transform=secs.DataTransform(output_dpath=str(tmp_path)),
    )
    config.session.orchestration.curriculum.single.phases[0].num_epochs = 1
    # should not raise
    executor._validate_upstream_pipelines(config, 'model-train')


# ----- _compare_config_section logic
def test_compare_config_section_missing_artifact(tmp_path):
    '''
    Given: An artifact path that does not exist.
    When: Comparing a config section.
    Then: Return an empty diff dict.
    '''
    diff = executor._compare_config_section(
        str(tmp_path / "missing.json"),
        "foundation",
        secs.DataFoundation(
            grid=secs.foundation._Grid(mode='grid'), output_dpath='out'
        )
    )
    assert isinstance(diff, dict) and not diff


def test_compare_config_section_with_differences(tmp_path):
    '''
    Given: A valid artifact with a saved configuration that differs from
        the current configuration.
    When: Comparing the config section.
    Then: Return the differences found.
    '''
    artifact_path = str(tmp_path / "config.json")
    saved_config = {'foundation': {'grid': 'old_grid', 'output_dpath': 'out'}}
    artifacts.Controller(artifact_path).persist(saved_config)

    current_config = secs.DataFoundation(
        grid=secs.foundation._Grid(mode='new_grid'), output_dpath='out'
    )

    diff = executor._compare_config_section(
        artifact_path,
        "foundation",
        current_config
    )

    # We expect 'new_grid' to be normalized as an absolute path vs 'old_grid'
    # depending on normalization
    assert 'foundation.grid' in diff


# ----- execute_pipeline logic
def test_execute_pipeline_success(mocker):
    '''
    Given: A valid root configuration and all upstream validation checks
        passing.
    When: Calling execute_pipeline.
    Then: Retrieve the correct pipeline command, execute it with the
        configuration, and return its result.
    '''
    config = configs.RootConfig(
        pipeline=secs.PipelineConfig(name='default'),
        foundation=secs.DataFoundation(),
        transform=secs.DataTransform(),
    )
    config.session.orchestration.curriculum.single.phases[0].num_epochs = 1

    mocker.patch('landseg.execution.executor._validate_upstream_pipelines')
    mocker.patch('landseg.execution.executor._check_config_staleness')

    mock_command = mocker.Mock(return_value='pipeline_result')
    mocker.patch('landseg.execution.pipelines.get', return_value=mock_command)

    result = executor.execute_pipeline(config)

    assert result == 'pipeline_result'
    mock_command.assert_called_once_with(config)
