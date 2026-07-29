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
Unit tests for `landseg.artifacts.controller`.
'''

# third-party imports
import numpy
import pytest
# local imports
import landseg.artifacts.controller as ctrl_mod
import landseg.artifacts.policy as policy_mod


# ----- `Controller` JSON tests
def test_json_persist_fetch_and_properties(tmp_path):
    '''
    Given: A target JSON file path and serializable dictionary payload.
    When: `Controller.persist()` and `Controller.fetch()` are executed.
    Then: Compute valid SHA-256 hash, update `_hash.json`, and return data.
    '''
    json_path = str(tmp_path / 'data.json')
    ctrl = ctrl_mod.Controller[dict](json_path)

    assert ctrl.is_valid is False
    data_in = {'name': 'test_artifact', 'values': [1, 2, 3]}
    ctrl.persist(data_in)

    assert ctrl.is_valid is True
    assert len(ctrl.sha256) == 64
    assert isinstance(ctrl.creation_time, str)

    loaded = ctrl.fetch()
    assert loaded == data_in


# ----- `Controller` NPZ dict tests
def test_npz_dict_persist_and_fetch(tmp_path):
    '''
    Given: A target NPZ file path and a dictionary of NumPy arrays with tuple keys.
    When: `Controller.persist()` and `Controller.fetch()` are executed.
    Then: Persist to compressed NPZ and reconstruct arrays with original keys.
    '''
    npz_path = str(tmp_path / 'tiles.npz')
    ctrl = ctrl_mod.Controller[dict](npz_path)

    arr1 = numpy.array([1, 2, 3])
    arr2 = numpy.array([4, 5, 6])
    data_in = {(0, 0): arr1, (0, 1): arr2}

    ctrl.persist(data_in)
    assert ctrl.is_valid is True

    loaded = ctrl.fetch()
    assert loaded is not None
    assert (0, 0) in loaded
    assert numpy.array_equal(loaded[(0, 0)], arr1)
    assert numpy.array_equal(loaded[(0, 1)], arr2)


def test_npz_write_validation(tmp_path):
    '''
    Given: An empty dictionary or invalid non-array dictionary values.
    When: Calling `Controller._npz_write_dict`.
    Then: Raise a ValueError indicating invalid input payload format.
    '''
    npz_path = str(tmp_path / 'invalid.npz')
    ctrl = ctrl_mod.Controller[dict](npz_path)

    with pytest.raises(ValueError, match='Cannot save empty dict'):
        ctrl._npz_write_dict(npz_path, {})

    with pytest.raises(ValueError, match='Input source must be a dictionary'):
        ctrl._npz_write_dict(npz_path, {'key': 'not_an_array'})  # type: ignore


# ----- `Controller` lifecycle policies tests
def test_lifecycle_policies_fetch(tmp_path):
    '''
    Given: Artifact controllers configured with `LOAD_OR_FAIL`, `BUILD_IF_MISSING`, or `REBUILD`.
    When: Invoking `Controller.fetch()`.
    Then: Enforce policy rules (raise error, return None, or bypass cache).
    '''
    json_path = str(tmp_path / 'policy_test.json')

    # LOAD_OR_FAIL on missing file
    ctrl_fail = ctrl_mod.Controller.load_json_or_fail(json_path)
    with pytest.raises(ctrl_mod.ArtifactError, match='Required artifact is missing'):
        ctrl_fail.fetch()

    # BUILD_IF_MISSING on missing file
    ctrl_build = ctrl_mod.Controller(
        file_path=json_path,
        policy=policy_mod.LifecyclePolicy.BUILD_IF_MISSING,
    )
    assert ctrl_build.fetch() is None

    # REBUILD policy on existing file
    ctrl_build.persist({'status': 'ok'})
    ctrl_rebuild = ctrl_mod.Controller(
        file_path=json_path,
        policy=policy_mod.LifecyclePolicy.REBUILD,
    )
    assert ctrl_rebuild.fetch() is None


# ----- Integrity mismatch tests
def test_controller_hash_mismatch_and_corruption(tmp_path):
    '''
    Given: An artifact file whose contents have been modified or corrupted on disk.
    When: `Controller.fetch()` is executed.
    Then: Detect hash mismatch or decode failure and raise `ArtifactError`.
    '''
    json_path = str(tmp_path / 'tampered.json')
    ctrl = ctrl_mod.Controller[dict](json_path)
    ctrl.persist({'key': 'original'})

    # modify file content without updating hash record
    with open(json_path, 'w', encoding='UTF-8') as f:
        f.write('{"key": "tampered"}')

    assert ctrl.is_valid is False
    with pytest.raises(ctrl_mod.ArtifactError):
        ctrl.fetch()

    # corrupt file contents with invalid JSON
    with open(json_path, 'w', encoding='UTF-8') as f:
        f.write('invalid json content...')

    with pytest.raises(ctrl_mod.ArtifactError):
        ctrl.fetch()
