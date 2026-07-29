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
Unit tests for `landseg.configs.schema.utils`.
'''

# third-party imports
import pytest
# local imports
import landseg.configs.schema.utils as utils


# ----- `file_exists` & `must_exist` tests
def test_file_exists_and_must_exist(tmp_path) -> None:
    '''
    Given: Existing and non-existent file paths.
    When: `file_exists` and `must_exist` helper functions are called.
    Then: Return boolean existence flags or raise FileNotFoundError for missing path.
    '''
    dummy_file = tmp_path / 'test.txt'
    dummy_file.write_text('content')

    assert utils.file_exists(str(dummy_file)) is True
    assert utils.file_exists(str(tmp_path / 'missing.txt')) is False

    # must_exist should pass silently for existing file or None
    utils.must_exist(str(dummy_file), 'dummy')
    utils.must_exist(None, 'none_path')

    # must_exist should raise FileNotFoundError for missing path
    with pytest.raises(FileNotFoundError, match='File \\[missing\\] is invalid'):
        utils.must_exist(str(tmp_path / 'missing.txt'), 'missing')


# ----- `must_within` tests
def test_must_within_validation() -> None:
    '''
    Given: Numeric values, non-numeric values, and lower/upper range boundaries.
    When: `must_within` validation function is executed.
    Then: Pass valid/non-numeric inputs and raise ValueError for out-of-bounds numbers.
    '''
    # non-numeric values should return early without error
    utils.must_within('string', 'tag', 0, 10)
    utils.must_within(None, 'tag', 0, 10)

    # valid numbers within bounds
    utils.must_within(5, 'tag', 0, 10)
    utils.must_within(0.5, 'tag', 0.0, 1.0)

    # out of bounds lower
    with pytest.raises(ValueError, match='Value \\[tag\\] must be within'):
        utils.must_within(-1, 'tag', 0, 10)

    # out of bounds upper
    with pytest.raises(ValueError, match='Value \\[tag\\] must be within'):
        utils.must_within(11, 'tag', 0, 10)
