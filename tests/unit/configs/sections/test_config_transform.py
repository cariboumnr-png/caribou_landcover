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
Unit tests for `landseg.configs.schema.sections.transform`.
'''

# third-party imports
import pytest
# local imports
import landseg.configs.schema.sections.transform as transform


# ----- `DataTransform` tests
def test_data_transform_defaults_and_validation():
    '''
    Given: A default `DataTransform` instance.
    When: Calling `DataTransform.validate()`.
    Then: Initialize default partition ratios and rebuild flags.
    '''
    dt = transform.DataTransform()
    dt.validate()

    assert dt.rebuild is False
    assert dt.partition.val_ratio == 0.1
    assert dt.partition.test_ratio == 0.0
    assert dt.hydration.max_skew_rate == 10.0


def test_catalog_view_validation():
    '''
    Given: `_CatalogView` with valid and out-of-range thresholds.
    When: `_CatalogView.validate()` is called.
    Then: Validate pixel threshold boundaries or raise ValueError.
    '''
    catalog = transform._CatalogView(valid_pxs={'image': 0.8, 'label': 0.95})
    catalog.validate()

    invalid_catalog = transform._CatalogView(valid_pxs={'image': 1.5})
    with pytest.raises(ValueError, match='valid threshold'):
        invalid_catalog.validate()


def test_partition_validation():
    '''
    Given: `_Partition` instances with valid and invalid split ratios.
    When: `_Partition.validate()` is executed.
    Then: Accept valid ratio bounds [0.0, 1.0] or raise ValueError.
    '''
    partition = transform._Partition(val_ratio=0.2, test_ratio=0.1)
    partition.validate()

    with pytest.raises(ValueError, match='validation block ratio'):
        transform._Partition(val_ratio=-0.1).validate()

    with pytest.raises(ValueError, match='test holdout block ratio'):
        transform._Partition(test_ratio=1.5).validate()


def test_scoring_and_hydration_validation():
    '''
    Given: `_Scoring` and `_Hydration` sub-configuration objects.
    When: `.validate()` is called with valid or negative boundaries.
    Then: Pass valid parameters or raise ValueError for negative rates.
    '''
    scoring = transform._Scoring(alpha=0.5, beta=0.5)
    scoring.validate()

    with pytest.raises(ValueError, match='scoring alpha'):
        transform._Scoring(alpha=-1.0).validate()

    hydration = transform._Hydration(max_skew_rate=5.0)
    hydration.validate()

    with pytest.raises(ValueError, match='hydration skew ratio'):
        transform._Hydration(max_skew_rate=-2.0).validate()
