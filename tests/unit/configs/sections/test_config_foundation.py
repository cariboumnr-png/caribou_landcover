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
Unit tests for `landseg.configs.schema.sections.data`.
'''

# third-party imports
import pytest
# local imports
import landseg.configs.schema.sections.data as data


# ----- `_TileSpecs` tests
def test_tile_specs_validation():
    '''
    Given: `_TileSpecs` instances with square or non-square dimensions.
    When: `_TileSpecs.validate()` is invoked.
    Then: Accept valid square configs or raise ValueError for invalid.
    '''
    tile_specs = data._TileSpecs(
        size_row=256,
        size_col=256,
        overlap_row=0,
        overlap_col=0,
    )
    tile_specs.validate()

    with pytest.raises(ValueError, match='Only square blocks are supported'):
        data._TileSpecs(size_row=256, size_col=512).validate()

    with pytest.raises(ValueError, match='Only equal row/column stride'):
        data._TileSpecs(
            size_row=256,
            size_col=256,
            overlap_row=10,
            overlap_col=20,
        ).validate()

    with pytest.raises(ValueError, match='Block size must be positive'):
        data._TileSpecs(size_row=0, size_col=0).validate()

    with pytest.raises(ValueError, match='stride must be zero or positive'):
        data._TileSpecs(
            size_row=256,
            size_col=256,
            overlap_row=-1,
            overlap_col=-1,
        ).validate()


# ----- `_Grid` tests
def test_grid_validation():
    '''
    Given: `_Grid` instances with valid or invalid parameters.
    When: `_Grid.validate()` is called.
    Then: Pass valid ref grid definitions and raise ValueError for non-ref modes.
    '''
    grid_ref = data._Grid(
        mode='ref',
        crs='EPSG:32617',
    )
    grid_ref.validate()
    assert grid_ref.tile_specs_tuple == (256, 256, 0, 0)

    # non-ref mode raises error
    with pytest.raises(ValueError, match='Invalid grid mode'):
        data._Grid(mode='aoi', crs='EPSG:32617').validate()

    with pytest.raises(ValueError, match='Invalid grid mode'):
        data._Grid(mode='invalid', crs='EPSG:32617').validate()

    # invalid CRS format
    with pytest.raises(ValueError, match='Invalid CRS identifier'):
        data._Grid(mode='ref', crs='WGS84').validate()


# ----- `_Domains` tests
def test_domains_management():
    '''
    Given: Default `_Domains` configuration object.
    When: `_Domains.validate()` is called.
    Then: Validate threshold settings.
    '''
    domains = data._Domains(valid_threshold=0.7, target_variance=0.9)
    domains.validate()
    assert domains.valid_threshold == 0.7


# ----- `_DataBlocks` & `_IngestionCfg` tests
def test_datablocks_and_data_validation():
    '''
    Given: Valid `_DataBlocks` instance.
    When: `_DataBlocks.validate()` and `_IngestionCfg.validate()` run.
    Then: Validate data config.
    '''
    blocks = data._DataBlocks(name='test_blocks')
    blocks.validate()

    grid = data._Grid(
        mode='ref',
        crs='EPSG:32617',
    )
    df = data._IngestionCfg(grid=grid, datablocks=blocks)
    df.validate()
