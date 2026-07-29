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
Unit tests for `landseg.configs.schema.sections.foundation`.
'''

# third-party imports
import pytest
# local imports
import landseg.configs.schema.sections.foundation as foundation


# ----- `_TileSpecs` tests
def test_tile_specs_validation():
    '''
    Given: `_TileSpecs` instances with square or non-square dimensions.
    When: `_TileSpecs.validate()` is invoked.
    Then: Accept valid square configs or raise ValueError for invalid.
    '''
    tile_specs = foundation._TileSpecs(
        size_row=256,
        size_col=256,
        overlap_row=0,
        overlap_col=0,
    )
    tile_specs.validate()

    with pytest.raises(ValueError, match='Only square blocks are supported'):
        foundation._TileSpecs(size_row=256, size_col=512).validate()

    with pytest.raises(ValueError, match='Only equal row/column stride'):
        foundation._TileSpecs(
            size_row=256,
            size_col=256,
            overlap_row=10,
            overlap_col=20,
        ).validate()

    with pytest.raises(ValueError, match='Block size must be positive'):
        foundation._TileSpecs(size_row=0, size_col=0).validate()

    with pytest.raises(ValueError, match='stride must be zero or positive'):
        foundation._TileSpecs(
            size_row=256,
            size_col=256,
            overlap_row=-1,
            overlap_col=-1,
        ).validate()


# ----- `_Grid` tests
def test_grid_validation(tmp_path):
    '''
    Given: `_Grid` instances across `ref`, `aoi`, and `tiles` modes.
    When: `_Grid.validate()` is called with valid or invalid parameters.
    Then: Pass valid grid definitions and raise ValueError for invalid.
    '''
    ref_raster = tmp_path / 'ref.tif'
    ref_raster.write_text('raster_data')

    # ref mode validation
    grid_ref = foundation._Grid(
        mode='ref',
        crs='EPSG:32617',
        extent=foundation._Extent(filepath=str(ref_raster)),
    )
    grid_ref.validate()
    assert grid_ref.tile_specs_tuple == (256, 256, 0, 0)

    # aoi mode validation
    grid_aoi = foundation._Grid(
        mode='aoi',
        crs='epsg:4326',
        extent=foundation._Extent(
            pixel_size=(10.0, 10.0),
            grid_extent=(1.0, 1.0, 100.0, 100.0),
        ),
    )
    grid_aoi.validate()

    # tiles mode validation
    grid_tiles = foundation._Grid(
        mode='tiles',
        crs='EPSG:26917',
        extent=foundation._Extent(
            pixel_size=(10.0, 10.0),
            grid_shape=(100, 100),
        ),
    )
    grid_tiles.validate()

    # invalid mode
    with pytest.raises(ValueError, match='Invalid grid mode'):
        foundation._Grid(mode='invalid', crs='EPSG:32617').validate()

    # invalid CRS format
    with pytest.raises(ValueError, match='Invalid CRS identifier'):
        foundation._Grid(
            mode='ref',
            crs='WGS84',
            extent=foundation._Extent(filepath=str(ref_raster)),
        ).validate()

    # invalid extent for aoi mode (missing pixel size)
    with pytest.raises(ValueError, match='Pixel size has zero'):
        foundation._Grid(
            mode='aoi',
            crs='EPSG:32617',
            extent=foundation._Extent(pixel_size=(0.0, 10.0)),
        ).validate()

    # invalid extent for tiles mode (grid shape has zero)
    with pytest.raises(ValueError, match='Grid shape has zero'):
        foundation._Grid(
            mode='tiles',
            crs='EPSG:32617',
            extent=foundation._Extent(
                pixel_size=(10.0, 10.0),
                grid_shape=(0, 100),
            ),
        ).validate()


# ----- `_Domains` tests
def test_domains_management(tmp_path):
    '''
    Given: Raster file for domain mapping and index base config.
    When: Instantiating `_DomainMap` sub-configuration object.
    Then: Auto-parse domain name and raise TypeError for bad base type.
    '''
    dom_raster = tmp_path / 'eco_region.tif'
    dom_raster.write_text('raster')

    domains = foundation._Domains()
    domains.add_domain(str(dom_raster), index_base=1)
    assert len(domains.files) == 1
    assert domains.files[0].path == str(dom_raster)

    # type check on index_base
    with pytest.raises(TypeError, match='Index base must be \\[int\\]'):
        domains.add_domain(str(dom_raster), index_base='1')  # type: ignore

    domains.validate()
    assert domains.files[0].name == 'eco_region'


# ----- `_DataBlocks` & `DataFoundation` tests
def test_datablocks_and_foundation_validation(tmp_path):
    '''
    Given: Valid file paths for data blocks and grid settings.
    When: `_DataBlocks.validate()` and `DataFoundation.validate()` run.
    Then: Ensure test data availability and validate foundation config.
    '''
    dev_img = tmp_path / 'dev_img.tif'
    dev_lbl = tmp_path / 'dev_lbl.tif'
    cfg_json = tmp_path / 'cfg.json'
    test_img = tmp_path / 'test_img.tif'
    test_lbl = tmp_path / 'test_lbl.tif'

    for f in (dev_img, dev_lbl, cfg_json, test_img, test_lbl):
        f.write_text('data')

    fps = foundation._FilePaths(
        dev_image=str(dev_img),
        dev_label=str(dev_lbl),
        config=str(cfg_json),
        test_image=str(test_img),
        test_label=str(test_lbl),
    )
    blocks = foundation._DataBlocks(name='test_blocks', filepaths=fps)
    assert blocks.has_test_data is True
    blocks.validate()

    # un-named data blocks failure
    with pytest.raises(ValueError, match='Input data name not provided'):
        foundation._DataBlocks(name='', filepaths=fps).validate()

    grid = foundation._Grid(
        mode='ref',
        crs='EPSG:32617',
        extent=foundation._Extent(filepath=str(dev_img)),
    )
    data_foundation = foundation.DataFoundation(grid=grid, datablocks=blocks)
    data_foundation.validate()
