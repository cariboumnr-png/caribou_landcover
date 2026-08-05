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

'''Unit tests for the canonical block-building pipeline module.'''

# standard imports
import json
import os
# third-party imports
import numpy
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.ingest as ingest_data
import landseg.geopipe.ingest.common as common
import landseg.geopipe.ingest.data_blocks as data_blocks


# ----- pipeline execution
def test_pipeline_run_dev_stage(tmp_path, dummy_data_paths, dummy_geotiff_factory):
    '''
    Given: A clean layout, grid parameters, and development inputs.
    When: Running data blocks building for the dev stage.
    Then: Compile development block datasets, manifest catalogs,
        and schemas.
    '''
    # Setup logger and execution summary
    report_file = str(tmp_path / 'ingest_report.json')
    logger = common.IngestionLogger(
        name='test_ingest_dev',
        log_file=report_file,
        enable_file_log=False
    )
    logger.init_summary(run_id='test_run_dev', timestamp='2026-07-08T18:00:00Z')

    dev_img = dummy_geotiff_factory(
        filename='comp_11band_dev.tif',
        width=256,
        height=256,
        bands=11,
        crs='EPSG:3161',
        dtype=numpy.float32
    )
    dev_lbl = dummy_geotiff_factory(
        filename='label_dev.tif',
        width=256,
        height=256,
        bands=1,
        crs='EPSG:3161',
        dtype=numpy.uint8
    )

    # Prepare world grid from the reference raster
    grid_config = ingest_data.GridParameters(
        mode='ref',
        crs='EPSG:3161',
        ref_fpath=str(dev_img),
        origin=(0.0, 0.0),
        pixel_size=(0.0, 0.0),
        grid_extent=None,
        grid_shape=None,
        tile_specs=(256, 256, 128, 128)
    )
    grid_file = str(tmp_path / 'grid.json')
    world_grid = ingest_data.prepare_world_grid(
        grid_file,
        grid_config,
        policy=artifacts.LifecyclePolicy.REBUILD,
        logger=logger
    )

    # Initialize pipeline path containers in temp output directory
    paths = artifacts.FoundationPaths(str(tmp_path))

    # Set pipeline configurations
    config = data_blocks.BlockBuildingParameters(
        stage='dev',
        image_fpath=str(dev_img),
        label_fpath=str(dev_lbl),
        data_config_fpath=dummy_data_paths.config,
        dem_pad=8,
        ignore_index=255
    )

    # Run the pipeline
    data_blocks.run_blocks_building(
        world_grid,
        paths.data_blocks.dev,
        config,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=logger
    )

    # Verify outputs under dev partition
    dev_paths = paths.data_blocks.dev
    assert os.path.exists(dev_paths.catalog)
    assert os.path.exists(dev_paths.schema)
    assert os.path.exists(dev_paths.blocks)

    # Read and inspect catalog
    with open(dev_paths.catalog, 'r', encoding='UTF-8') as f:
        catalog_data = json.load(f)
    assert len(catalog_data) > 0
    first_key = list(catalog_data.keys())[0]
    assert 'block_name' in catalog_data[first_key]
    assert 'file_path' in catalog_data[first_key]

    # Verify logger records reports correctly
    assert logger.summary is not None
    assert 'data_blocks' in logger.summary
    assert 'dev' in logger.summary['data_blocks']
    report = logger.summary['data_blocks']['dev']
    assert report['image_filepath'] == str(dev_img)
    assert report['label_filepath'] == str(dev_lbl)


def test_pipeline_run_test_stage(tmp_path, dummy_data_paths, dummy_geotiff_factory):
    '''
    Given: A clean layout, grid parameters, and test inputs.
    When: Running data blocks building for the test stage.
    Then: Compile holdout test block datasets, manifests, and catalogs.
    '''
    # Setup logger and execution summary
    report_file = str(tmp_path / 'ingest_report.json')
    logger = common.IngestionLogger(
        name='test_ingest_test',
        log_file=report_file,
        enable_file_log=False
    )
    logger.init_summary(run_id='test_run_test', timestamp='2026-07-08T18:00:00Z')

    test_img = dummy_geotiff_factory(
        filename='comp_11band_test.tif',
        width=256,
        height=256,
        bands=11,
        crs='EPSG:3161',
        dtype=numpy.float32
    )
    test_lbl = dummy_geotiff_factory(
        filename='label_test.tif',
        width=256,
        height=256,
        bands=1,
        crs='EPSG:3161',
        dtype=numpy.uint8
    )

    # Prepare world grid from the reference raster
    grid_config = ingest_data.GridParameters(
        mode='ref',
        crs='EPSG:3161',
        ref_fpath=str(test_img),
        origin=(0.0, 0.0),
        pixel_size=(0.0, 0.0),
        grid_extent=None,
        grid_shape=None,
        tile_specs=(256, 256, 128, 128)
    )
    grid_file = str(tmp_path / 'grid.json')
    world_grid = ingest_data.prepare_world_grid(
        grid_file,
        grid_config,
        policy=artifacts.LifecyclePolicy.REBUILD,
        logger=logger
    )

    # Initialize pipeline path containers in temp output directory
    paths = artifacts.FoundationPaths(str(tmp_path))

    # Set pipeline configurations
    config = data_blocks.BlockBuildingParameters(
        stage='test',
        image_fpath=str(test_img),
        label_fpath=str(test_lbl),
        data_config_fpath=dummy_data_paths.config,
        dem_pad=8,
        ignore_index=255
    )

    # Run the pipeline
    data_blocks.run_blocks_building(
        world_grid,
        paths.data_blocks.test,
        config,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=logger
    )

    # Verify outputs under test partition
    test_paths = paths.data_blocks.test
    assert os.path.exists(test_paths.catalog)
    assert os.path.exists(test_paths.schema)
    assert os.path.exists(test_paths.blocks)

    # Read and inspect catalog
    with open(test_paths.catalog, 'r', encoding='UTF-8') as f:
        catalog_data = json.load(f)
    assert len(catalog_data) > 0
    first_key = list(catalog_data.keys())[0]
    assert 'block_name' in catalog_data[first_key]
    assert 'file_path' in catalog_data[first_key]

    # Verify logger records reports correctly
    assert logger.summary is not None
    assert 'data_blocks' in logger.summary
    assert 'test' in logger.summary['data_blocks']
    report = logger.summary['data_blocks']['test']
    assert report['image_filepath'] == str(test_img)
    assert report['label_filepath'] == str(test_lbl)
