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
import rasterio
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.harmonize as harmonize
import landseg.geopipe.ingest.common as common
import landseg.geopipe.ingest.data_blocks as data_blocks


# ----- pipeline execution
def test_pipeline_run_canonical_blocks(tmp_path, dummy_geotiff_factory):
    '''
    Given: A clean layout, grid parameters, and canonical raster inputs.
    When: Running data blocks building.
    Then: Compile canonical block datasets, manifest catalog,
        and schema.
    '''
    report_file = str(tmp_path / 'ingest_report.json')
    logger = common.IngestionLogger(
        name='test_ingest_canonical',
        log_file=report_file,
        enable_file_log=False
    )
    logger.init_summary(
        run_id='test_run_canonical',
        timestamp='2026-07-08T18:00:00Z'
    )

    img = dummy_geotiff_factory(
        filename='comp_11band.tif',
        width=256,
        height=256,
        bands=11,
        crs='EPSG:3161',
        dtype=numpy.float32
    )
    lbl = dummy_geotiff_factory(
        filename='label.tif',
        width=256,
        height=256,
        bands=1,
        crs='EPSG:3161',
        dtype=numpy.uint8
    )
    with rasterio.open(lbl, 'r+') as dataset:
        dataset.set_band_description(1, 'land_cover')
        dataset.update_tags(1, num_cls=2, ignore_cls='[255]')

    # prepare world grid from the reference raster
    grid_config = harmonize.GridParameters(
        mode='ref',
        crs='EPSG:3161',
        ref_fpath=str(img),
        origin=(0.0, 0.0),
        pixel_size=(0.0, 0.0),
        grid_extent=None,
        grid_shape=None,
        tile_specs=(256, 256, 128, 128)
    )
    grid_file = str(tmp_path / 'grid.json')
    world_grid = harmonize.prepare_world_grid(
        grid_file,
        grid_config,
        policy=artifacts.LifecyclePolicy.REBUILD,
    )

    # initialize pipeline path containers in temp output directory
    paths = artifacts.IngestionPaths(str(tmp_path))

    # set pipeline configurations
    config = data_blocks.BlockBuildingParameters(
        image_fpath=str(img),
        label_fpath=str(lbl),
        dem_pad=8,
        ignore_index=255,
        stage='canonical',
    )

    # run the pipeline
    data_blocks.run_blocks_building(
        world_grid,
        paths.data_blocks,
        config,
        policy=artifacts.LifecyclePolicy.BUILD_IF_MISSING,
        logger=logger
    )

    # verify outputs
    db_paths = paths.data_blocks
    assert os.path.exists(db_paths.catalog)
    assert os.path.exists(db_paths.schema)
    assert os.path.exists(db_paths.blocks)

    # read and inspect catalog
    with open(db_paths.catalog, 'r', encoding='UTF-8') as f:
        catalog_data = json.load(f)
    assert len(catalog_data) > 0
    first_key = list(catalog_data.keys())[0]
    assert 'block_name' in catalog_data[first_key]
    assert 'file_path' in catalog_data[first_key]

    # read and inspect report
    assert logger.summary is not None
    assert 'data_blocks' in logger.summary
    assert 'canonical' in logger.summary['data_blocks']
    report = logger.summary['data_blocks']['canonical']
    assert report['image_filepath'] == str(img)
    assert report['label_filepath'] == str(lbl)
