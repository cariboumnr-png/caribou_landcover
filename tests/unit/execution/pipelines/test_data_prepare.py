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

'''Unit tests for the data preparation execution pipeline.'''

# standard imports
import os
import typing
# third-party imports
import omegaconf
# local imports
import landseg.configs as configs
import landseg.execution.pipelines as pipelines


# ----- pipeline execution
def test_data_prepare_pipeline_success(tmp_path, dummy_data_paths):
    '''
    Given: A RootConfig pointing to valid raster inputs and temporary
        output directories.
    When: Running data ingest followed by data prepare pipelines.
    Then: Correctly partition the data blocks, aggregate image stats,
        normalize blocks, and compile schemas.
    '''
    # compose config with OmegaConf
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)

    # override foundation grid fields
    grid_cfg = cfg_schema.foundation.grid
    grid_cfg.mode = 'ref'
    grid_cfg.crs = 'EPSG:3161'
    grid_cfg.tile_specs.size_row = 256
    grid_cfg.tile_specs.size_col = 256
    grid_cfg.tile_specs.overlap_row = 128
    grid_cfg.tile_specs.overlap_col = 128

    # override foundation datablocks fields
    blocks_cfg = cfg_schema.foundation.datablocks
    blocks_cfg.name = 'test_prepare_run'

    cfg_schema.etl.canvas.reference_raster = dummy_data_paths.extent
    cfg_schema.etl.canvas.target_crs = 'EPSG:3161'
    cfg_schema.etl.canvas.target_resolution = 10.0
    cfg_schema.etl.dataset_config = dummy_data_paths.config
    cfg_schema.etl.output_dpath = str(tmp_path / 'harmonized')
    cfg_schema.etl.raw_data.domains = {
        'domain_1': dummy_data_paths.domain_1
    }
    cfg_schema.etl.raw_data.dev_features = {
        'sentinel2': dummy_data_paths.raw_sentinel2,
        'dem': dummy_data_paths.raw_dem
    }
    cfg_schema.etl.raw_data.dev_labels = {
        'landcover': dummy_data_paths.raw_landcover
    }
    cfg_schema.etl.raw_data.test_features = {
        'sentinel2': dummy_data_paths.raw_test_sentinel2,
        'dem': dummy_data_paths.raw_test_dem
    }
    cfg_schema.etl.raw_data.test_labels = {
        'landcover': dummy_data_paths.raw_test_landcover
    }

    cfg_schema.foundation.output_dpath = str(tmp_path / 'foundation')
    cfg_schema.foundation.rebuild = True

    # override transform fields
    transform_cfg = cfg_schema.transform
    transform_cfg.output_dpath = str(tmp_path / 'transform')
    transform_cfg.rebuild = True

    transform_cfg.catalog.valid_pxs = {'image': 0.05}
    transform_cfg.catalog.focal_target = None

    transform_cfg.partition.val_ratio = 0.2
    transform_cfg.partition.test_ratio = 0.1
    transform_cfg.partition.buffer_step = 1

    transform_cfg.scoring.reward = {0: 1.0}
    transform_cfg.scoring.alpha = 1.0
    transform_cfg.scoring.beta = 0.5

    transform_cfg.hydration.max_skew_rate = 1.5

    # convert back to standard typed `RootConfig` dataclass
    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    # 1) run harmonize to populate ETL outputs in EPSG:3161
    pipelines.harmonize(config)

    # 2) run the ingestion pipeline to build foundation inputs
    pipelines.ingest(config)

    # 3) run the preparation pipeline
    pipelines.prepare(config)

    # verify the generated transform outputs
    out_dpath = config.transform.output_dpath
    assert os.path.exists(os.path.join(out_dpath, 'block_splits_source.json'))
    assert os.path.exists(
        os.path.join(out_dpath, 'block_splits_transformed.json')
    )
    assert os.path.exists(os.path.join(out_dpath, 'image_stats.json'))
    assert os.path.exists(os.path.join(out_dpath, 'prep_report.json'))
    assert os.path.exists(os.path.join(out_dpath, 'schema.json'))
