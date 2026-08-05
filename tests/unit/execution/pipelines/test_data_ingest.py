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

# pylint: disable=missing-function-docstring

'''Unit tests for the data ingestion execution pipeline.'''

# standard imports
import os
import typing
# third-party imports
import omegaconf
# local imports
import landseg.configs as configs
import landseg.execution.pipelines as pipelines


# ----- pipeline execution
def test_data_ingest_pipeline_success(tmp_path, dummy_data_paths):
    # compose config with OmegaConf
    cfg_schema = omegaconf.OmegaConf.structured(configs.RootConfig)

    # override foundation fields
    grid_cfg = cfg_schema.foundation.grid
    grid_cfg.mode = 'ref'
    grid_cfg.crs = 'EPSG:3161'
    grid_cfg.tile_specs.size_row = 256
    grid_cfg.tile_specs.size_col = 256
    grid_cfg.tile_specs.overlap_row = 128
    grid_cfg.tile_specs.overlap_col = 128

    blocks_cfg = cfg_schema.foundation.datablocks
    blocks_cfg.name = 'test_ingest_run'

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

    # convert back to standard typed `RootConfig` dataclass
    config = typing.cast(
        configs.RootConfig,
        omegaconf.OmegaConf.to_object(cfg_schema)
    )

    # run harmonize first to generate matching CRS rasters
    pipelines.harmonize(config)

    # run the ingestion pipeline
    pipelines.ingest(config)

    # verify the generated outputs
    out_dpath = config.foundation.output_dpath
    assert os.path.exists(
        os.path.join(out_dpath, 'data_blocks', 'model_dev', 'catalog.json')
    )
    assert os.path.exists(
        os.path.join(out_dpath, 'data_blocks', 'test_holdout', 'catalog.json')
    )
    assert os.path.exists(
        os.path.join(out_dpath, 'ingest_report.json')
    )
