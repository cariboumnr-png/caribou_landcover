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

'''Smoke test for pipeline=data-harmonize execution.'''

# standard imports
import json
import os
# local imports
import landseg.configs as configs
import landseg.execution.executor as executor


# ----- test cases
def test_execute_pipeline_data_harmonize(dummy_data_paths, tmp_path):
    '''
    Given: Pre-generated dummy data paths for raw Sentinel-2, DEM, and landcover in EPSG:32618.
    When: Calling `execute_pipeline` with pipeline.name='data-harmonize'.
    Then: Warps features and labels to EPSG:3161 at 20m and writes etl_report.json and VRT outputs.
    '''
    out_dpath = str(tmp_path / 'etl_out')

    root_cfg = configs.RootConfig()
    root_cfg.pipeline.name = 'data-harmonize'
    root_cfg.etl.target_crs = 'EPSG:3161'
    root_cfg.etl.target_resolution = 20.0
    root_cfg.etl.output_dpath = out_dpath
    root_cfg.etl.features = {
        'sentinel2': dummy_data_paths.raw_sentinel2,
        'dem': dummy_data_paths.raw_dem
    }
    root_cfg.etl.labels = {
        'landcover': dummy_data_paths.raw_landcover
    }

    report = executor.execute_pipeline(root_cfg)

    assert report['status'] == 'SUCCESS'
    assert report['target_crs'] == 'EPSG:3161'
    assert report['target_resolution'] == 20.0

    report_file = os.path.join(out_dpath, 'etl_report.json')
    assert os.path.exists(report_file)

    with open(report_file, 'r', encoding='utf-8') as f:
        saved_report = json.load(f)
    assert saved_report['status'] == 'SUCCESS'
    assert os.path.exists(os.path.join(out_dpath, 'harmonized_image_composite.vrt'))
    assert os.path.exists(os.path.join(out_dpath, 'valid_pixel_mask.vrt'))
