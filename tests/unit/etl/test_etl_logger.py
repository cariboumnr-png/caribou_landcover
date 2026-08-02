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

'''Unit tests for ETL `HarmonizationLogger` structured logger and report persistence.'''

# standard imports
import json
import os
# local imports
import landseg.etl.logger as etl_logger


# ----- test cases
def test_harmonization_logger_summary_lifecycle(tmp_path):
    '''
    Given: A HarmonizationLogger initialized with a target CRS and output path.
    When: Recording grid shape, source outputs, composite path, and closing.
    Then: Persists structured etl_report.json to output_dpath.
    '''
    out_dpath = str(tmp_path / 'etl_out')
    os.makedirs(out_dpath, exist_ok=True)
    report_file = os.path.join(out_dpath, 'etl_report.json')

    logger = etl_logger.HarmonizationLogger(
        name='test_harmonize',
        log_file=report_file,
        enable_file_log=False
    )
    logger.init_summary(
        target_crs='EPSG:3161',
        target_resolution=20.0,
        output_dpath=out_dpath
    )
    logger.set_grid_shape(500, 500)
    logger.add_harmonized_source('sentinel2', '/path/to/s2.tif')
    logger.set_composite_raster('/path/to/comp.tif')
    logger.set_valid_mask_raster('/path/to/mask.tif')
    logger.set_summary_status('SUCCESS')

    logger.close()

    assert os.path.exists(report_file)
    with open(report_file, 'r', encoding='utf-8') as f:
        saved_report = json.load(f)

    assert saved_report['status'] == 'SUCCESS'
    assert saved_report['target_crs'] == 'EPSG:3161'
    assert saved_report['target_resolution'] == 20.0
    assert saved_report['grid_shape'] == [500, 500]
    assert saved_report['harmonized_sources']['sentinel2'] == '/path/to/s2.tif'
    assert saved_report['composite_raster'] == '/path/to/comp.tif'
    assert saved_report['valid_mask_raster'] == '/path/to/mask.tif'


def test_harmonization_logger_add_provenance(tmp_path):
    '''
    Given: A source file on disk and an initialized HarmonizationLogger.
    When: Calling `add_source_provenance`.
    Then: Records file size_bytes, mtime, and absolute path in report summary.
    '''
    out_dpath = str(tmp_path / 'prov_out')
    os.makedirs(out_dpath, exist_ok=True)
    sample_file = tmp_path / 'sample_raw.tif'
    sample_file.write_bytes(b'dummy_content_bytes')

    report_file = os.path.join(out_dpath, 'etl_report.json')
    logger = etl_logger.HarmonizationLogger(
        name='test_provenance',
        log_file=report_file,
        enable_file_log=False
    )
    logger.init_summary(
        target_crs='EPSG:3161',
        target_resolution=20.0,
        output_dpath=out_dpath
    )

    logger.add_source_provenance('sentinel2', str(sample_file))
    logger.close()

    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)

    prov = report['provenance']['sentinel2']
    assert prov['size_bytes'] == len(b'dummy_content_bytes')
    assert 'mtime' in prov
    assert prov['path'] == os.path.abspath(str(sample_file))
