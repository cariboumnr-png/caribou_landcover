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

'''Unit tests for dataset manifest validation (validator.py).'''

# standard imports
import json
# third-party imports
import pytest
# local imports
import landseg.geopipe.harmonize.validator as validator


# ----- `compile_dataset_manifest` tests
def test_compile_dataset_manifest_success(tmp_path):
    '''
    Given: A valid manifest JSON and associated raster config files.
    When: `compile_dataset_manifest` is executed.
    Then: Return a dictionary indexed by raster file path.
    '''
    raster_file = tmp_path / 'sample.tif'
    raster_file.write_text('dummy raster')
    cfg_file = tmp_path / 'sample.json'
    cfg_data = {
        'category': 'features',
        'band_mapping': {1: 'red'},
        'label_specs': None,
    }
    cfg_file.write_text(json.dumps(cfg_data))

    manifest_file = tmp_path / 'manifest.json'
    manifest_data = [
        {
            'name': 's2_sample',
            'path': str(raster_file),
            'config': str(cfg_file),
        }
    ]
    manifest_file.write_text(json.dumps(manifest_data))

    compiled = validator.compile_dataset_manifest(str(manifest_file))
    assert str(raster_file) in compiled
    assert compiled[str(raster_file)]['name'] == 's2_sample'
    assert compiled[str(raster_file)]['category'] == 'features'


def test_compile_dataset_manifest_invalid_json(tmp_path):
    '''
    Given: A manifest JSON that is a dictionary instead of a list.
    When: `compile_dataset_manifest` is executed.
    Then: Raise a ValueError.
    '''
    manifest_file = tmp_path / 'manifest.json'
    manifest_file.write_text(json.dumps({'invalid': 'shape'}))

    with pytest.raises(ValueError, match='expected to read as a list'):
        validator.compile_dataset_manifest(str(manifest_file))
