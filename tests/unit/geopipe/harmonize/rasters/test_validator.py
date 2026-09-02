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

'''Unit tests for dataset manifest validation (validator.py).'''

# standard imports
import json
# third-party imports
import pytest
# local imports
import landseg.geopipe.harmonize.manifest as validator



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
        'categorical_specs': None,
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
    assert compiled[str(raster_file)]['categorical_specs'] is None


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


def test_compile_dataset_manifest_categorical_specs(tmp_path):
    '''
    Given: Manifest with categorical domain having index_base=0.
    When: `compile_dataset_manifest` is executed.
    Then: Correctly parse and populate categorical_specs in compiled item.
    '''
    raster_file = tmp_path / 'domain.tif'
    raster_file.write_text('dummy raster')
    cfg_file = tmp_path / 'domain.json'
    cfg_data = {
        'category': 'domains',
        'band_mapping': {1: 'soil'},
        'categorical_specs': {
            'index_base': 0,
            'num_cls': 5,
            'ignore_cls': [],
        },
    }
    cfg_file.write_text(json.dumps(cfg_data))

    manifest_file = tmp_path / 'manifest.json'
    manifest_data = [
        {
            'name': 'soil_domain',
            'path': str(raster_file),
            'config': str(cfg_file),
        }
    ]
    manifest_file.write_text(json.dumps(manifest_data))

    compiled = validator.compile_dataset_manifest(str(manifest_file))
    assert str(raster_file) in compiled
    cat_specs = compiled[str(raster_file)]['categorical_specs']
    assert cat_specs is not None
    assert cat_specs['index_base'] == 0
    assert cat_specs['num_cls'] == 5
    assert cat_specs['ignore_cls'] == []


def test_compile_dataset_manifest_label_specs(tmp_path):
    '''
    Given: Manifest with categorical label having complete specs.
    When: `compile_dataset_manifest` is executed.
    Then: Correctly parse and populate categorical_specs in compiled item.
    '''
    raster_file = tmp_path / 'landcover.tif'
    raster_file.write_text('dummy raster')
    cfg_file = tmp_path / 'landcover.json'
    cfg_data = {
        'category': 'labels',
        'band_mapping': {1: 'landcover'},
        'categorical_specs': {
            'index_base': 1,
            'num_cls': 2,
            'ignore_cls': [255],
            'class_name': {'1': 'coniferous', '2': 'deciduous'},
            'color_map': {'1': [34, 139, 34], '2': [218, 165, 32]},
        },
    }
    cfg_file.write_text(json.dumps(cfg_data))

    manifest_file = tmp_path / 'manifest.json'
    manifest_data = [
        {
            'name': 'landcover',
            'path': str(raster_file),
            'config': str(cfg_file),
        }
    ]
    manifest_file.write_text(json.dumps(manifest_data))

    compiled = validator.compile_dataset_manifest(str(manifest_file))
    cat_specs = compiled[str(raster_file)]['categorical_specs']
    assert cat_specs is not None
    assert cat_specs['index_base'] == 1
    assert cat_specs['num_cls'] == 2
    assert cat_specs['ignore_cls'] == [255]
    assert cat_specs.get('class_name') == {
        '1': 'coniferous',
        '2': 'deciduous'
    }


def test_resolve_categorical_specs():
    '''
    Given: Various categorical_specs values and categories.
    When: Running `_resolve_categorical_specs`.
    Then: Correctly validate categorical specifications.
    '''
    # categorical domain valid
    dom_res = validator._resolve_categorical_specs(
        {'index_base': 0, 'num_cls': 5, 'ignore_cls': []}, 'domain'
    )
    assert dom_res == {
        'index_base': 0,
        'num_cls': 5,
        'ignore_cls': [],
    }

    # categorical labels valid
    lbl_res = validator._resolve_categorical_specs(
        {
            'index_base': 1,
            'num_cls': 2,
            'ignore_cls': [255],
        },
        'labels'
    )
    assert lbl_res is not None
    assert lbl_res['index_base'] == 1
    assert lbl_res['num_cls'] == 2
    assert lbl_res['ignore_cls'] == [255]

    # non-categorical returns None regardless
    assert validator._resolve_categorical_specs(None, 'features') is None
    assert validator._resolve_categorical_specs(
        {'index_base': 1, 'num_cls': 2, 'ignore_cls': []}, 'features'
    ) is None

    # categorical invalid
    with pytest.raises(ValueError, match='categorical_specs'):
        validator._resolve_categorical_specs(None, 'domain')

    with pytest.raises(ValueError, match='non-negative "index_base"'):
        validator._resolve_categorical_specs(
            {'index_base': -1, 'num_cls': 1, 'ignore_cls': []}, 'domain'
        )

    with pytest.raises(ValueError, match='num_cls'):
        validator._resolve_categorical_specs(
            {'index_base': 1, 'num_cls': 0, 'ignore_cls': []}, 'labels'
        )

    with pytest.raises(ValueError, match='ignore_cls'):
        validator._resolve_categorical_specs(
            {'index_base': 1, 'num_cls': 2, 'ignore_cls': 'invalid'}, 'domain'
        )


def test_resolve_schemes_features():
    '''
    Given: Feature schemes with valid and invalid band names.
    When: Running `_resolve_schemes` on feature category.
    Then: Correctly validate band names against band_mapping.
    '''
    band_map = {1: 'blue', 2: 'green', 3: 'red', 4: 'nir'}
    schemes = {
        'rgb': ['blue', 'green', 'red'],
        'rgb_nir': ['blue', 'green', 'red', 'nir'],
    }
    resolved = validator._resolve_schemes(schemes, 'features', band_map, None)
    assert resolved == schemes

    # invalid band name
    with pytest.raises(ValueError, match='not in band_mapping'):
        validator._resolve_schemes(
            {'bad': ['blue', 'swir1']}, 'features', band_map, None
        )

    # empty bands list
    with pytest.raises(ValueError, match='non-empty list'):
        validator._resolve_schemes({'empty': []}, 'features', band_map, None)


def test_resolve_schemes_labels():
    '''
    Given: Label schemes with valid and invalid class IDs.
    When: Running `_resolve_schemes` on label category.
    Then: Correctly validate class IDs against index_base and num_cls.
    '''
    cat_specs: validator.CategoricalSpecs = {
        'index_base': 1,
        'num_cls': 3,
        'ignore_cls': [255],
    }
    schemes = {
        'binary': {
            'reclass': {'1': [1], '2': [2, 3]},
            'reclass_name': {'1': 'WAT', '2': 'VEG'},
        }
    }
    resolved = validator._resolve_schemes(
        schemes, 'labels', {1: 'landcover'}, cat_specs
    )
    assert resolved == schemes

    # class ID out of range
    with pytest.raises(ValueError, match='outside valid class range'):
        validator._resolve_schemes(
            {
                'invalid': {
                    'reclass': {'1': [4]},
                    'reclass_name': {'1': 'OUT'},
                }
            },
            'labels',
            {1: 'landcover'},
            cat_specs,
        )


def test_resolve_schemes_domains():
    '''
    Given: Domain category with schemes defined.
    When: Running `_resolve_schemes`.
    Then: Raise ValueError as domains should not have schemes.
    '''
    assert validator._resolve_schemes(None, 'domain', {1: 'soil'}, None) is None
    with pytest.raises(ValueError, match='should not define "schemes"'):
        validator._resolve_schemes(
            {'dummy': ['soil']}, 'domain', {1: 'soil'}, None
        )
