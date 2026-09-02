# =========================================================================== #
#            Copyright © His Majesty the King in right of Ontario,            #
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

'''
Data harmonization pipeline command implementation.
'''

# standard imports
import os
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.geopipe.core as geo_core
import landseg.geopipe.harmonize.taxonomy as taxonomy


# ----- public types
AllowedCategory = typing.Literal[
    'domains',
    'domain',
    'features',
    'feature',
    'labels',
    'label',
]


class DatasetConfigItem(typing.TypedDict):
    '''Expected shape of dataset config (per raster).'''
    name: str
    path: str
    category: AllowedCategory
    band_mapping: dict[int, str]
    index_base: int | None
    label_specs: geo_core.LabelSpecs | None


# ----- public functions
def compile_dataset_manifest(manifest_fp: str) -> dict[str, DatasetConfigItem]:
    '''Read and validate dataset manifest JSON.'''
    # load JSON via artifact controller
    ctrl = artifacts.Controller[list[dict]].load_json_or_fail(manifest_fp)
    ctrl.hash(overwrite=False) # hash once
    manifest = ctrl.fetch()

    # expect JSON read as a list of dicts
    if not isinstance(manifest, list):
        raise ValueError(
            f'Manifest JSON expected to read as a list dictionaries, '
            f'got: {type(manifest)}'
        )

    compiled: list[DatasetConfigItem] = []
    for i, mfst in enumerate(manifest):
        try:
            name, raster_p, cfg = _resolve_manifest(mfst)
        except ValueError as e:
            raise ValueError(f'Invalid data config entry at index {i}') from e

        category = cfg.get('category')
        try:
            category = _resolve_category(category)
        except ValueError as e:
            raise ValueError(f'Invalid category at index {i}') from e

        index_base = cfg.get('index_base')
        try:
            index_base = _resolve_index_base(index_base, category)
        except ValueError as e:
            raise ValueError(f'Invalid index base at index {i}') from e

        band_map = cfg.get('band_mapping')
        try:
            band_map = _resolve_band_mapping(band_map)
        except ValueError as e:
            raise ValueError(f'Invalid band mapping at index {i}') from e

        label_specs = cfg.get('label_specs')
        try:
            label_specs = _resolve_label_specs(label_specs)
        except ValueError as e:
            raise ValueError(f'Invalid label specs at index {i}') from e

        compiled.append({
            'name': name,
            'path': raster_p,
            'category': category,
            'band_mapping': band_map,
            'index_base': index_base,
            'label_specs': label_specs
        })

    # return a dict indexed by file path
    return {c['path']: c for c in compiled}


# ----- private helpers
def _resolve_manifest(mfst: typing.Any) -> tuple[str, str, DatasetConfigItem]:
    '''Resolve one data config entry'''
    if not isinstance(mfst, dict):
        raise ValueError(
            f'Manifest JSON expected to read as a list dictionaries, '
            f'got: {type(mfst)}'
        )

    name = mfst.get('name')
    raster_p = mfst.get('path')
    config_p = mfst.get('config')

    if not name or not isinstance(name, str):
        raise ValueError(
            f'Required value for [name] missing or of wrong type, '
            f'got: {name} ({type(name)}'
        )

    if not raster_p or not isinstance(raster_p, str):
        raise ValueError(
            f'Required value for [name] missing or of wrong type, '
            f'got: {raster_p} ({type(raster_p)}'
        )

    if not os.path.exists(raster_p):
        raise ValueError(f'Source file at {raster_p} does not exsit')


    if not config_p or not isinstance(config_p, str):
        raise ValueError(
            f'Required value for [name] missing or of wrong type, '
            f'got: {config_p} ({type(config_p)}'
        )

    if not os.path.exists(config_p):
        raise ValueError(f'Source file at {config_p} does not exsit')

    # read data config json via controller
    ctrl = artifacts.Controller[DatasetConfigItem].load_json_or_fail(config_p)
    ctrl.hash(overwrite=False) # has once
    cfg = ctrl.fetch()

    return name, raster_p, cfg


def _resolve_category(cat: typing.Any) -> AllowedCategory:
    '''Validate and return `category` value from a config entry.'''
    if not cat or not isinstance(cat, str):
        raise ValueError(
            f'Required value for [name] missing or of wrong type, '
            f'got: {cat} ({type(cat)}'
        )

    allowed = typing.get_args(AllowedCategory)
    if cat not in allowed:
        raise ValueError(
            f'Invalid data category: {cat}, '
            f'must be one of {allowed}'
        )

    return typing.cast(AllowedCategory, cat)


def _resolve_index_base(base: typing.Any, cat: AllowedCategory) -> int | None:
    '''Validate and return `index base` for categorical data source.'''
    is_categorical = cat in ['domain', 'domains', 'label', 'labels']
    if is_categorical:
        if not (isinstance(base, int) and base >= 0):
            raise ValueError(
                'Categorical raster should have a non-negative index base, '
                f'got: {base} with type {type(base)}'
            )
        return base
    return None # ignore non-categorical data sources


def _resolve_band_mapping(mapping: typing.Any) -> dict[int, str]:
    '''Validate and return `band_mapping` dict from a config entry.'''
    _mapping: dict[int, str] = {}
    if not isinstance(mapping, dict):
        raise ValueError(
            f'Band mapping must be a dictionary, got: {type(mapping)}'
        )

    for k, v in mapping.items():
        try:
            int(k)
        except ValueError as e:
            raise ValueError(
                f'Invalid band_mapping key: {k}, must be valid integer'
            ) from e

        if not isinstance(v, str):
            raise ValueError(
                f'Invalid band_mapping value {v} with type: {type(v)} '
                f'must be a string'
            )
        _mapping[int(k)] = v

    if set(_mapping.keys()) != set(range(1, len(_mapping) + 1)):
        raise ValueError(
            f'Band mapping dict keys should be contiguous integers from 1, '
            f'got: {sorted(_mapping.keys())}'
        )

    return _mapping


def _resolve_label_specs(lbl_specs: typing.Any) -> geo_core.LabelSpecs | None:
    '''Resolve label specs with validation.'''
    if lbl_specs is None:
        return None

    num_cls = lbl_specs.get('num_cls')
    if not isinstance(num_cls, int) or num_cls < 1:
        raise ValueError(f'"num_cls" in label_specs: {num_cls} < 1')

    if not 'taxonomy' in lbl_specs:
        return lbl_specs

    if lbl_specs['taxonomy']:
        profile = lbl_specs['taxonomy'].get('profile')

        if not profile:
            raise ValueError('Profile not provided for "taxonomy"')

        if not ('class_name' in lbl_specs and lbl_specs['class_name']):
            raise ValueError('Class names not provided for taxonomy lookup')

        try:
            canonical_indices = taxonomy.validate_taxonomy_specs(
                profile,
                lbl_specs['class_name'],
                num_cls
            )
            lbl_specs['taxonomy']['canonical_indices'] = canonical_indices
        except ValueError as e:
            raise e

        return lbl_specs

    return lbl_specs
