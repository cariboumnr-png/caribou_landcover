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


class CategoricalSpecs(typing.TypedDict):
    '''Typed dictionary for categorical raster specifications.'''
    # required
    index_base: int
    num_cls: int
    ignore_cls: list[int]
    # optional
    class_name: typing.NotRequired[dict[str, str]]
    color_map: typing.NotRequired[dict[str, list[int]]]
    taxonomy: typing.NotRequired[dict[str, typing.Any]]
    #
    reclass: typing.NotRequired[dict[str, list[int]]]
    reclass_name: typing.NotRequired[dict[str, str]]


class DatasetConfigItem(typing.TypedDict):
    '''Expected shape of dataset config (per raster).'''
    name: str
    path: str
    category: AllowedCategory
    band_mapping: dict[int, str]
    categorical_specs: CategoricalSpecs | None


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

        band_map = cfg.get('band_mapping')
        try:
            band_map = _resolve_band_mapping(band_map)
        except ValueError as e:
            raise ValueError(f'Invalid band mapping at index {i}') from e

        cat_specs = cfg.get('categorical_specs')
        try:
            cat_specs = _resolve_categorical_specs(cat_specs, category)
        except ValueError as e:
            raise ValueError(f'Invalid categorical specs at index {i}') from e

        compiled.append({
            'name': name,
            'path': raster_p,
            'category': category,
            'band_mapping': band_map,
            'categorical_specs': cat_specs,
        })

    # return a dict indexed by file path
    return {c['path']: c for c in compiled}


# ----- private helpers
def _resolve_manifest(mfst: typing.Any) -> tuple[str, str, DatasetConfigItem]:
    '''Resolve one data config entry.'''
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
            f'got: {name} ({type(name)})'
        )

    if not raster_p or not isinstance(raster_p, str):
        raise ValueError(
            f'Required value for [path] missing or of wrong type, '
            f'got: {raster_p} ({type(raster_p)})'
        )

    if not os.path.exists(raster_p):
        raise ValueError(f'Source file at {raster_p} does not exsit')

    if not config_p or not isinstance(config_p, str):
        raise ValueError(
            f'Required value for [config] missing or of wrong type, '
            f'got: {config_p} ({type(config_p)})'
        )

    if not os.path.exists(config_p):
        raise ValueError(f'Source file at {config_p} does not exsit')

    # read data config json via controller
    ctrl = artifacts.Controller[DatasetConfigItem].load_json_or_fail(config_p)
    ctrl.hash(overwrite=False) # hash once
    cfg = ctrl.fetch()

    return name, raster_p, cfg


def _resolve_category(cat: typing.Any) -> AllowedCategory:
    '''Validate and return `category` value from a config entry.'''
    if not cat or not isinstance(cat, str):
        raise ValueError(
            f'Required value for [category] missing or of wrong type, '
            f'got: {cat} ({type(cat)})'
        )

    allowed = typing.get_args(AllowedCategory)
    if cat not in allowed:
        raise ValueError(
            f'Invalid data category: {cat}, '
            f'must be one of {allowed}'
        )

    return typing.cast(AllowedCategory, cat)


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


def _resolve_categorical_specs(
    cat_specs: typing.Any,
    cat: AllowedCategory
) -> CategoricalSpecs | None:
    '''Resolve categorical specs with validation.'''
    is_categorical = cat in ['domain', 'domains', 'label', 'labels']
    is_label = cat in ['label', 'labels']

    if not is_categorical:
        return None

    if not isinstance(cat_specs, dict):
        raise ValueError(
            f'Categorical raster must have a "categorical_specs" dictionary, '
            f'got: {type(cat_specs)}'
        )

    index_base = cat_specs.get('index_base')
    if not (isinstance(index_base, int) and index_base >= 0):
        raise ValueError(
            'Categorical raster should have a non-negative "index_base", '
            f'got: {index_base} with type {type(index_base)}'
        )

    num_cls = cat_specs.get('num_cls')
    if not isinstance(num_cls, int) or num_cls < 1:
        raise ValueError(
            f'"num_cls" in categorical_specs: {num_cls} < 1'
        )

    ignore_cls = cat_specs.get('ignore_cls')
    if not isinstance(ignore_cls, list) or not all(
        isinstance(x, int) for x in ignore_cls
    ):
        raise ValueError(
            f'"ignore_cls" in categorical_specs must be list of ints, '
            f'got: {ignore_cls}'
        )

    resolved: CategoricalSpecs = {
        'index_base': index_base,
        'num_cls': num_cls,
        'ignore_cls': ignore_cls,
    }

    if is_label:
        for key in ('class_name', 'color_map', 'reclass', 'reclass_name'):
            val = cat_specs.get(key)
            if val is not None:
                if not isinstance(val, dict):
                    raise ValueError(
                        f'"{key}" in categorical_specs must be dict'
                    )
                resolved[key] = val

        tax_spec = cat_specs.get('taxonomy')
        if tax_spec:
            if not isinstance(tax_spec, dict):
                raise ValueError(
                    '"taxonomy" in categorical_specs must be dict'
                )
            profile = tax_spec.get('profile')
            if not profile:
                raise ValueError('Profile not provided for "taxonomy"')
            class_name = cat_specs.get('class_name')
            if not class_name:
                raise ValueError('Class names not provided for taxonomy lookup')

            canonical_indices = taxonomy.validate_taxonomy_specs(
                profile,
                class_name,
                num_cls
            )
            tax_dict = dict(tax_spec)
            tax_dict['canonical_indices'] = canonical_indices
            resolved['taxonomy'] = tax_dict

    return resolved
