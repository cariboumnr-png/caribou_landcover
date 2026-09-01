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


class DatasetConfigItem(typing.TypedDict):
    '''Expected shape of dataset config (per raster).'''
    name: str
    path: str
    category: typing.Literal[
        'domains',
        'domain',
        'features',
        'feature',
        'labels',
        'label',
    ]
    band_mapping: dict[int, str] | None
    label_specs: geo_core.LabelSpecs | None


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
        # safe retrieve values from manifest with validation
        name, raster_p, cfg_p = _validate_manifest(i, mfst)

        # read individual config json
        ctrl = artifacts.Controller[DatasetConfigItem].load_json_or_fail(cfg_p)
        ctrl.hash(overwrite=False) # has once
        cfg = ctrl.fetch()

        # get category with checkes
        category = cfg.get('category')
        if not category or not isinstance(category, str):
            raise ValueError(
                f'Required value for [name] missing or of wrong type, '
                f'got: {category} ({type(category)} at dictionary index {i}'
            )

        # let band mapping pass through
        band_mapping = cfg.get('band_mapping')

        # resolve label specs
        label_specs = _resolve_label_specs(name, cfg.get('label_specs'))

        compiled.append({
            'name': name,
            'path': raster_p,
            'category': category,
            'band_mapping': band_mapping,
            'label_specs': label_specs
        })

    # return a dict indexed by file path
    return {c['path']: c for c in compiled}


def _validate_manifest(
    i: int,
    manifest: dict[str, typing.Any]
) -> tuple[str, str, str]:
    '''Validate per file manifest dict.'''
    if not isinstance(manifest, dict):
        raise ValueError(
            f'Manifest JSON expected to read as a list dictionaries, '
            f'got: {type(manifest)} at index {i}'
        )

    # safe retrieve values
    name = manifest.get('name')
    raster_p = manifest.get('path')
    config_p = manifest.get('config')

    # checks before proceeding
    if not name or not isinstance(name, str):
        raise ValueError(
            f'Required value for [name] missing or of wrong type, '
            f'got: {name} ({type(name)} at dictionary index {i}'
        )

    if not raster_p or not isinstance(raster_p, str):
        raise ValueError(
            f'Required value for [name] missing or of wrong type, '
            f'got: {raster_p} ({type(raster_p)} at dictionary index {i}'
        )
    if not os.path.exists(raster_p):
        raise ValueError(f'Source file at {raster_p} does not exsit')

    if not config_p or not isinstance(config_p, str):
        raise ValueError(
            f'Required value for [name] missing or of wrong type, '
            f'got: {config_p} ({type(config_p)} at dictionary index {i}'
        )
    if not os.path.exists(config_p):
        raise ValueError(f'Source file at {config_p} does not exsit')

    return name, raster_p, config_p


def _resolve_label_specs(
    name: str,
    label_specs: geo_core.LabelSpecs | None
) -> geo_core.LabelSpecs | None:
    '''Resolve label specs with validation.'''
    if label_specs is None:
        return None

    num_cls = label_specs.get('num_cls')
    if not isinstance(num_cls, int) or num_cls < 1:
        raise ValueError(
            f'Invalid "num_cls" in label_specs for "{name}": {num_cls}'
            f', note number of classes must be at least 1'
        )

    if not 'taxonomy' in label_specs:
        return label_specs

    if label_specs['taxonomy']:
        profile = label_specs['taxonomy'].get('profile')

        if not profile:
            raise ValueError(
                f'Profile not provided for "taxonomy" for label {name}'
            )

        if not ('class_name' in label_specs and label_specs['class_name']):
            raise ValueError(
                'Class names not provided for taxonomy lookup'
            )

        try:
            canonical_indices = taxonomy.validate_taxonomy_specs(
                profile,
                label_specs['class_name'],
                num_cls
            )
            label_specs['taxonomy']['canonical_indices'] = canonical_indices
        except ValueError as e:
            raise e

        return label_specs

    return label_specs
