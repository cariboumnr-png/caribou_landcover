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
from .taxonomy import validate_taxonomy_specs


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
    '''Read dataset manifest JSON.'''
    # load JSON as artifact
    ctrl = artifacts.Controller[list[dict]].load_json_or_fail(manifest_fp)
    ctrl.hash(overwrite=False) # hash once
    manifest = ctrl.fetch()
    assert manifest # typing

    # expect JSON rea as a list of dicts
    if not isinstance(manifest, list):
        raise ValueError(
            f'Manifest JSON expected to read as a list dictionaries, '
            f'got: {type(manifest)}'
        )

    compiled: list[DatasetConfigItem] = []
    # got through each item with checks
    for i, mfst in enumerate(manifest):
        if not isinstance(mfst, dict):
            raise ValueError(
                f'Manifest JSON expected to read as a list dictionaries, '
                f'got: {type(mfst)} at index {i}'
            )

        # safe retrieve values
        name = mfst.get('name')
        raster_p = mfst.get('path')
        config_p = mfst.get('config')

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

        # read config json
        _ctrl = artifacts.Controller[DatasetConfigItem].load_json_or_fail(config_p)
        _ctrl.hash(overwrite=False) # has once
        cfg = _ctrl.fetch()
        assert cfg # typing

        category = cfg.get('category')
        band_mapping = cfg.get('band_mapping')
        label_specs = cfg.get('label_specs')

        if not category or not isinstance(category, str):
            raise ValueError(
                f'Required value for [name] missing or of wrong type, '
                f'got: {category} ({type(category)} at dictionary index {i}'
            )

        # validate label specs and taxonomy if present
        if label_specs is not None:
            num_cls = label_specs.get('num_cls')
            if not isinstance(num_cls, int) or num_cls < 1:
                raise ValueError(
                    f"Invalid 'num_cls' in label_specs for '{name}': {num_cls}"
                )
            if 'taxonomy' in label_specs and label_specs['taxonomy']:
                try:
                    resolved_tax, inferred_names, _ = validate_taxonomy_specs(
                        label_specs['taxonomy'],
                        num_cls=num_cls,
                    )
                except ValueError as e:
                    raise ValueError(
                        f"Taxonomy validation failed for label layer '{name}' "
                        f"in '{config_p}': {e}"
                    ) from e

                label_specs['taxonomy'] = resolved_tax
                if 'class_name' not in label_specs or not label_specs['class_name']:
                    label_specs['class_name'] = inferred_names

        compiled.append({
            'name': name,
            'path': raster_p,
            'category': category,
            'band_mapping': band_mapping,
            'label_specs': label_specs
        })

    # return a dict indexed by file path
    return {c['path']: c for c in compiled}
