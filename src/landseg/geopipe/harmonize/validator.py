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


class DatasetConfigItem(typing.TypedDict):
    '''Expected shape of dataset config (per raster).'''
    name: str
    path: str
    category: str
    band_mapping: dict[int, str] | None
    label_specs: geo_core.LabelSpecs | None


def read_dataset_config(fp: str) -> dict[str, DatasetConfigItem]:
    '''Read dataset config JSON.'''
    # load JSON as artifact
    ctrl = artifacts.Controller[list[dict]].load_json_or_fail(fp)
    ctrl.hash(overwrite=False) # hash once
    dt_cfg = ctrl.fetch()
    assert dt_cfg # typing

    # expect JSON rea as a list of dicts
    if not isinstance(dt_cfg, list):
        raise ValueError(
            f'User-provided JSON expected to read as a list dictionaries, '
            f'got: {type(dt_cfg)}'
        )

    # got through each item with checks
    compiled: list[DatasetConfigItem] = []
    for i, cfg in enumerate(dt_cfg):
        if not isinstance(cfg, dict):
            raise ValueError(
                f'User-provided JSON expected to read as a list dictionaries, '
                f'got: {type(cfg)} at index {i}'
            )

        # safe retrieve values
        name = cfg.get('name')
        path = cfg.get('path')
        category = cfg.get('category')
        band_mapping = cfg.get('band_mapping')
        label_specs = cfg.get('label_specs')

        # checks before appending
        if not name or not isinstance(name, str):
            raise ValueError(
                f'Required value for [name] missing or of wrong type, '
                f'got: {name} ({type(name)} at dictionary index {i}'
            )

        if not path or not isinstance(path, str):
            raise ValueError(
                f'Required value for [name] missing or of wrong type, '
                f'got: {path} ({type(path)} at dictionary index {i}'
            )
        if not os.path.exists(path):
            raise ValueError(f'Source file at {path} does not exsit')

        if not category or not isinstance(category, str):
            raise ValueError(
                f'Required value for [name] missing or of wrong type, '
                f'got: {category} ({type(category)} at dictionary index {i}'
            )
        # NOTE here we skip checking band mapping and label specs for now

        compiled.append({
            'name': name,
            'path': path,
            'category': category,
            'band_mapping': band_mapping,
            'label_specs': label_specs
        })

    # return a dict indexed by file path
    return {c['path']: c for c in compiled}
