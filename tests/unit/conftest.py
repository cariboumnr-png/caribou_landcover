# =========================================================================== #
#           Copyright (c) His Majesty the King in right of Ontario,           #
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

'''Fixtures for testing `landseg.core` module.'''

# standard imports
import os
# third-party imports
import numpy
import pytest
# local imports
import landseg.core as core

@pytest.fixture
def dataspecs(tmp_path):
    # write temp block files
    blk_dict = {
        'image': numpy.random.rand(4, 256, 256), # as per 4 bands
        'label': numpy.random.randint(1, 4, size=(2, 256, 256)) # two heads
    }
    if not os.path.exists(f'{tmp_path}//train_block.npz'): # just check one
        numpy.savez(f'{tmp_path}//train_block.npz', **blk_dict)
        numpy.savez(f'{tmp_path}//val_block.npz', **blk_dict)
        numpy.savez(f'{tmp_path}//test_block.npz', **blk_dict)

    return core.DataSpecs(
        name="test_dataset",
        mode="default",
        meta=core.Meta(
            blk_bytes=1024, # dummy value, != acutal size from the npz files
            test_blks_grid=(1, 1),
            label_color_map=None,
            image_specs=core.Meta.Image(
                num_channels=4,
                height_width=256,
                array_key='image',
                band_map={'red': 0, 'green': 1, 'blue': 2, 'dem': 3}
            ),
            label_specs=core.Meta.Label(
                ignore_index=255,
                array_key='label'
            )
        ),
        heads=core.Heads(
            class_counts={
                'head_1': [100, 200],
                'head_2': [50, 150, 250],
            },
            logits_adjust={
                'head_1': [0.2, 0.1],
                'head_2': [0.1, 0.1, 0.1],
            },
            head_parent={'head_1': None, 'head_2': None},
            head_parent_cls={'head_1': None, 'head_2': None},
        ),
        splits=core.Splits(
            train={'train_block': f'{tmp_path}//train_block.npz'},
            val={'val_block': f'{tmp_path}//val_block.npz'},
            test={'test_block': f'{tmp_path}//test_block.npz'},
        ),
        domains=core.Domains(
        train=core.Domains.Dom(
                ids_domain={'train_block': 1},
                vec_domain={'train_block': [0.1, 0.2]}
            ),
            val=core.Domains.Dom(
                ids_domain={'val_block': 2},
                vec_domain={'val_block': [0.3, 0.4]}
            ),
            test=core.Domains.Dom(
                ids_domain={'test_block': 3},
                vec_domain={'test_block': [0.5, 0.6]}
            ),
            ids_num=3,
            vec_dim=2,
        ),
    )
