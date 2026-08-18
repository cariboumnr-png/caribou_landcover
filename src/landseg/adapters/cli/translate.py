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
User settings translation helper.
'''

# standard imports
import typing
# third-party imports
import omegaconf

def translate_user_config(raw: omegaconf.DictConfig) -> omegaconf.DictConfig:
    '''Translate user.yaml DictConfig into RootConfig overrides.'''

    translated: dict[str, typing.Any] = {
        'execution': {},
        'data':{
            'harmonization': {
                'canvas': {},
                'grid': {
                    'extent': {},
                    'tile_specs': {},
                },
            },
            'ingestion': {
                'domains': {},
                'datablocks': {},
            },

            'preparation': {
                'catalog': {},
                'partition': {},
                'scoring': {},
            },
            'specification': {},
        },
        'models': {},
        'session': {
            'data_loader': {},
            'orchestration': {
                'curriculum': {
                    'single': {
                        'phases': [{}],
                    },
                },
            },
        },
    }

    if 'exp_root' in raw:
        _set_paths(translated, ['execution.exp_root'], raw['exp_root'])

    if 'data-harmonize' in raw:
        _translate_data_harmonize(raw['data-harmonize'], translated)

    if 'data-ingest' in raw:
        _translate_data_ingest(raw['data-ingest'], translated)

    if 'data-prepare' in raw:
        _translate_data_prepare(raw['data-prepare'], translated)

    if 'model-train' in raw:
        _translate_model_train(raw['model-train'], translated)

    return omegaconf.OmegaConf.create(translated)


def _translate_data_harmonize(
    harmonization: omegaconf.DictConfig,
    translated: dict
) -> None:
    '''Map data-harmonize settings to harmonization fields.'''

    mapping = {
        'target_crs': ['data.harmonization.canvas.target_crs'],
        'target_resolution': ['data.harmonization.canvas.target_resolution'],
        'reference_raster': ['data.harmonization.canvas.reference_raster'],
        'grid_mode': ['data.harmonization.grid.mode'],
        'grid_crs': ['data.harmonization.grid.crs'],
        'tile_size': [
            'data.harmonization.grid.tile_specs.size_row',
            'data.harmonization.grid.tile_specs.size_col'
        ],
        'tile_overlap': [
            'data.harmonization.grid.tile_specs.overlap_row',
            'data.harmonization.grid.tile_specs.overlap_col'
        ],
        'resampling_continuous': ['data.harmonization.resampling_continuous'],
        'resampling_categorical': ['data.harmonization.resampling_categorical'],
        'dataset_manifest': ['data.harmonization.dataset_manifest'],
        'dataset_name': [
            'data.harmonization.dataset_name',
            'data.ingestion.datablocks.name'
        ],
        'output_dpath': ['data.harmonization.output_dpath'],
    }
    _apply_mapping(harmonization, translated, mapping)


def _translate_data_ingest(
    ingestion: omegaconf.DictConfig,
    translated: dict
) -> None:
    '''Map data-ingest settings to ingestion fields.'''

    mapping = {
        'harmonization_run': ['data.ingestion.harmonization_run'],
        'domain_ids_name': ['dataspecs.domain_ids_name'],
        'domain_vec_name': ['dataspecs.domain_vec_name'],
        'dataset_name': ['data.ingestion.datablocks.name'],
        'rebuild': ['data.ingestion.rebuild'],
        'output_dpath': ['data.ingestion.output_dpath'],
    }
    _apply_mapping(ingestion, translated, mapping)



def _translate_data_prepare(
    preparation: omegaconf.DictConfig,
    translated: dict
) -> None:
    '''Map data-prepare settings to preparation fields.'''

    mapping = {
        'val_ratio': ['data.preparation.partition.val_ratio'],
        'test_ratio': ['data.preparation.partition.test_ratio'],
        'target_head': ['data.preparation.catalog.focal_target'],
        'reward_classes': ['data.preparation.scoring.reward'],
        'rebuild': ['data.preparation.rebuild'],
        'output_dpath': ['data.preparation.output_dpath'],
    }
    _apply_mapping(preparation, translated, mapping)


def _translate_model_train(
    model_train: omegaconf.DictConfig,
    translated: dict
) -> None:
    '''Map model-train settings to models and session fields.'''

    mapping = {
        'exp_root': ['execution.exp_root'],
        'model_body': ['models.model_body'],
        'bottleneck': ['models.bottleneck'],
        'conditioners': ['models.conditioners'],
        'patch_size': ['session.data_loader.patch_size'],
        'batch_size': ['session.data_loader.batch_size'],
        'head_loss_weights': ['session.engine_tasks.head_weights'],
        'head_metrics_weights': ['session.orchestration.monitor.track_heads'],
        'output_dpath': ['session.output_dpath'],
    }
    _apply_mapping(model_train, translated, mapping)

    # infer active heads for phase from active_tasks or head weights
    active_tasks = model_train.get('active_tasks', None)
    loss_heads = model_train.get('head_loss_weights', {})
    metrics_heads = model_train.get('head_metrics_weights', {})
    if loss_heads and metrics_heads and set(loss_heads) != set(metrics_heads):
        raise ValueError('Different heads between loss and metrics weights')

    if active_tasks is not None:
        active_heads = list(active_tasks)
    elif loss_heads:
        active_heads = list(loss_heads.keys())
    else:
        active_heads = None

    phase = {
        'name': 'demo-train',
        'num_epochs': model_train['epochs'],
        'active_heads': active_heads
    }
    _set_paths(
        translated,
        ['session.orchestration.curriculum.single.phases'],
        [phase]
    )

def _apply_mapping(
    src: omegaconf.DictConfig,
    translated: dict,
    mapping: dict[str, list[str]]
) -> None:
    '''Apply mapping from source DictConfig to translated dictionary.'''
    for src_key, dest_paths in mapping.items():
        if src_key in src:
            _set_paths(translated, dest_paths, src[src_key])

def _set_paths(
    translated: dict,
    paths: list[str],
    val: typing.Any
) -> None:
    '''Set value at multiple target paths.'''

    def _set_path(
        d: dict,
        path: str,
        val: typing.Any
    ) -> None:
        parts = path.split('.')
        curr = d
        for part in parts[:-1]:
            if part not in curr:
                curr[part] = {}
            curr = curr[part]
        curr[parts[-1]] = val

    for path in paths:
        _set_path(translated, path, val)
