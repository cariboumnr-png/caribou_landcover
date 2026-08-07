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
Overfit test pipeline.

Constructs a single valid block, builds minimal data specifications,
and trains until near-perfect IoU to validate the end-to-end stack.
'''

# standard imports
import os
import typing
# local imports
import landseg._constants as c
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.core as core
import landseg.geopipe.core as geo_core
import landseg.geopipe.ingest.data_blocks.assembler as assembler
import landseg.geopipe.ingest.data_blocks.mapper as mapper
import landseg.geopipe.ingest.world_grids as world_grids
import landseg.geopipe.utils as geo_utils
import landseg.models as models
import landseg.session as session
import landseg.session.engine as engine
import landseg.session.factory as session_factory

# aliases
DatasetConfigCtrl = artifacts.Controller[dict[str, typing.Any]]


# ----- overfit test pipeline
def overfit(config: configs.RootConfig) -> None:
    '''
    Run an overfit test on a single block.

    Creates or loads a block, builds a small `DataSpecs`, instantiates
    the model and a trainer with minimal logging, and trains until an
    IoU threshold or the epoch limit is reached.

    Args:
        config: RootConfig with model/trainer settings.
    '''
    root = f'{config.execution.exp_root}/results/overfit_test'
    log_file = f'{root}/log/overfit_summary.json'
    logger = session.SessionLogger('overfit', log_file=log_file)
    logger.init_summary(run_id='overfit_test', pipeline='diagnose_overfit')

    try:
        logger.log_sep()
        logger.set_inputs({
            'exp_root': config.execution.exp_root,
            'patch_size': config.session.data_loader.patch_size,
            'lr': config.session.engine_optim.lr,
            'max_epoch': c.OVERFIT_MAX_EPOCH,
        })

        dataspecs = _prepare_dataspecs(root, config, logger)

        model = models.build_multihead_unet(
            patch_size=config.session.data_loader.patch_size,
            dataspecs=dataspecs,
            unet_backbone_config=config.models.unet_backbone_config,
            conditioning_config=config.models.conditioning_config,
            enable_clamp=config.models.numeric_safety.enable_clamp,
            clamp_range=config.models.numeric_safety.clamp_range,
        )

        runner = session_factory.build_overfit_session(
            dataspecs=dataspecs,
            model=model,
            config=config.session,
            context=session.SessionBuildContext(device=c.DEVICE),
            logger=logger,
        )

        monitor_head = config.session.orchestration.monitor.track_heads
        active_heads = (
            list(monitor_head.keys())
            if monitor_head
            else list(dataspecs.heads.class_counts.keys())
        )
        runner.set_head_state(active_heads=active_heads)

        results = _run_overfit_loop(runner, config, logger)
        status = 'SUCCESS' if results['overfit_reached'] else 'FAILED'

        logger.set_results(results)
        logger.set_summary_status(status)

    except Exception as e:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Overfit pipeline failed: {e}', exc_info=True)
        raise e

    finally:
        logger.log_sep()
        logger.close()


# ----- dataspecs builder
def _prepare_dataspecs(
    save_dpath: str,
    config: configs.RootConfig,
    logger: session.SessionLogger,
) -> core.DataSpecs:
    '''Build or select a single test block and construct `DataSpecs`.'''
    block_fpath: str | None = None
    if os.path.exists(save_dpath):
        for f in os.listdir(save_dpath):
            if f.endswith('.npz'):
                block_fpath = os.path.join(save_dpath, f)
                logger.log('INFO', f'Using existing block: {block_fpath}')
                break

    if not block_fpath:
        block_fpath = _create_block(save_dpath, config, logger)

    block = geo_core.DataBlock.load(block_fpath)
    counts = block.manifest['label_count']
    cc = {k: [1] * len(counts[k]) for k in counts if k != 'original'}

    return core.DataSpecs(
        name='overfit_single_block',
        mode='default',
        meta=core.Meta(
            blk_bytes=0,
            test_blks_grid=(0, 0),
            label_color_map=None,
            image_specs=core.Meta.Image(
                num_channels=block.data.image.shape[0],
                height_width=block.data.image.shape[1],  # assume H == W
                array_key='image',
                band_map=block.manifest['image_band_map'],
            ),
            label_specs=core.Meta.Label(
                array_key='label_stack',
                ignore_index=block.manifest['ignore_index'],
            ),
        ),
        heads=core.Heads(
            class_counts=cc,  # neutral
            logits_adjust={k: [1.0] * len(v) for k, v in cc.items()},  # neutral
            head_parent=block.manifest['label_parent'],
            head_parent_cls=block.manifest['label_parent_cls'],
        ),
        splits=core.Splits(
            train={block.manifest['block_name']: block_fpath},
            val={block.manifest['block_name']: block_fpath},
            test={},
        ),
        domains=core.Domains(
            train={'ids_domain': None, 'vec_domain': None},
            val={'ids_domain': None, 'vec_domain': None},
            test={'ids_domain': None, 'vec_domain': None},
            ids_num=0,
            vec_dim=0,
        ),
    )


# ----- block construction helper
def _create_block(
    save_dpath: str,
    config: configs.RootConfig,
    logger: session.SessionLogger,
) -> str:
    '''Build one valid block for the overfit test.'''
    # construct world grid layout
    logger.log('INFO', 'Preparing world grid')
    harmonization_paths = artifacts.ArtifactPaths.from_config(config).data_harmonization
    grid_cfg = config.data.ingestion.grid
    world_grid = world_grids.build_grid(
        world_grids.GridParameters(
            mode='ref',
            crs=grid_cfg.crs,
            ref_fpath=harmonization_paths.valid_mask_raster,
            origin=grid_cfg.extent.origin,
            pixel_size=grid_cfg.extent.pixel_size,
            grid_extent=grid_cfg.extent.grid_extent,
            grid_shape=grid_cfg.extent.grid_shape,
            tile_specs=grid_cfg.tile_specs_tuple,
        )
    )

    # map raster windows onto world grid
    logger.log('INFO', 'Mapping image unto the world grid')
    harmonization_paths = artifacts.ArtifactPaths.from_config(config).data_harmonization
    datablocks_cfg = config.data.ingestion.datablocks
    mapped = mapper.map_rasters(
        world_grid,
        harmonization_paths.feature_raster,
        harmonization_paths.label_raster,
    )

    # load dataset config JSON
    logger.log('INFO', 'Building a single data block')
    ctrl = DatasetConfigCtrl.load_json_or_fail(harmonization_paths.dataset_config)
    ctrl.hash(overwrite=False)
    dataset_config = ctrl.fetch()
    assert dataset_config

    # construct `RasterReadInput` mapping for mapped windows
    inputs_map = {
        geo_utils.xy_name(coord): assembler.RasterReadInput(
            image_fpath=harmonization_paths.feature_raster,
            image_window=mapped.image[coord],
            image_band_map=dataset_config['image_band_map'],
            image_dem_pad_px=datablocks_cfg.image_dem_pad,
            label_fpath=harmonization_paths.label_raster,
            label_window=mapped.label[coord] if mapped.label else None,
            label_specs=dataset_config.get('label_specs'),
        )
        for coord in mapped.image
    }

    # resolve target head for filtering
    target_head = _resolve_target_head(config, dataset_config)

    # build single valid test block matching criteria
    logger.log('DEBUG', 'Try: valid_px_per=0.95; need_all_class=True')
    block_fpath = assembler.build_test_block(
        save_dpath=save_dpath,
        inputs=inputs_map,
        target_head=target_head,
        valid_px_per=0.95,
        need_all_classes=True,
    )
    # second try
    if not block_fpath:
        logger.log('DEBUG', 'Try: valid_px_per=0.95; need_all_class=False')
        block_fpath = assembler.build_test_block(
            save_dpath=save_dpath,
            inputs=inputs_map,
            target_head=target_head,
            valid_px_per=0.95,
            need_all_classes=False,
        )
    # fail
    if not block_fpath:
        raise ValueError('No valid block for testing is found')

    logger.log('INFO', f'Single block successfully created: {block_fpath}')
    return block_fpath


# ----- target head resolution helper
def _resolve_target_head(
    config: configs.RootConfig,
    dataset_config: dict[str, typing.Any],
) -> str:
    '''Resolve the target head for test block filtering.'''
    label_specs = dataset_config.get('label_specs')
    if not label_specs:
        raise ValueError('No label specifications found in dataset config')

    first_head = list(label_specs.keys())[0]
    active_heads = config.session.orchestration.single_phase.active_heads

    return active_heads[0] if active_heads else first_head


# ----- overfit epoch training loop helper
def _run_overfit_loop(
    runner: engine.EpochEngine,
    config: configs.RootConfig,
    logger: session.SessionLogger,
) -> dict[str, typing.Any]:
    '''Execute epoch training loop until threshold or max epochs.'''
    max_epoch = c.OVERFIT_MAX_EPOCH
    lr = config.session.engine_optim.lr
    logger.log('INFO', 'Starting overfit test')
    logger.log('INFO', f'Maximum epoch: {max_epoch}')
    logger.log('INFO', f'Learning rate: {lr}')

    loss, iou = 0.0, 0.0
    for ep in range(1, max_epoch + 1):
        results = runner.run_epoch(ep)
        assert results.training and results.validation  # typing
        results.track(
            config.session.orchestration.monitor.metric_name,
            config.session.orchestration.monitor.track_heads,
        )
        loss = results.training.total_objective
        iou = results.target_metrics
        logger.log(
            'INFO',
            f'Epoch: {ep:04d} | Loss: {loss:.4f} | IoU: {iou:.4f}'
        )
        if iou >= 0.99:
            logger.log('INFO', 'Overfit reached - test complete')
            return {
                'final_epoch': ep,
                'final_loss': loss,
                'final_iou': iou,
                'overfit_reached': True,
            }

    logger.log('WARNING', f'IoU did not reach 99% after {max_epoch} epochs.')
    return {
        'final_epoch': max_epoch,
        'final_loss': loss,
        'final_iou': iou,
        'overfit_reached': False,
    }
