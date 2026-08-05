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
Data ingestion pipeline.

Prepares the world grid, materializes domain knowledge, and builds
the immutable raw block catalogue for later experiments.
'''

# standard imports
import os
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe.ingest as ingest_data


def ingest(config: configs.RootConfig):
    '''
    Run the ingestion pipeline.

    Steps:
    1) Build or load the world grid.
    2) Prepare domain knowledge aligned to the grid.
    3) Build raw `.npz` data blocks, and update `catalog.json` and
    `schema.json`.

    Args:
        config: RootConfig with ingestion settings.
    '''
    # artifact paths
    artifact_paths = artifacts.ArtifactPaths.from_config(config)
    paths = artifact_paths.data_ingestion
    harmonize_paths = artifact_paths.data_harmonization

    # locate latest harmonization run folder if present
    try:
        harmonize_paths.get_run_folder()
    except FileNotFoundError:
        pass

    # init an IngestionLogger with summary
    logger = ingest_data.IngestionLogger(
        name='ingest',
        log_file=paths.report,
        enable_file_log=False
    )
    logger.init_summary(run_id='ingest')
    assert logger.summary # typing

    try:
        logger.log_sep()

        # resolve lifecycle policy dynamically
        policy = (
            artifacts.LifecyclePolicy.REBUILD
            if config.data.ingestion.rebuild
            else artifacts.LifecyclePolicy.BUILD_IF_MISSING
        )

        # config aliases
        domain_cfg = config.data.ingestion.domains
        grid_cfg = config.data.ingestion.grid
        datablocks_cfg = config.data.ingestion.datablocks

        # world grid
        logger.log('INFO', '[START] World grid preparation')
        grid_config = ingest_data.GridParameters(
            mode='ref',
            crs=grid_cfg.crs,
            ref_fpath=harmonize_paths.valid_mask_raster,
            origin=grid_cfg.extent.origin,
            pixel_size=grid_cfg.extent.pixel_size,
            grid_extent=grid_cfg.extent.grid_extent,
            grid_shape=grid_cfg.extent.grid_shape,
            tile_specs=grid_cfg.tile_specs_tuple,
        )
        world_grid = ingest_data.prepare_world_grid(
            paths.grids.fpath(grid_cfg.tile_specs_tuple),
            grid_config,
            policy=policy,
            logger=logger,
        )

        # log to console with duration
        assert logger.summary['world_grid'] # typing: should already populate
        d = logger.summary['world_grid']['duration_sec']
        logger.log('INFO', f'[COMPLETE] World grid preparation (D_{d:.2f}s)')

        # domain maps
        gid = world_grid.gid
        if os.path.exists(harmonize_paths.domain_raster):
            logger.log(
                'INFO',
                '[START] Domain maps preparation (canonical stacked domains)'
            )
            domain_config = [
                ingest_data.DomainBuildingParameters(
                    input_fpath=harmonize_paths.domain_raster,
                    domain_fpath=paths.domains.domain_map_fpath(
                        'stacked_domains'
                    ),
                    tiles_fpath=paths.domains.mapped_tiles_fpath(
                        'stacked_domains', gid
                    ),
                    index_base=1,
                    valid_threshold=domain_cfg.valid_threshold,
                    target_variance=domain_cfg.target_variance,
                )
            ]
            ingest_data.prepare_domain_maps(
                world_grid,
                domain_config,
                policy=policy,
                logger=logger,
            )
            d = sum(dm['duration_sec'] for dm in logger.summary['domain_maps'])
            logger.log(
                'INFO',
                f'[COMPLETE] Domain maps preparation (D_{d:.2f}s)'
            )
        else:
            logger.log('INFO', '[NOTE] No domain knowledge layers provided')

        # dataset config path from harmonization
        data_config_fpath = harmonize_paths.dataset_config

        # build dev data blocks
        logger.log('INFO', '[START] Development data blocks building')
        data_blocks_config = ingest_data.BlockBuildingParameters(
            stage='dev',
            image_fpath=harmonize_paths.feature_raster,
            label_fpath=harmonize_paths.label_raster,
            data_config_fpath=data_config_fpath,
            dem_pad=datablocks_cfg.image_dem_pad,
            ignore_index=datablocks_cfg.ignore_index,
        )
        ingest_data.run_blocks_building(
            world_grid,
            paths.data_blocks.dev,
            data_blocks_config,
            policy=policy,
            logger=logger,
        )

        # log to console with duration
        d = logger.summary['data_blocks']['dev']['duration_sec']
        logger.log(
            'INFO',
            f'[COMPLETE] Development data blocks preparation (D_{d:.2f}s)'
        )

        # build test data blocks - if provided by harmonization stage
        if not harmonize_paths.has_test_data:
            logger.log(
                'INFO',
                '[NOTE] Test holdout dataset not provided by harmonization'
            )
        else:
            logger.log('INFO', '[START] Test data blocks building')
            data_blocks_config = ingest_data.BlockBuildingParameters(
                stage='test',
                image_fpath=harmonize_paths.test_feature_raster,
                label_fpath=harmonize_paths.test_label_raster,
                data_config_fpath=data_config_fpath,
                dem_pad=datablocks_cfg.image_dem_pad,
                ignore_index=datablocks_cfg.ignore_index,
            )
            ingest_data.run_blocks_building(
                world_grid,
                paths.data_blocks.test,
                data_blocks_config,
                policy=policy,
                logger=logger,
            )
            assert logger.summary['data_blocks']['test']

            d = logger.summary['data_blocks']['test']['duration_sec']
            logger.log(
                'INFO',
                f'[COMPLETE] Test data blocks preparation (D_{d:.2f}s)'
            )

        artifacts.Controller[dict](paths.config).persist(config.as_dict)

    # propagate all exceptions here
    except Exception as e:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Ingestion pipeline failed: {e}', exc_info=True)
        raise e

    # close logger
    finally:
        logger.log_sep()
        logger.close()
