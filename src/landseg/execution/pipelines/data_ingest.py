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

# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe.grid as grid
import landseg.geopipe.ingest as ingest

# aliases
ConfigController = artifacts.Controller[dict]


def exec_ingest_data(config: configs.RootConfig) -> None:
    '''
    Run the ingestion pipeline.

    Steps:
    1) Load the canonical world grid from harmonization.
    2) Prepare domain knowledge aligned to the grid.
    3) Build raw canonical `.npz` data blocks, and update `catalog.json` and
       `schema.json`.

    Args:
        config: RootConfig with ingestion settings.
    '''
    # artifact paths
    artifact_paths = artifacts.ArtifactPaths.from_config(config)
    ingestion_paths = artifact_paths.data_ingestion

    # read harmonization report
    harmonized = ingest.read_harmonization_report(
        artifact_paths.data_harmonization,
        config.data.ingestion.harmonization_run
    )

    # init an IngestionLogger with summary
    logger = ingest.IngestionLogger(
        name='ingest',
        log_file=ingestion_paths.report,
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

        # ----- load canonical world grid
        logger.log('INFO', '[START] Loading world grid from configuration')
        world_grid = grid.load_grid_from_fpath(harmonized.grid_fpath)
        gid = world_grid.gid
        logger.log('INFO', f'[COMPLETE] World grid loaded: {gid}')

        # ----- materialize domain maps
        domain_cfg = config.data.ingestion.domains
        if harmonized.domains:
            logger.log('INFO', '[START] Domain maps preparation')
            domain_configs = [
                ingest.DomainBuildingParameters(
                    input_fpath=path,
                    domain_fpath=ingestion_paths.domains.domain_map_fpath(name),
                    tiles_fpath=ingestion_paths.domains.mapped_tiles_fpath(name, gid),
                    index_base=1,
                    valid_threshold=domain_cfg.valid_threshold,
                    target_variance=domain_cfg.target_variance,
                ) for name, path in harmonized.domains.items()
            ]
            ingest.prepare_domain_maps(
                world_grid,
                domain_configs,
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

        # ----- build canonical data blocks if provided
        if not harmonized.has_data:
            logger.log('INFO', 'Harmonized feature/label rasters not provided')
        else:
            logger.log('INFO', '[START] Canonical data blocks building')
            assert harmonized.features
            data_blocks_config = ingest.BlockBuildingParameters(
                image_fpath=harmonized.features,
                label_fpath=harmonized.labels,
                dem_pad=config.data.ingestion.datablocks.image_dem_pad,
                ignore_index=config.data.ingestion.datablocks.ignore_index,
            )
            ingest.run_blocks_building(
                world_grid,
                ingestion_paths.data_blocks,
                data_blocks_config,
                policy=policy,
                logger=logger,
            )

            assert logger.summary['data_blocks'] # typing
            d = logger.summary['data_blocks']['duration_sec']
            logger.log(
                'INFO',
                f'[COMPLETE] Canonical data blocks preparation (D_{d:.2f}s)'
            )

        # persist the config -> JSON
        ConfigController(ingestion_paths.config).persist(config.as_dict)

    # propagate all exceptions here
    except Exception as e:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Ingestion pipeline failed: {e}', exc_info=True)
        raise e

    # close logger
    finally:
        logger.log_sep()
        logger.close()
