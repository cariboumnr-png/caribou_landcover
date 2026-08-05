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
Data preparation (experiment-materialized) pipeline.

Splits raw blocks into train/val(/test), computes train-only band
statistics, normalizes all splits, and emits the final dataset schema.
'''

# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe.prepare as prepare_data


def prepare(config: configs.RootConfig):
    '''
    Run the preparation pipeline for an experiment.

    Steps:
    1) Parse current data blocks catalog and schema by config.
    2) Split the blocks into train/val(/test) with configured hydration.
    3) Normalize all blocks using image stats from the train split.
    4) Build schame for downstream consumption.

    Args:
        config: RootConfig with preparation settings.
    '''
    # artifact paths
    artifact_paths = artifacts.ArtifactPaths.from_config(config)
    paths = artifact_paths.data_preparation

    # init a PreparationLogger
    logger = prepare_data.PreparationLogger(
        name='prep',
        log_file=paths.report,
        enable_file_log=False
    )
    logger.init_summary(run_id='prepare')

    try:
        logger.log_sep()

        # resolve lifecycle policy dynamically
        policy = (
            artifacts.LifecyclePolicy.REBUILD
            if config.data.preparation.rebuild
            else artifacts.LifecyclePolicy.BUILD_IF_MISSING
        )


        # parse catalog from data ingestion stage
        parsed_catalog = prepare_data.data_blocks_adapter(
            artifact_paths.data_ingestion.data_blocks.dev.catalog,
            artifact_paths.data_ingestion.data_blocks.dev.schema,
            artifact_paths.data_ingestion.data_blocks.test.catalog,
            config=config.data.preparation.catalog
        )

        # datablocks partition
        logger.log('INFO', '[START] Dataset partitioning splits')
        # data transform config aliases
        partition = config.data.preparation.partition
        scoring = config.data.preparation.scoring
        hydration = config.data.preparation.hydration
        # partition config
        partition_config = prepare_data.PartitionParameters(
            val_test_ratios=(partition.val_ratio, partition.test_ratio),
            buffer_step=partition.buffer_step,
            reward_ratios=scoring.reward,
            scoring_alpha=scoring.alpha,
            scoring_beta=scoring.beta,
            max_skew_rate=hydration.max_skew_rate,
            block_spec=config.data.ingestion.grid.tile_specs_tuple
        )
        prepare_data.run_datablocks_partition(
            parsed_catalog,
            paths,
            partition_config,
            policy=policy,
            logger=logger,
        )
        assert logger.summary
        assert logger.summary['data_partition']
        d = logger.summary['data_partition']['duration_sec']
        logger.log('INFO', f'[COMPLETE] Dataset partitioning splits (D_{d:.2f}s)')

        # normalize
        logger.log('INFO', '[START] Block normalization')
        prepare_data.run_normalize_blocks(
            paths,
            policy=policy,
            logger=logger
        )
        assert logger.summary['normalization']
        d = logger.summary['normalization']['duration_sec']
        logger.log('INFO', f'[COMPLETE] Block normalization (D_{d:.2f}s)')

        # build schema
        logger.log('INFO', '[START] Transform schema building')
        prepare_data.build_schema(
            paths,
            policy=policy,
            logger=logger
        )
        assert logger.summary['schema']
        d = logger.summary['schema']['duration_sec']
        logger.log('INFO', f'[COMPLETE] Transform schema building (D_{d:.2f}s)')

        # write config JSON sidecar upon successful execution
        artifacts.Controller[dict](paths.config).persist(config.as_dict)

    except Exception as e:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Preparation pipeline failed: {e}', exc_info=True)
        raise e
    finally:
        logger.log_sep()
        logger.close()
