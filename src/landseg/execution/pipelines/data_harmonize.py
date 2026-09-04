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

# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe.grid as grid
import landseg.geopipe.harmonize as harmonize


# ----- public functions
def exec_harmonize_data(config: configs.RootConfig) -> None:
    '''
    Execute the data-harmonize pipeline.

    Args:
        config: Resolved root configuration object.

    Returns:
        Summary report dictionary of the data harmonization execution.
    '''
    root_paths = artifacts.ArtifactPaths.from_config(config)

    paths = root_paths.data_harmonization
    paths.init()

    logger = harmonize.HarmonizationLogger(
        name='data-harmonize',
        log_file=paths.report,
        enable_file_log=False
    )
    logger.init_summary(run_id=paths.run_id)

    try:
        logger.log_sep()

        # load world grid - will raise if grid not present (run prior pipeline)
        logger.log('INFO', '[START] Loading world grid from configuration')
        grid_fpath, world_grid = grid.load_grid_from_config(config.data.world_grid)
        gid = world_grid.gid
        grid_report: harmonize.WorldGridReport = {
            'grid_fpath': grid_fpath,
            'grid_id': gid,
            'crs': world_grid.crs,
            'pixel_size': world_grid.pixel_size,
            'tile_size': world_grid.tile_size,
            'tile_overlap': world_grid.tile_overlap,
        }
        logger.set_world_grid_report(grid_report)
        logger.log('INFO', f'[COMPLETE] World grid loaded: {gid}')

        logger.log('INFO', f'[START] Harmonizing data onto grid: {gid}')
        cfg = config.data.harmonization
        compiled = harmonize.compile_dataset_manifest(cfg.dataset_manifest)
        gen = harmonize.harmonize_sources(
            compiled,
            paths.effective_root,
            world_grid,
            categorical_resampling=cfg.resampling_categorical,
            continuous_resampling=cfg.resampling_continuous,
        )

        processed: harmonize.ProcessedRasters
        while True:
            try:
                log_message = next(gen)
                logger.log('INFO', log_message)
            except StopIteration as s:
                processed = s.value
                break

        # log processed file paths
        for name, path in processed.provenance.items():
            logger.add_source_provenance(name, path)

        for name, path in processed.harmonized.items():
            logger.add_harmonized_source(name, path)

        for name, path in processed.finalized.items():
            logger.add_finalized_raster(name, path)

        # generate valid feature pixel mask if feature raster is provided
        feature_raster = processed.finalized.get('features')
        if feature_raster:
            mask_path = paths.valid_mask_raster
            logger.log('INFO', f'Generating valid mask raster: {mask_path}')
            harmonize.unify_nodata_mask(feature_raster, mask_path)
            logger.set_valid_mask_raster(mask_path)


        # persist the whole config dict
        artifacts.Controller[dict](paths.config).persist(config.as_dict)

        logger.log('INFO', '[COMPLETE] Harmonization finished')

    except Exception as err:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Data harmonization failed: {err}')
        raise

    finally:
        logger.log_sep()
        logger.close()
