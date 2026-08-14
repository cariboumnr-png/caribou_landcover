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
    paths = artifacts.ArtifactPaths.from_config(config).data_harmonization
    paths.init()

    cfg = config.data.harmonization

    canvas_spec = harmonize.create_canvas(
        reference_raster=cfg.canvas.reference_raster,
        target_crs=cfg.canvas.target_crs,
        target_resolution=cfg.canvas.target_resolution
    )

    logger = harmonize.HarmonizationLogger(
        name='data-harmonize',
        log_file=paths.report,
        enable_file_log=False
    )

    logger.init_summary(
        run_id=paths.run_id,
        target_crs=canvas_spec.crs,
        target_resolution=canvas_spec.resolution
    )
    logger.set_grid_shape(canvas_spec.height, canvas_spec.width)

    try:
        logger.log_sep()
        logger.log(
            'INFO',
            f'Starting data harmonization pipeline... '
            f'Target CRS={canvas_spec.crs}, Res={canvas_spec.resolution}m'
        )

        # ----- read dataset JSON config
        cfg_fpath = cfg.dataset_config
        dataset_cfg = harmonize.read_dataset_config(cfg_fpath)

        # ----- categorical domain rasters
        for k, v in cfg.raw_data.domains.items():
            harmonize.validate_domain_raster_index(v, 1)
            output_path = harmonize.process_source(
                source_fpaths={k: v},
                source_configs=dataset_cfg,
                output_dir=paths.effective_root,
                canvas_spec=canvas_spec,
                resampling=cfg.resampling_categorical,
                logger=logger
            )
            logger.add_finalized_raster(f'domain_{k}', output_path)
            # handle each domain separately

        # ----- continuous dev feature rasters
        output_path_dev_features = harmonize.process_source(
            source_fpaths=cfg.raw_data.dev_features,
            source_configs=dataset_cfg,
            output_dir=paths.effective_root,
            canvas_spec=canvas_spec,
            resampling=cfg.resampling_continuous,
            logger=logger
        )
        logger.add_finalized_raster('dev_features', output_path_dev_features)

        # ----- categorical dev label rasters
        output_path = harmonize.process_source(
            source_fpaths=cfg.raw_data.dev_labels,
            source_configs=dataset_cfg,
            output_dir=paths.effective_root,
            canvas_spec=canvas_spec,
            resampling=cfg.resampling_categorical,
            logger=logger
        )
        logger.add_finalized_raster('dev_labels', output_path)

        # ----- test holdout feature rasters
        if cfg.raw_data.test_features:
            output_path = harmonize.process_source(
                source_fpaths=cfg.raw_data.test_features,
                source_configs=dataset_cfg,
                output_dir=paths.effective_root,
                canvas_spec=canvas_spec,
                resampling=cfg.resampling_continuous,
                logger=logger
            )
            logger.add_finalized_raster('test_features', output_path)

        # ----- test holdout label rasters
        if cfg.raw_data.test_labels:
            output_path = harmonize.process_source(
                source_fpaths=cfg.raw_data.test_labels,
                source_configs=dataset_cfg,
                output_dir=paths.effective_root,
                canvas_spec=canvas_spec,
                resampling=cfg.resampling_categorical,
                logger=logger
            )
            logger.add_finalized_raster('test_labels', output_path)

        # ----- generate valid feature pixel mask
        mask_path = paths.valid_mask_raster
        logger.log('INFO', f'Generating valid mask raster: {mask_path}')
        harmonize.unify_nodata_mask(output_path_dev_features, mask_path)
        logger.set_valid_mask_raster(mask_path)

        artifacts.Controller[dict](paths.config).persist(config.as_dict)

    except Exception as err:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Data harmonization failed: {err}')
        raise

    finally:
        logger.log_sep()
        logger.close()
