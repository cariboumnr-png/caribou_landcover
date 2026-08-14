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

        # read dataset JSON config
        compiled = harmonize.compile_dataset_manifest(cfg.dataset_manifest)
        gen = harmonize.process_source(
            compiled,
            paths.effective_root,
            canvas_spec,
            categorical_resampling=cfg.resampling_categorical,
            continuous_resampling=cfg.resampling_continuous,
        )

        # process sources
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

        # generate valid feature pixel mask if dev feature raster is provided
        if processed.finalized.get('dev_features'):
            output_path = processed.finalized['dev_features']
            mask_path = paths.valid_mask_raster
            logger.log('INFO', f'Generating valid mask raster: {mask_path}')
            harmonize.unify_nodata_mask(output_path, mask_path)
            logger.set_valid_mask_raster(mask_path)

        # persist the whole config dict
        artifacts.Controller[dict](paths.config).persist(config.as_dict)

    except Exception as err:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Data harmonization failed: {err}')
        raise

    finally:
        logger.log_sep()
        logger.close()
