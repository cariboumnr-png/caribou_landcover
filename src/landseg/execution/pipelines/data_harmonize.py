# =========================================================================== #
#           Copyright © His Majesty the King in right of Ontario,           #
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
Data harmonization ETL pipeline command implementation.
'''

# standard imports
import os
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.etl as etl


# ----- public functions
def harmonize(config: configs.RootConfig) -> None:
    '''
    Execute the data-harmonize pipeline.

    Args:
        config: Resolved root configuration object.

    Returns:
        Summary report dictionary of the ETL execution.
    '''

    def _process_source(
        *,
        source: dict[str, str],
        output_composite: str,
        tag: str,
        resampling: str,
        logger: etl.HarmonizationLogger,
    ) -> None:
        '''Process one data source.'''
        aligned: list[str] = []
        for name, path in source.items():
            if not path or not os.path.exists(path):
                logger.log('INFO', f'Skipping missing {tag} layer: {name}')
                continue

            logger.add_source_provenance(name, path)
            out_path = paths.harmonized_raster(f'{tag}_{name}')
            logger.log('INFO',
                f'Harmonizing {tag} layer [{name}] -> {out_path} '
                f'(resampling: {resampling})'
            )
            warped = etl.warp_to_canvas(
                input_path=path,
                output_path=out_path,
                canvas=canvas_spec,
                is_categorical=True,
                resampling_method=resampling
            )
            aligned.append(warped)
            logger.add_harmonized_source(name, warped)
        logger.log('INFO', f'Stacking {len(aligned)} {tag} layers')
        etl.stack_canonical_raster(aligned, output_composite)

    paths = artifacts.ArtifactPaths.from_config(config).etl
    paths.init()

    canvas_spec = etl.create_canvas(
        reference_raster=config.etl.canvas.reference_raster,
        target_crs=config.etl.canvas.target_crs,
        target_resolution=config.etl.canvas.target_resolution
    )

    logger = etl.HarmonizationLogger(
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

        # ----- continous feature rasters
        # harmonize feature rasters
        _process_source(
            source=config.etl.raw_data.features,
            output_composite=paths.feature_raster,
            tag='feature',
            resampling=config.etl.resampling_continuous,
            logger=logger
        )
        logger.add_stacked_raster('features', paths.feature_raster)

        # ----- categorical label rasters
        _process_source(
            source=config.etl.raw_data.labels,
            output_composite=paths.label_raster,
            tag='label',
            resampling=config.etl.resampling_categorical,
            logger=logger
        )
        logger.add_stacked_raster('labels', paths.feature_raster)

        # -----categorical domain rasters
        _process_source(
            source=config.etl.raw_data.domains,
            output_composite=paths.domain_raster,
            tag='domain',
            resampling=config.etl.resampling_categorical,
            logger=logger
        )
        logger.add_stacked_raster('domains', paths.domain_raster)

        # ----- generate valid feature pixel mask
        mask_path = paths.valid_mask_raster
        logger.log('INFO', f'Generating valid mask raster: {mask_path}')
        etl.unify_nodata_mask(paths.feature_raster, mask_path)
        logger.set_valid_mask_raster(mask_path)

        artifacts.Controller[dict](paths.config).persist(config.as_dict)

    except Exception as err:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Data harmonization failed: {err}')
        raise

    finally:
        logger.log_sep()
        logger.close()
