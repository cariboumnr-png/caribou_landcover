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
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.etl as etl


# ----- public functions
def harmonize(config: configs.RootConfig) -> dict[str, typing.Any]:
    '''
    Execute the data-harmonize pipeline.

    Args:
        config: Resolved root configuration object.

    Returns:
        Summary report dictionary of the ETL execution.
    '''
    etl_cfg = config.etl
    out_dpath = os.path.abspath(etl_cfg.output_dpath)
    etl_paths = artifacts.ETLPaths(out_dpath)
    os.makedirs(out_dpath, exist_ok=True)

    logger = etl.HarmonizationLogger(
        name='data-harmonize',
        log_file=etl_paths.report,
        enable_file_log=False
    )

    canvas_spec = etl.create_canvas(
        target_crs=etl_cfg.target_crs,
        target_resolution=etl_cfg.target_resolution,
        reference_raster=etl_cfg.reference_raster
    )

    logger.init_summary(
        target_crs=canvas_spec.crs,
        target_resolution=canvas_spec.resolution,
        output_dpath=out_dpath
    )
    logger.set_grid_shape(canvas_spec.height, canvas_spec.width)
    logger.log(
        'INFO',
        f'Starting data harmonization pipeline... '
        f'Target CRS={canvas_spec.crs}, Res={canvas_spec.resolution}m'
    )

    aligned_features: list[str] = []
    try:
        # 1. Process explicit continuous feature rasters
        for name, path in etl_cfg.features.items():
            if not path or not os.path.exists(path):
                logger.log('INFO', f'Skipping missing feature layer: {name}')
                continue

            logger.add_source_provenance(name, path)
            out_path = etl_paths.harmonized_raster(name)
            logger.log(
                'INFO',
                f'Harmonizing feature [{name}] -> {out_path} '
                f'(resampling: {etl_cfg.resampling_continuous})'
            )
            warped = etl.warp_to_canvas(
                input_path=path,
                output_path=out_path,
                canvas=canvas_spec,
                is_categorical=False,
                resampling_method=etl_cfg.resampling_continuous
            )
            logger.add_harmonized_source(name, warped)
            aligned_features.append(warped)

        # 2. Process explicit categorical label rasters
        for name, path in etl_cfg.labels.items():
            if not path or not os.path.exists(path):
                logger.log('INFO', f'Skipping missing label layer: {name}')
                continue

            logger.add_source_provenance(name, path)
            out_path = etl_paths.harmonized_raster(name)
            logger.log(
                'INFO',
                f'Harmonizing label [{name}] -> {out_path} '
                f'(resampling: {etl_cfg.resampling_categorical})'
            )
            warped = etl.warp_to_canvas(
                input_path=path,
                output_path=out_path,
                canvas=canvas_spec,
                is_categorical=True,
                resampling_method=etl_cfg.resampling_categorical
            )
            logger.add_harmonized_source(name, warped)

        # 3. Stack feature rasters into composite if multiple exist
        composite_path = ''
        if len(aligned_features) > 1:
            composite_path = etl_paths.composite_raster
            logger.log(
                'INFO',
                f'Stacking {len(aligned_features)} feature layers...'
            )
            etl.stack_canonical_raster(aligned_features, composite_path)
            logger.set_composite_raster(composite_path)

        # 4. Generate valid pixel mask across features
        mask_path = ''
        if composite_path:
            mask_path = etl_paths.valid_mask_raster
            logger.log('INFO', f'Generating valid pixel mask raster: {mask_path}')
            etl.unify_nodata_mask(composite_path, mask_path)
            logger.set_valid_mask_raster(mask_path)

        logger.set_summary_status('SUCCESS')
        logger.log('INFO', 'Data harmonization completed successfully.')
    except Exception as err:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Data harmonization failed: {err}')
        raise
    finally:
        report = dict(logger.summary) if logger.summary else {}
        logger.close()

    return report
