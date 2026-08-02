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
ETL pipeline orchestrator coordinating raster harmonization.
'''

# standard imports
import os
import typing
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.etl.logger as etl_logger
import landseg.etl.raster_ops as raster_ops
import landseg.etl.spatial as spatial


# ----- public functions
def orchestrate_etl(root_config: configs.RootConfig) -> dict[str, typing.Any]:
    '''
    Orchestrate full data harmonization ETL execution based on `RootConfig`.

    Args:
        root_config: Configured root configuration dataclass.

    Returns:
        Execution report summary dictionary.
    '''
    etl_cfg = root_config.etl
    out_dpath = os.path.abspath(etl_cfg.output_dpath)
    os.makedirs(out_dpath, exist_ok=True)

    etl_paths = artifacts.ETLPaths(out_dpath)
    canvas_spec = _resolve_canvas(etl_cfg)

    logger = etl_logger.HarmonizationLogger(
        name='data-harmonize',
        log_file=etl_paths.report,
        enable_file_log=False
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
            warped = spatial.warp_to_canvas(
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
            warped = spatial.warp_to_canvas(
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
            raster_ops.stack_canonical_raster(aligned_features, composite_path)
            logger.set_composite_raster(composite_path)

        # 4. Generate valid pixel mask across features
        mask_path = ''
        if composite_path:
            mask_path = etl_paths.valid_mask_raster
            logger.log('INFO', f'Generating valid pixel mask raster: {mask_path}')
            raster_ops.unify_nodata_mask(composite_path, mask_path)
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


# ----- private functions
def _resolve_canvas(etl_cfg: configs.sec.ETLConfig) -> spatial.CanvasSpec:
    '''Resolve `CanvasSpec` from configuration reference raster or default bounds.'''
    if etl_cfg.reference_raster and os.path.exists(etl_cfg.reference_raster):
        return spatial.from_reference_raster(
            etl_cfg.reference_raster,
            target_crs=etl_cfg.target_crs,
            target_resolution=etl_cfg.target_resolution
        )
    return spatial.CanvasSpec(
        crs=etl_cfg.target_crs,
        resolution=etl_cfg.target_resolution,
        bounds=(500000.0, 600000.0, 510240.0, 610240.0) # 512x512 at 20m
    )
