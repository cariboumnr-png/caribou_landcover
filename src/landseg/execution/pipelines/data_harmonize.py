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

# standard imports
import os
# local imports
import landseg.artifacts as artifacts
import landseg.configs as configs
import landseg.geopipe.harmonize as harmonize_mod


# ----- public functions
def harmonize(config: configs.RootConfig) -> None:
    '''
    Execute the data-harmonize pipeline.

    Args:
        config: Resolved root configuration object.

    Returns:
        Summary report dictionary of the data harmonization execution.
    '''
    paths = artifacts.ArtifactPaths.from_config(config).data_harmonization
    paths.init()

    canvas_spec = harmonize_mod.create_canvas(
        reference_raster=config.data.harmonization.canvas.reference_raster,
        target_crs=config.data.harmonization.canvas.target_crs,
        target_resolution=config.data.harmonization.canvas.target_resolution
    )

    logger = harmonize_mod.HarmonizationLogger(
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

        # ----- get band mapping from JSON config
        ctrl = artifacts.Controller[dict].load_json_or_fail(
            config.data.harmonization.dataset_config
        )
        ctrl.hash(overwrite=False) # hash once
        dataset_config = ctrl.fetch()
        assert dataset_config
        band_mapping = dataset_config.get('band_mapping')

        # ----- categorical domain rasters
        for k, v in config.data.harmonization.raw_data.domains.items():
            harmonize_mod.validate_domain_raster_index(v, 1)
            output_path = _process_source(
                canvas_spec=canvas_spec,
                source={k: v},
                output_dir=paths.effective_root,
                tag='domains',
                resampling=config.data.harmonization.resampling_categorical,
                band_mapping=band_mapping,
                logger=logger
            )
            logger.add_finalized_raster(f'domain_{k}', output_path)
            # handle each domain separately

        # ----- continuous dev feature rasters
        output_path_dev_features = _process_source(
            canvas_spec=canvas_spec,
            source=config.data.harmonization.raw_data.dev_features,
            output_dir=paths.effective_root,
            tag='dev_features',
            resampling=config.data.harmonization.resampling_continuous,
            band_mapping=band_mapping,
            logger=logger
        )
        logger.add_finalized_raster('dev_features', output_path_dev_features)

        # ----- categorical dev label rasters
        output_path = _process_source(
            canvas_spec=canvas_spec,
            source=config.data.harmonization.raw_data.dev_labels,
            output_dir=paths.effective_root,
            tag='dev_labels',
            resampling=config.data.harmonization.resampling_categorical,
            band_mapping=band_mapping,
            logger=logger
        )
        logger.add_finalized_raster('dev_labels', output_path)

        # ----- test holdout feature rasters
        if config.data.harmonization.raw_data.test_features:
            output_path = _process_source(
                canvas_spec=canvas_spec,
                source=config.data.harmonization.raw_data.test_features,
                output_dir=paths.effective_root,
                tag='test_features',
                resampling=config.data.harmonization.resampling_continuous,
                band_mapping=band_mapping,
                logger=logger
            )
            logger.add_finalized_raster('test_features', output_path)

        # ----- test holdout label rasters
        if config.data.harmonization.raw_data.test_labels:
            output_path = _process_source(
                canvas_spec=canvas_spec,
                source=config.data.harmonization.raw_data.test_labels,
                output_dir=paths.effective_root,
                tag='test_labels',
                resampling=config.data.harmonization.resampling_categorical,
                band_mapping=band_mapping,
                logger=logger
            )
            logger.add_finalized_raster('test_labels', output_path)

        # ----- generate valid feature pixel mask
        mask_path = paths.valid_mask_raster
        logger.log('INFO', f'Generating valid mask raster: {mask_path}')
        harmonize_mod.unify_nodata_mask(output_path_dev_features, mask_path)
        logger.set_valid_mask_raster(mask_path)

        artifacts.Controller[dict](paths.config).persist(config.as_dict)

    except Exception as err:
        logger.set_summary_status('FAILED')
        logger.log('ERROR', f'Data harmonization failed: {err}')
        raise

    finally:
        logger.log_sep()
        logger.close()


# ----- private helpers
def _process_source(
    *,
    canvas_spec: harmonize_mod.CanvasSpec,
    source: dict[str, str],
    output_dir: str,
    tag: str,
    resampling: str,
    band_mapping: dict[str, dict[int, str]] | None,
    logger: harmonize_mod.HarmonizationLogger,
) -> str:
    '''Process one data source.'''
    aligned: list[str] = []
    band_mapping = band_mapping or {}

    # iterate through raster source
    for name, path in source.items():
        if not path or not os.path.exists(path):
            logger.log('INFO', f'Skipping missing {tag} layer: {name}')
            continue

        logger.add_source_provenance(f'{tag}_{name}', path)

        out_path = os.path.join(output_dir, f'harmonized_{tag}_{name}.vrt')
        logger.log(
            'INFO',
            f'Harmonizing {tag} layer [{name}] -> {out_path} '
            f'(resampling: {resampling})'
        )

        warped = harmonize_mod.warp_to_canvas(
            input_path=path,
            output_path=out_path,
            canvas=canvas_spec,
            is_categorical='label' in tag or 'domain' in tag,
            resampling_method=resampling,
            band_mapping=band_mapping.get(path)
        )
        aligned.append(warped)

        logger.add_harmonized_source(f'{tag}_{name}', warped)

    # stack if multiband
    if len(aligned) > 1:
        out_path = os.path.join(output_dir, f'harmonized_{tag}_STACKED.vrt')
        logger.log(
            'INFO',
            f' |- {len(aligned)} {tag} layers stacked to {out_path}'
        )
        harmonize_mod.stack_canonical_raster(aligned, out_path)
    else:
        name = list(source.keys())[0] # single item
        out_path = os.path.join(output_dir, f'harmonized_{tag}_{name}.vrt')

    return out_path
