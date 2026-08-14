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
Data harmonization pipeline.
'''

# standard imports
import os
# local imports
import landseg.geopipe.harmonize as harmonize


def process_source(
    *,
    source_fpaths: dict[str, str],
    source_configs: dict[str, harmonize.DatasetConfigItem],
    output_dir: str,
    canvas_spec: harmonize.CanvasSpec,
    resampling: str,
    logger: harmonize.HarmonizationLogger,
) -> str:
    '''Process one data source.'''
    aligned: list[str] = []
    out_path: str = ''
    category: str = ''
    # iterate through raster source
    for name, path in source_fpaths.items():
        path = os.path.abspath(path) # align with config
        cfg = source_configs.get(path)
        if not cfg:
            raise ValueError(f'No configuration found for raster {path}')

        if not (cfg['name'] == name and cfg['path'] == path):
            raise ValueError(
                f'Raster name or file path do not match between '
                f'\nsource: {name}: {path} | '
                f'\nconfig: {cfg["name"]}: {cfg["path"]}'
            )
        category = cfg['category']
        tagged_name = f'{cfg["category"]}_{cfg["name"]}'


        logger.add_source_provenance(tagged_name, path)
        out_path = os.path.join(output_dir, f'{tagged_name}.vrt')

        logger.log(
            'INFO',
            f'Harmonizing {cfg["category"]} layer [{cfg["name"]}] -> '
            f'{out_path} (resampling: {resampling})'
        )

        is_categorical = any(c in cfg['category'] for c in ['domain', 'label'])
        warped = harmonize.warp_to_canvas(
            input_path=path,
            output_path=out_path,
            canvas=canvas_spec,
            is_categorical=is_categorical,
            resampling_method=resampling,
        )

        if cfg['band_mapping']:
            harmonize.add_band_description_to_vrt(warped, cfg['band_mapping'])

        if cfg['label_specs']:
            harmonize.add_tag_to_vrt(
                warped,
                num_cls=cfg['label_specs']['num_cls'],
                ignore_cls=cfg['label_specs']['ignore_cls'],
                class_name=cfg['label_specs'].get('class_name', []),
                reclass=cfg['label_specs'].get('reclass', {}),
                reclass_name=cfg['label_specs'].get('reclass_name', {})
            )

        aligned.append(warped)

        logger.add_harmonized_source(tagged_name, warped)

    assert aligned # should not be empty by now

    # stack if multiband
    if len(aligned) > 1:
        out_path = os.path.join(output_dir, f'harmonized_{category}_STACKED.vrt')
        logger.log(
            'INFO',
            f' |- {len(aligned)} {category} layers stacked to {out_path}'
        )
        harmonize.stack_canonical_raster(aligned, out_path)

    return out_path
