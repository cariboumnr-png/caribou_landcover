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
import dataclasses
import os
import typing
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.harmonize.rasters as rasters
import landseg.geopipe.harmonize.manifest as manifest


@dataclasses.dataclass
class ProcessedRasters:
    '''Container for processed raster paths dictionaries.'''
    provenance: dict[str, str] = dataclasses.field(default_factory=dict)
    harmonized: dict[str, str] = dataclasses.field(default_factory=dict)
    finalized: dict[str, str] = dataclasses.field(default_factory=dict)


# note: input label rasters must be single banded
def process_source(
    compiled_sources: dict[str, manifest.DatasetConfigItem],
    output_dir: str,
    world_grid: geo_core.GridLayout,
    *,
    categorical_resampling: str,
    continuous_resampling: str,
) -> typing.Generator[str, None, ProcessedRasters]:
    '''Process one data source.'''
    features: list[str] = []
    labels: list[str] = []
    processed = ProcessedRasters()

    # iterate through raster source
    for path, cfg in compiled_sources.items():
        path = os.path.abspath(path) # guard
        if not cfg:
            raise ValueError(f'No configuration found for raster {path}')

        category = cfg['category']
        tagged_name = f'{category}_{cfg['name']}'
        is_categorical = category in ['domains', 'domain', 'labels', 'label']
        resampling = (
            categorical_resampling
            if is_categorical
            else continuous_resampling
        )

        processed.provenance.update({tagged_name: path})
        outp = os.path.join(output_dir, f'{tagged_name}.vrt')

        yield f'Harmonizing raster {path} -> {outp} (resampling: {resampling})'

        warped = rasters.warp_to_grid(
            input_path=path,
            output_path=outp,
            world_grid=world_grid,
            is_categorical=is_categorical,
            resampling_method=resampling,
        )

        if cfg.get('band_mapping'):
            rasters.add_band_description_to_vrt(warped, cfg['band_mapping'])

        cat_specs = cfg.get('categorical_specs')
        if cat_specs and 'index_base' in cat_specs:
            rasters.add_tag_to_vrt(warped, index_base=cat_specs['index_base'])

        schemes = cfg.get('schemes')
        if schemes:
            rasters.add_tag_to_vrt(warped, schemes={cfg['name']: schemes})

        match category:
            case 'domains' | 'domain':
                processed.harmonized.update({tagged_name: warped})
                processed.finalized.update({tagged_name: warped}) # no further
            case 'features' | 'feature':
                processed.harmonized.update({tagged_name: warped})
                features.append(warped) # for stacking
            case 'labels' | 'labels':
                processed.harmonized.update({tagged_name: warped})
                assert cat_specs, f'missing categorical specs for {tagged_name}'
                rasters.add_tag_to_vrt(
                    warped,
                    num_cls=cat_specs['num_cls'],
                    ignore_cls=cat_specs['ignore_cls'],
                    class_name=cat_specs.get('class_name', {}),
                    color_map=cat_specs.get('color_map', {}),
                    taxonomy=cat_specs.get('taxonomy', {}),
                )
                labels.append(warped) # for stacking

    stacked: dict[str, str] = {}
    gen = rasters.stack_rasters(features, labels, output_dir)
    while True:
        try:
            yield next(gen)
        except StopIteration as s:
            stacked = s.value
            break

    processed.finalized.update(**stacked)
    return processed
