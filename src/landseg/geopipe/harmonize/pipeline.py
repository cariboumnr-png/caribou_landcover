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


@dataclasses.dataclass
class _AlignedRasters:
    '''Container for aligned rasters (`.vrt` file paths) by category.'''
    domains: dict[str, str] = dataclasses.field(default_factory=dict)
    features: dict[str, str] = dataclasses.field(default_factory=dict)
    labels: dict[str, str] = dataclasses.field(default_factory=dict)

    def add_raster(
        self,
        category: str,
        name: str,
        filepath: str,
    ) -> None:
        '''Add raster by category.'''
        match category:
            case 'domains' | 'domain':
                self.domains.update({name: filepath})
            case 'features' | 'feature':
                self.features.update({name: filepath})
            case 'labels' | 'label':
                self.labels.update({name: filepath})
            case _:
                raise ValueError(f'Unknown raster category: {category}')


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
    aligned = _AlignedRasters()
    processed = ProcessedRasters()

    # iterate through raster source
    for path, cfg in compiled_sources.items():
        path = os.path.abspath(path) # guard
        if not cfg:
            raise ValueError(f'No configuration found for raster {path}')

        name = cfg['name']
        category = cfg['category']
        tagged_name = f'{category}_{name}'
        is_categorical = category in ['domains', 'domain', 'labels', 'label']
        resampling = (
            categorical_resampling
            if is_categorical
            else continuous_resampling
        )

        processed.provenance.update({tagged_name: path})
        out_path = os.path.join(output_dir, f'{tagged_name}.vrt')

        yield (
            f'Harmonizing {category} layer [{name}] -> {out_path} '
            f'(resampling: {resampling})'
        )

        warped = rasters.warp_to_grid(
            input_path=path,
            output_path=out_path,
            world_grid=world_grid,
            is_categorical=is_categorical,
            resampling_method=resampling,
        )

        if category in ['domains', 'domain']:
            processed.harmonized.update({tagged_name: warped})
            processed.finalized.update({tagged_name: warped})
            continue # fast tracking domain rasters

        rasters.add_band_description_to_vrt(warped, cfg['band_mapping'])

        if cfg['label_specs']:
            rasters.add_tag_to_vrt(
                warped,
                num_cls=cfg['label_specs']['num_cls'],
                ignore_cls=cfg['label_specs']['ignore_cls'],
                class_name=cfg['label_specs'].get('class_name', {}),
                reclass=cfg['label_specs'].get('reclass', {}),
                reclass_name=cfg['label_specs'].get('reclass_name', {}),
                color_map=cfg['label_specs'].get('color_map', {}),
                taxonomy=cfg['label_specs'].get('taxonomy', {}),
            )

        aligned.add_raster(category, tagged_name, out_path)
        processed.harmonized.update({tagged_name: warped})

    stacked: dict[str, str] = {}
    gen = rasters.stack_rasters(
        list(aligned.features.values()),
        list(aligned.labels.values()),
        output_dir
    )
    while True:
        try:
            yield next(gen)
        except StopIteration as s:
            stacked = s.value
            break

    processed.finalized.update(**stacked)
    return processed
