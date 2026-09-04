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
Data harmonization processor for geospatial raster sources.
'''

# standard imports
import dataclasses
import os
import typing
# local imports
import landseg.geopipe.core as geo_core
import landseg.geopipe.harmonize.manifest as manifest
import landseg.geopipe.harmonize.rasters as rasters


# ----- public dataclass
@dataclasses.dataclass
class ProcessedRasters:
    '''Container for processed raster paths dictionaries.'''
    provenance: dict[str, str] = dataclasses.field(default_factory=dict)
    harmonized: dict[str, str] = dataclasses.field(default_factory=dict)
    finalized: dict[str, str] = dataclasses.field(default_factory=dict)


# ----- public functions
def harmonize_sources(
    compiled_sources: dict[str, manifest.ManifestEntry],
    output_dir: str,
    world_grid: geo_core.GridLayout,
    *,
    categorical_resampling: str,
    continuous_resampling: str,
) -> typing.Generator[str, None, ProcessedRasters]:
    '''Harmonize all compiled raster sources onto the canonical grid.'''
    features: list[str] = []
    labels: list[str] = []
    processed = ProcessedRasters()

    for path, cfg in compiled_sources.items():
        if not cfg:
            raise ValueError(f'No configuration found for raster {path}')

        tagged_name = f"{cfg['category']}_{cfg['name']}"
        resampling = (
            categorical_resampling
            if cfg['category'] in {'domains', 'domain', 'labels', 'label'}
            else continuous_resampling
        )
        out_vrt = os.path.join(output_dir, f'{tagged_name}.vrt')
        yield (
            f'Harmonizing raster {path} -> {out_vrt} '
            f'(resampling: {resampling})'
        )

        warped = _warp_and_tag_entry(
            os.path.abspath(path), out_vrt, cfg, world_grid, resampling
        )
        processed.provenance[tagged_name] = os.path.abspath(path)
        processed.harmonized[tagged_name] = warped

        match cfg['category']:
            case 'domains' | 'domain':
                processed.finalized[tagged_name] = warped
            case 'features' | 'feature':
                features.append(warped)
            case 'labels' | 'label':
                _tag_label_metadata(warped, cfg.get('categorical_specs'))
                labels.append(warped)

    processed.finalized.update(
        **(yield from rasters.stack_rasters(features, labels, output_dir))
    )
    return processed


# ----- private functions
def _warp_and_tag_entry(
    raw_path: str,
    output_path: str,
    cfg: manifest.ManifestEntry,
    world_grid: geo_core.GridLayout,
    resampling: str,
) -> str:
    '''Warp a single raster to grid and attach base tags.'''
    is_cat = cfg['category'] in {'domains', 'domain', 'labels', 'label'}
    warped = rasters.warp_to_grid(
        input_path=raw_path,
        output_path=output_path,
        world_grid=world_grid,
        is_categorical=is_cat,
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

    return warped


def _tag_label_metadata(
    warped_vrt: str,
    cat_specs: manifest.CategoricalSpecs | None,
) -> None:
    '''Attach label class configuration tags to warped VRT.'''
    if not cat_specs:
        raise ValueError('Missing categorical specs for label raster')

    rasters.add_tag_to_vrt(
        warped_vrt,
        num_cls=cat_specs['num_cls'],
        ignore_cls=cat_specs['ignore_cls'],
        class_name=cat_specs.get('class_name', {}),
        color_map=cat_specs.get('color_map', {}),
        taxonomy=cat_specs.get('taxonomy', {}),
    )
