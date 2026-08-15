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
import landseg.geopipe.harmonize as harmonize


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
    dev_features: dict[str, str] = dataclasses.field(default_factory=dict)
    dev_labels: dict[str, str] = dataclasses.field(default_factory=dict)
    test_features: dict[str, str] = dataclasses.field(default_factory=dict)
    test_labels: dict[str, str] = dataclasses.field(default_factory=dict)

    def add_raster(
        self,
        category: str,
        name: str,
        filepath: str,
    ) -> None:
        '''Add raster by category'''
        match category:
            case 'domains': self.domains.update({name: filepath})
            case 'dev_features': self.dev_features.update({name: filepath})
            case 'dev_labels': self.dev_labels.update({name: filepath})
            case 'test_features': self.test_features.update({name: filepath})
            case 'test_labels': self.test_labels.update({name: filepath})
            case _: raise ValueError(f'Unknown raster category: {category}')


# TODO enforce input label rasters to be single banded
def process_source(
    compiled_sources: dict[str, harmonize.DatasetConfigItem],
    output_dir: str,
    canvas_spec: harmonize.CanvasSpec,
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
        is_categorical = category in ['domains', 'dev_labels', 'test_labels']
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

        warped = harmonize.warp_to_canvas(
            input_path=path,
            output_path=out_path,
            canvas=canvas_spec,
            is_categorical=is_categorical,
            resampling_method=resampling,
        )

        if category == 'domains':
            processed.harmonized.update({tagged_name: warped})
            processed.finalized.update({tagged_name: warped})
            continue # fast tracking domain rasters

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

        aligned.add_raster(category, tagged_name, out_path)
        processed.harmonized.update({tagged_name: warped})

    stacked: dict[str, str] = {}
    gen = _stack_rasters(aligned, output_dir)
    while True:
        try:
            yield next(gen)
        except StopIteration as s:
            stacked = s.value
            break

    processed.finalized.update(**stacked)
    return processed


def _stack_rasters(
    aligned: _AlignedRasters,
    output_dir: str,
) -> typing.Generator[str, None, dict[str, str]]:
    '''Stack feature and label rasters if applicable.'''

    def _out_path(tag: str) -> str:
        return os.path.join(output_dir, f'harmonized_{tag}_STACKED.vrt')

    stacked: dict[str, str] = {}
    yield 'Stacking rasters if applicable'

    fpaths = list(aligned.dev_features.values())
    n = len(fpaths)
    if n == 0:
        pass
    elif n == 1:
        stacked.update({'dev_features': fpaths[0]})
    else:
        out_path = _out_path('dev_features')
        harmonize.stack_canonical_raster(fpaths, out_path)
        stacked.update({'dev_features': out_path})
        yield f'Development feature rasters stacked to {out_path} (n={n})'

    fpaths = list(aligned.dev_labels.values())
    n = len(fpaths)
    if n == 0:
        pass
    elif n == 1:
        stacked.update({'dev_labels': fpaths[0]})
    else:
        out_path = _out_path('dev_labels')
        harmonize.stack_canonical_raster(fpaths, out_path)
        stacked.update({'dev_labels': out_path})
        yield f'Development label rasters stacked to {out_path} (n={n})'

    fpaths = list(aligned.test_features.values())
    n = len(fpaths)
    if n == 0:
        pass
    elif n == 1:
        stacked.update({'test_features': fpaths[0]})
    else:
        out_path = _out_path('test_features')
        harmonize.stack_canonical_raster(fpaths, out_path)
        stacked.update({'test_features': out_path})
        yield f'Test feature rasters stacked to {out_path} (n={n})'

    fpaths = list(aligned.test_labels.values())
    n = len(fpaths)
    if n == 0:
        pass
    elif n == 1:
        stacked.update({'test_labels': fpaths[0]})
    else:
        out_path = _out_path('test_labels')
        harmonize.stack_canonical_raster(fpaths, out_path)
        stacked.update({'test_labels': out_path})
        yield f'Test label rasters stacked to {out_path} (n={n})'

    return stacked
