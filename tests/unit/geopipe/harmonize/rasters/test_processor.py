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

# pylint: disable=no-member

'''Unit tests for data harmonization processor (processor.py).'''


# standard imports
import os
import typing
# third-party imports
import rasterio
# local imports
import landseg.geopipe.harmonize as harmonize
import landseg.geopipe.harmonize.rasters.processor as processor



# ----- `process_source` tests
def test_process_source_features_and_labels(tmp_path, dummy_geotiff_factory):
    '''
    Given: Two feature rasters and one label raster.
    When: `process_source` is executed.
    Then: Warp rasters to canvas and stack multiple features into one VRT.
    '''
    ref_path = dummy_geotiff_factory(
        filename='ref.tif', width=16, height=16, bands=1
    )
    s2_path = dummy_geotiff_factory(
        filename='s2.tif', width=16, height=16, bands=3
    )
    dem_path = dummy_geotiff_factory(
        filename='dem.tif', width=16, height=16, bands=1
    )
    lbl_path = dummy_geotiff_factory(
        filename='lbl.tif', width=16, height=16, bands=1
    )

    canvas_spec = harmonize.create_canvas(
        reference_raster=str(ref_path),
        target_crs='EPSG:3161',
        target_resolution=20.0
    )

    compiled: dict[str, harmonize.DatasetConfigItem] = {
        str(s2_path): {
            'name': 's2',
            'path': str(s2_path),
            'category': 'features',
            'band_mapping': {1: 'blue', 2: 'green', 3: 'red'},
            'label_specs': None,
        },
        str(dem_path): {
            'name': 'dem',
            'path': str(dem_path),
            'category': 'features',
            'band_mapping': {1: 'elevation'},
            'label_specs': None,
        },
        str(lbl_path): {
            'name': 'landcover',
            'path': str(lbl_path),
            'category': 'labels',
            'band_mapping': None,
            'label_specs': {'num_cls': 2, 'ignore_cls': [255]},
        },
    }

    out_dir = str(tmp_path / 'harmonized')
    os.makedirs(out_dir, exist_ok=True)

    gen = processor.process_source(
        compiled,
        out_dir,
        canvas_spec,
        categorical_resampling='nearest',
        continuous_resampling='bilinear',
    )

    res: typing.Any = None
    try:
        while True:
            next(gen)
    except StopIteration as s:
        res = s.value

    assert isinstance(res, processor.ProcessedRasters)
    assert 'features' in res.finalized
    assert 'labels' in res.finalized
    assert res.finalized['features'].endswith(
        'harmonized_features_STACKED.vrt'
    )
    assert os.path.exists(res.finalized['features'])

    # verify stacked band count: 3 (s2) + 1 (dem) = 4
    with rasterio.open(res.finalized['features']) as src:
        assert src.count == 4


def test_process_source_domains(tmp_path, dummy_geotiff_factory):
    '''
    Given: A domain categorical raster.
    When: `process_source` is executed.
    Then: Fast-track domain raster into finalized without stacking.
    '''
    ref_path = dummy_geotiff_factory(
        filename='ref.tif', width=16, height=16, bands=1
    )
    dom_path = dummy_geotiff_factory(
        filename='eco.tif', width=16, height=16, bands=1
    )

    canvas_spec = harmonize.create_canvas(
        reference_raster=str(ref_path),
        target_crs='EPSG:3161',
        target_resolution=20.0
    )

    compiled: dict[str, harmonize.DatasetConfigItem] = {
        str(dom_path): {
            'name': 'ecodistrict',
            'path': str(dom_path),
            'category': 'domains',
            'band_mapping': None,
            'label_specs': None,
        },
    }

    out_dir = str(tmp_path / 'harmonized')
    os.makedirs(out_dir, exist_ok=True)

    gen = processor.process_source(
        compiled,
        out_dir,
        canvas_spec,
        categorical_resampling='nearest',
        continuous_resampling='bilinear',
    )

    res: typing.Any = None
    try:
        while True:
            next(gen)
    except StopIteration as s:
        res = s.value

    assert isinstance(res, processor.ProcessedRasters)
    assert 'domains_ecodistrict' in res.finalized
    assert os.path.exists(res.finalized['domains_ecodistrict'])
