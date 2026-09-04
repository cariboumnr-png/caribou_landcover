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
import dataclasses
import os
import typing
# third-party imports
import rasterio
# local imports
import landseg.geopipe.grid as grid
import landseg.geopipe.harmonize.manifest as manifest
import landseg.geopipe.harmonize.processor as processor


@dataclasses.dataclass
class _GridParams:
    ref_fpath: str
    crs_string: str = 'EPSG:3161'
    tile_size: tuple[int, int] = (16, 16)
    tile_stride: tuple[int, int] = (8, 8)
    origin: tuple[float, float] | None = None
    pixel_size: tuple[float, float] | None = None
    extent_in_crs_units: tuple[float, float] | None = None


# ----- `harmonize_sources` tests
def test_harmonize_sources_features_and_labels(
    tmp_path, dummy_geotiff_factory
):
    '''
    Given: Two feature rasters and one label raster.
    When: `harmonize_sources` is executed.
    Then: Warp rasters to grid and stack multiple features into one VRT.
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

    grid_params = _GridParams(ref_fpath=str(ref_path))
    world_grid = grid.build_grid('ref', grid_params)

    compiled: dict[str, manifest.ManifestEntry] = {
        str(s2_path): {
            'name': 's2',
            'path': str(s2_path),
            'category': 'features',
            'band_mapping': {1: 'blue', 2: 'green', 3: 'red'},
            'categorical_specs': None,
            'schemes': None,
        },
        str(dem_path): {
            'name': 'dem',
            'path': str(dem_path),
            'category': 'features',
            'band_mapping': {1: 'elevation'},
            'categorical_specs': None,
            'schemes': None,
        },
        str(lbl_path): {
            'name': 'landcover',
            'path': str(lbl_path),
            'category': 'labels',
            'band_mapping': {1: 'landcover'},
            'categorical_specs': {
                'index_base': 1,
                'num_cls': 2,
                'ignore_cls': [255],
            },
            'schemes': None,
        },
    }

    out_dir = str(tmp_path / 'harmonized')
    os.makedirs(out_dir, exist_ok=True)

    gen = processor.harmonize_sources(
        compiled,
        out_dir,
        world_grid,
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


def test_harmonize_sources_domains(tmp_path, dummy_geotiff_factory):
    '''
    Given: A domain categorical raster.
    When: `harmonize_sources` is executed.
    Then: Fast-track domain raster into finalized without stacking.
    '''
    ref_path = dummy_geotiff_factory(
        filename='ref.tif', width=16, height=16, bands=1
    )
    dom_path = dummy_geotiff_factory(
        filename='eco.tif', width=16, height=16, bands=1
    )

    grid_params = _GridParams(ref_fpath=str(ref_path))
    world_grid = grid.build_grid('ref', grid_params)

    compiled: dict[str, manifest.ManifestEntry] = {
        str(dom_path): {
            'name': 'ecodistrict',
            'path': str(dom_path),
            'category': 'domains',
            'band_mapping': {1: 'ecodistrict'},
            'categorical_specs': None,
            'schemes': None,
        },
    }

    out_dir = str(tmp_path / 'harmonized')
    os.makedirs(out_dir, exist_ok=True)

    gen = processor.harmonize_sources(
        compiled,
        out_dir,
        world_grid,
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
