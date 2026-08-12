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

# pylint: disable=protected-access

'''Unit tests for spatial geometry definition and raster warping.'''

# third-party imports
import numpy
import rasterio
# local imports
import landseg.geopipe.harmonize.spatial as spatial


# ----- test cases
def test_canvas_spec_dimensions():
    '''
    Given: A `CanvasSpec` defined with EPSG:3161 and 20m resolution.
    When: Accessing calculated width and height properties.
    Then: Correct pixel dimensions are calculated.
    '''
    spec = spatial.CanvasSpec(
        crs='EPSG:3161',
        resolution=20.0,
        bounds=(500000.0, 600000.0, 510000.0, 610000.0) # 10,000m / 20m = 500px
    )
    assert spec.width == 500
    assert spec.height == 500
    assert spec.crs == 'EPSG:3161'


def test_create_canvas_fallback_bounds():
    '''
    Given: Target CRS and resolution without a reference raster.
    When: Invoking `create_canvas`.
    Then: Constructs `CanvasSpec` using default spatial bounds.
    '''
    spec = spatial.create_canvas(
        target_crs='EPSG:3161',
        target_resolution=20.0
    )
    assert spec.crs == 'EPSG:3161'
    assert spec.resolution == 20.0
    assert spec.width == 512
    assert spec.height == 512


def test_create_canvas_with_reference_raster(dummy_geotiff_factory):
    '''
    Given: A reference GeoTIFF file on disk.
    When: Invoking `create_canvas` passing reference_raster path.
    Then: Constructs `CanvasSpec` with properties from reference raster.
    '''
    ref_path = dummy_geotiff_factory(
        filename='ref_canvas.tif',
        width=100,
        height=100,
        bands=1,
        dtype=numpy.float32
    )
    spec = spatial.create_canvas(
        target_crs='EPSG:3161',
        target_resolution=20.0,
        reference_raster=str(ref_path)
    )
    assert spec.crs == 'EPSG:3161'
    assert spec.resolution == 20.0
    assert spec.width == 100
    assert spec.height == 100


def test_warp_sentinel2_to_lambert(dummy_geotiff_factory, tmp_path):
    '''
    Given: A 10-band Sentinel-2 composite in UTM Zone 17N (EPSG:32617).
    When: Warping to target canvas in EPSG:3161 at 20m resolution.
    Then: Output Virtual Raster (.vrt) has 10 bands in EPSG:3161 with exact canvas dimensions.
    '''
    s2_path = dummy_geotiff_factory(
        filename='sentinel2_10band_utm17.tif',
        width=50,
        height=50,
        bands=10,
        crs='EPSG:32617',
        dtype=numpy.uint16
    )
    canvas = spatial.CanvasSpec(
        crs='EPSG:3161',
        resolution=20.0,
        bounds=(500000.0, 600000.0, 501000.0, 601000.0) # 50px x 50px
    )

    out_path = str(tmp_path / 'warped_s2.vrt')
    spatial.warp_to_canvas(
        input_path=str(s2_path),
        output_path=out_path,
        canvas=canvas,
        is_categorical=False
    )

    with rasterio.open(out_path) as dst:
        assert dst.crs.to_string() == 'EPSG:3161'
        assert dst.count == 10
        assert dst.width == 50
        assert dst.height == 50


def test_warp_categorical_labels_preserves_integer_values(dummy_geotiff_factory, tmp_path):
    '''
    Given: A 5-class integer landcover label mask in EPSG:32617.
    When: Warping using nearest-neighbor categorical resampling.
    Then: Output Virtual Raster contains strictly integer class IDs without interpolation.
    '''
    def _label_gen(shape, _):
        return numpy.random.randint(1, 6, shape).astype(numpy.uint8)

    label_path = dummy_geotiff_factory(
        filename='labels_5class_utm17.tif',
        width=50,
        height=50,
        bands=1,
        crs='EPSG:32617',
        dtype=numpy.uint8,
        data_gen_func=_label_gen
    )
    canvas = spatial.CanvasSpec(
        crs='EPSG:3161',
        resolution=20.0,
        bounds=(500000.0, 600000.0, 501000.0, 601000.0)
    )

    out_path = str(tmp_path / 'warped_labels.vrt')
    spatial.warp_to_canvas(
        input_path=str(label_path),
        output_path=out_path,
        canvas=canvas,
        is_categorical=True
    )

    with rasterio.open(out_path) as dst:
        assert dst.crs.to_string() == 'EPSG:3161'
        assert dst.count == 1
        data = dst.read(1)
        assert data.dtype == numpy.uint8
