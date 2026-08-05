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

'''Unit tests for raster band composition and nodata mask unification.'''

# third-party imports
import numpy
import pytest
import rasterio
# local imports
import landseg.etl.raster_ops as raster_ops


# ----- test cases
def test_stack_canonical_raster(dummy_geotiff_factory, tmp_path):
    '''
    Given: A 10-band optical TIFF and a 1-band DEM TIFF aligned in EPSG:3161.
    When: Invoking `stack_canonical_raster`.
    Then: Produces an 11-band composite Virtual Raster (.vrt).
    '''
    opt_path = dummy_geotiff_factory(
        filename='aligned_opt.tif',
        width=20,
        height=20,
        bands=10,
        dtype=numpy.uint16
    )
    dem_path = dummy_geotiff_factory(
        filename='aligned_dem.tif',
        width=20,
        height=20,
        bands=1,
        dtype=numpy.float32
    )

    out_composite = str(tmp_path / 'composite_11band.vrt')
    source_paths = [str(opt_path), str(dem_path)]
    raster_ops.stack_canonical_raster(source_paths, out_composite)

    with rasterio.open(out_composite) as dst:
        assert dst.count == 11
        assert dst.crs.to_string() == 'EPSG:3161'
        assert dst.width == 20
        assert dst.height == 20


def test_stack_canonical_raster_empty_sources_raises_error(tmp_path):
    '''
    Given: An empty list of source raster paths.
    When: Invoking `stack_canonical_raster`.
    Then: Raises ValueError.
    '''
    out_composite = str(tmp_path / 'empty_composite.vrt')
    with pytest.raises(ValueError, match='source_paths list cannot be empty.'):
        raster_ops.stack_canonical_raster([], out_composite)


def test_unify_nodata_mask(dummy_geotiff_factory, tmp_path):
    '''
    Given: A multi-band composite GeoTIFF with nodata flags.
    When: Invoking `unify_nodata_mask`.
    Then: Creates a 1-band boolean valid pixel mask Virtual Raster (.vrt).
    '''
    comp_path = dummy_geotiff_factory(
        filename='comp_with_nodata.tif',
        width=20,
        height=20,
        bands=3,
        dtype=numpy.uint8
    )

    out_mask = str(tmp_path / 'valid_mask.vrt')
    raster_ops.unify_nodata_mask(str(comp_path), out_mask)

    with rasterio.open(out_mask) as dst:
        assert dst.count == 1
        assert dst.dtypes[0] == 'uint8'
        assert dst.width == 20
        assert dst.height == 20


def test_unify_nodata_mask_gtiff_output(dummy_geotiff_factory, tmp_path):
    '''
    Given: A multi-band composite GeoTIFF.
    When: Invoking `unify_nodata_mask` with a .tif destination path.
    Then: Creates a 1-band GeoTIFF mask file directly.
    '''
    comp_path = dummy_geotiff_factory(
        filename='comp_gtiff.tif',
        width=20,
        height=20,
        bands=2,
        dtype=numpy.uint8
    )

    out_mask = str(tmp_path / 'valid_mask.tif')
    raster_ops.unify_nodata_mask(str(comp_path), out_mask)

    with rasterio.open(out_mask) as dst:
        assert dst.count == 1
        assert dst.driver == 'GTiff'
        assert dst.width == 20
        assert dst.height == 20


def test_validate_domain_raster_index_valid(dummy_geotiff_factory):
    '''
    Given: A domain raster with pixel values >= 1.
    When: Invoking `validate_domain_raster_index`.
    Then: Passes silently.
    '''
    domain_path = dummy_geotiff_factory(
        filename='domain_valid.tif',
        width=10,
        height=10,
        bands=1,
        dtype=numpy.uint8
    )
    # write values 1..10
    with rasterio.open(domain_path, 'r+') as dst:
        data = numpy.ones((1, 10, 10), dtype=numpy.uint8)
        dst.write(data)

    raster_ops.validate_domain_raster_index(str(domain_path), min_allowed=1)


def test_validate_domain_raster_index_invalid(dummy_geotiff_factory):
    '''
    Given: A domain raster with pixel value 0.
    When: Invoking `validate_domain_raster_index`.
    Then: Raises ValueError.
    '''
    domain_path = dummy_geotiff_factory(
        filename='domain_invalid.tif',
        width=10,
        height=10,
        bands=1,
        dtype=numpy.uint8
    )
    # write values including 0 (with nodata set to 255)
    with rasterio.open(domain_path, 'r+') as dst:
        dst.nodata = 255
        data = numpy.zeros((1, 10, 10), dtype=numpy.uint8)
        dst.write(data)

    with pytest.raises(ValueError, match='Categorical domain rasters must use 1-based indexing'):
        raster_ops.validate_domain_raster_index(str(domain_path), min_allowed=1)
