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

'''Unit tests for band stacker and multi-raster composition.'''

# third-party imports
import numpy
import rasterio
# local imports
import landseg.etl.stacker as stacker_mod


# ----- test cases
def test_stack_canonical_raster(dummy_geotiff_factory, tmp_path):
    '''
    Given: A 10-band optical TIFF and a 1-band DEM TIFF aligned in EPSG:3161.
    When: Invoking `stack_canonical_raster`.
    Then: Produces an 11-band composite GeoTIFF.
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

    out_composite = str(tmp_path / 'composite_11band.tif')
    source_paths = [str(opt_path), str(dem_path)]
    stacker_mod.stack_canonical_raster(source_paths, out_composite)

    with rasterio.open(out_composite) as dst:
        assert dst.count == 11
        assert dst.crs.to_string() == 'EPSG:3161'
        assert dst.width == 20
        assert dst.height == 20
