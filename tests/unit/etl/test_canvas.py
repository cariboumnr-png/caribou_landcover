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

'''Unit tests for ETL `CanvasSpec` spatial geometry resolution.'''

# third-party imports
import numpy
# local imports
import landseg.etl.canvas as canvas_mod


# ----- test cases
def test_canvas_spec_dimensions():
    '''
    Given: A `CanvasSpec` defined with EPSG:3161 and 20m resolution.
    When: Accessing calculated width and height properties.
    Then: Correct pixel dimensions are calculated.
    '''
    spec = canvas_mod.CanvasSpec(
        crs='EPSG:3161',
        resolution=20.0,
        bounds=(500000.0, 600000.0, 510000.0, 610000.0) # 10,000m / 20m = 500px
    )
    assert spec.width == 500
    assert spec.height == 500
    assert spec.crs == 'EPSG:3161'


def test_from_reference_raster(dummy_geotiff_factory):
    '''
    Given: A reference GeoTIFF file in EPSG:3161.
    When: Constructing a `CanvasSpec` via from_reference_raster.
    Then: Canvas properties match the reference raster.
    '''
    ref_path = dummy_geotiff_factory(
        filename='ref.tif',
        width=100,
        height=100,
        bands=1,
        dtype=numpy.float32
    )
    spec = canvas_mod.from_reference_raster(str(ref_path))
    assert spec.crs == 'EPSG:3161'
    assert spec.resolution == 20.0
    assert spec.width == 100
    assert spec.height == 100
