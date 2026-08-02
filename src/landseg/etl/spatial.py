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

'''
Spatial canvas specification and grid reprojection operations.
'''

# standard imports
import dataclasses
import os
# third-party imports
import rasterio
import rasterio.transform
import rasterio.vrt
from rasterio.warp import Resampling


# ----- public types
@dataclasses.dataclass(frozen=True)
class CanvasSpec:
    '''Spatial target canvas definition for raster harmonization.'''
    crs: str
    resolution: float
    bounds: tuple[float, float, float, float] # (minx, miny, maxx, maxy)

    @property
    def width(self) -> int:
        '''Calculate width in pixels.'''
        return int(round((self.bounds[2] - self.bounds[0]) / self.resolution))

    @property
    def height(self) -> int:
        '''Calculate height in pixels.'''
        return int(round((self.bounds[3] - self.bounds[1]) / self.resolution))

    @property
    def transform(self) -> rasterio.Affine:
        '''Calculate affine transform matrix.'''
        return rasterio.transform.from_bounds(
            *self.bounds,
            width=self.width,
            height=self.height
        )


# ----- public functions
def from_reference_raster(
    raster_path: str,
    *,
    target_crs: str | None = None,
    target_resolution: float | None = None
) -> CanvasSpec:
    '''
    Derive `CanvasSpec` from a reference GeoTIFF file.

    Args:
        raster_path:
            Path to the reference GeoTIFF file.
        target_crs:
            Optional target CRS override.
        target_resolution:
            Optional target spatial resolution override in meters.

    Returns:
        Configured `CanvasSpec` instance matching the reference raster.
    '''
    with rasterio.open(raster_path) as src:
        bounds_tuple = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        crs = target_crs if target_crs is not None else src.crs.to_string()
        res_val = target_resolution if target_resolution is not None else src.res[0]

    return CanvasSpec(
        crs=crs,
        resolution=float(res_val),
        bounds=bounds_tuple
    )


def warp_to_canvas(
    *,
    input_path: str,
    output_path: str,
    canvas: CanvasSpec,
    is_categorical: bool = False,
    resampling_method: str | None = None
) -> str:
    '''
    Reproject and snap an input raster to target `CanvasSpec` grid as a VRT.

    Args:
        input_path:
            Path to the raw source GeoTIFF.
        output_path:
            Destination path for the harmonized Virtual Raster (.vrt).
        canvas:
            Configured `CanvasSpec` target.
        is_categorical:
            If True, strictly uses nearest-neighbor resampling.
        resampling_method:
            Optional string override ('nearest', 'bilinear', 'cubic').

    Returns:
        Absolute path to the output harmonized Virtual Raster file.
    '''
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if is_categorical or resampling_method == 'nearest':
        resample_alg = Resampling.nearest
    elif resampling_method == 'cubic':
        resample_alg = Resampling.cubic
    else:
        resample_alg = Resampling.bilinear

    with rasterio.open(input_path) as src:
        nodata_val = (
            src.nodata
            if src.nodata is not None
            else (255 if is_categorical else -9999)
        )
        with rasterio.vrt.WarpedVRT(
            src,
            crs=canvas.crs,
            transform=canvas.transform,
            width=canvas.width,
            height=canvas.height,
            resampling=resample_alg,
            nodata=nodata_val
        ) as vrt:
            vrt_xml = vrt.to_xml()
            vrt_bytes = vrt_xml.encode('utf-8') if isinstance(vrt_xml, str) else vrt_xml

            with open(output_path, 'wb') as f:
                f.write(vrt_bytes)

    return os.path.abspath(output_path)
