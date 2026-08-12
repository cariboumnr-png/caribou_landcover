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
Spatial grid specification and raster warping operations.
'''

# standard imports
import dataclasses
import os
# third-party imports
import rasterio
import rasterio.enums
import rasterio.shutil
import rasterio.transform
import rasterio.vrt

# Public Dataclasses
@dataclasses.dataclass(frozen=True)
class CanvasSpec:
    '''Target spatial grid specification defining CRS, resolution, and extent.'''
    crs: str
    resolution: float
    bounds: tuple[float, float, float, float] # (xmin, ymin, xmax, ymax)

    @property
    def width(self) -> int:
        '''Calculated canvas width in pixels.'''
        return int(round((self.bounds[2] - self.bounds[0]) / self.resolution))

    @property
    def height(self) -> int:
        '''Calculated canvas height in pixels.'''
        return int(round((self.bounds[3] - self.bounds[1]) / self.resolution))

    @property
    def transform(self) -> rasterio.transform.Affine:
        '''Calculated affine transform for top-left origin grid.'''
        return rasterio.transform.from_origin(
            self.bounds[0], self.bounds[3], self.resolution, self.resolution
        )


# ----- public functions
def create_canvas(
    *,
    reference_raster: str | None = None,
    target_crs: str | None = None,
    target_resolution: float | None = None,
) -> CanvasSpec:
    '''
    Create `CanvasSpec` from a reference raster file or fallback bounds.

    Args:
        reference_raster:
            Optional file path to reference raster dataset.
        target_crs:
            Coordinate Reference System string (e.g. 'EPSG:3161').
        target_resolution:
            Target pixel resolution in meters.

    Returns:
        Configured `CanvasSpec` instance.
    '''
    if reference_raster:
        if not os.path.exists(reference_raster):
            raise FileNotFoundError(
                f'Reference raster file not found: {reference_raster}'
            )

        with rasterio.open(reference_raster) as src:
            crs = target_crs or src.crs.to_string()
            res_val = (
                target_resolution
                if target_resolution is not None
                else src.res[0]
            )
            bounds_tuple = (
                src.bounds.left,
                src.bounds.bottom,
                src.bounds.right,
                src.bounds.top
            )

        return CanvasSpec(
            crs=crs,
            resolution=float(res_val),
            bounds=bounds_tuple
        )

    crs = target_crs or 'EPSG:3161'
    res_val = target_resolution if target_resolution is not None else 20.0
    w_m = 512 * res_val
    bounds_tuple = (500000.0, 600000.0, 500000.0 + w_m, 600000.0 + w_m)
    return CanvasSpec(crs=crs, resolution=float(res_val), bounds=bounds_tuple)


def warp_to_canvas(
    *,
    input_path: str,
    output_path: str,
    canvas: CanvasSpec,
    is_categorical: bool = False,
    resampling_method: str | None = None,
    band_mapping: dict[int, str] | None = None
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
        resample_alg = rasterio.enums.Resampling.nearest
    elif resampling_method == 'cubic':
        resample_alg = rasterio.enums.Resampling.cubic
    else:
        resample_alg = rasterio.enums.Resampling.bilinear

    with rasterio.open(input_path) as src:

        if band_mapping is not None:
            if len(band_mapping) != src.count:
                raise ValueError(
                    f'Expected {src.count} band descriptions, '
                    f'got {len(band_mapping)}'
                )

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
            rasterio.shutil.copy(vrt, output_path, driver='VRT')

    # add band description (name) to the resulting VRT
    if band_mapping is not None:
        with rasterio.open(output_path, 'r+') as dst:
            for band, name in band_mapping.items():
                dst.set_band_description(int(band), name)

    return os.path.abspath(output_path)
