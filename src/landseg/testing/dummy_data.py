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

# pylint: disable=missing-function-docstring

'''
Utility script to generate dummy/mock geospatial datasets (GeoTIFFs)
and corresponding configurations for local pipeline runs and testing.
'''

# standard imports
import dataclasses
import json
import os
import typing
# third-party imports
import numpy
import rasterio
import rasterio.transform
import rasterio.warp

# Public Dataclass
@dataclasses.dataclass
class TIFFConfig:
    '''Container for creating a dummy GeoTIFF file via `rasterio`.'''
    shape: tuple[int, int]
    bands: int
    crs: str
    transform: rasterio.transform.Affine
    dtype: typing.Any

# ------------------------------private dataclass------------------------------
@dataclasses.dataclass
class TIFFPaths:
    '''Container for dummy TIFF file paths.'''
    input_root: str

    @property
    def extent(self) -> str:
        return f'{self.input_root}/extent_reference/sample_extent.tif'

    @property
    def domain_1(self) -> str:
        return f'{self.input_root}/domain_knowledge/sample_domain_1.tif'

    @property
    def domain_2(self) -> str:
        return f'{self.input_root}/domain_knowledge/sample_domain_2.tif'

    @property
    def raw_dev_sentinel2(self) -> str:
        return f'{self.input_root}/raw/sample_dev_sentinel2.tif'

    @property
    def raw_dev_dem(self) -> str:
        return f'{self.input_root}/raw/sample_dev_dem.tif'

    @property
    def raw_dev_landcover(self) -> str:
        return f'{self.input_root}/raw/sample_dev_landcover.tif'

    @property
    def raw_dev_leadspc(self) -> str:
        return f'{self.input_root}/raw/sample_dev_leadspc.tif'

    @property
    def raw_test_sentinel2(self) -> str:
        return f'{self.input_root}/raw/sample_test_sentinel2.tif'

    @property
    def raw_test_dem(self) -> str:
        return f'{self.input_root}/raw/sample_test_dem.tif'

    @property
    def raw_test_landcover(self) -> str:
        return f'{self.input_root}/raw/sample_test_landcover.tif'

    @property
    def raw_test_leadspc(self) -> str:
        return f'{self.input_root}/raw/sample_test_leadspc.tif'

    @property
    def config(self) -> str:
        return f'{self.input_root}/raw/sample_config.json'

    @property
    def all_paths_exist(self) -> bool:
        return all(
            os.path.exists(p) for p in [
                self.extent,
                self.domain_1,
                self.domain_2,
                self.raw_dev_sentinel2,
                self.raw_dev_dem,
                self.raw_dev_landcover,
                self.raw_test_sentinel2,
                self.raw_test_dem,
                self.raw_test_landcover,
                self.config
            ]
        )

# -------------------------------Public  Function------------------------------
def create_dummy_geotiff(
    fpath: str,
    *,
    config: TIFFConfig,
    data_gen_func: typing.Callable[[tuple[int, int], int], numpy.ndarray],
) -> None:
    '''Write a multi-band GeoTIFF with coordinate metadata.

    Args:
        fpath: Output file path.
        shape: Height and width in pixels.
        bands: Number of bands to write.
        crs: Coordinate Reference System string.
        transform: Affine geospatial transform.
        dtype: NumPy datatype of pixels.
        data_gen_func: Callback generating data per band.
    '''

    # make sure output directory exists
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    # set nodata based on dtype
    nodata_val = (
        0 if config.dtype == numpy.uint8
        else 255 if config.dtype == numpy.uint16
        else -9999.0
    )
    # write GeoTIFF
    with rasterio.open(
        fpath,
        'w',
        driver='GTiff',
        height=config.shape[0],
        width=config.shape[1],
        count=config.bands,
        dtype=config.dtype,
        crs=config.crs,
        transform=config.transform,
        nodata=nodata_val,
    ) as dst:
        for b in range(1, config.bands + 1):
            band_data = data_gen_func(config.shape, b)
            dst.write(band_data, b)

def generate_dummy_data(input_root: str = './experiment/input') -> TIFFPaths:
    '''Generate the full dummy dataset under input root.

    Args:
        input_root: Root directory for output files.
    '''

    print('Generating dummy geospatial data for landseg pipeline...')
    # spatial parameters matching configs/user.yaml
    crs = 'EPSG:3161'
    pxs = 10.0
    orig_x = 500000.0
    orig_y = 5000000.0
    width, height = 512, 512 # this gives 4 256*256 tiles per image
    shape = (height, width)

    # combined extent shape covers both dev and test side-by-side
    extent_shape = (height, width * 2)

    # create affine transform for extent reference
    extent_transform = rasterio.transform.from_origin(orig_x, orig_y, pxs, pxs)

    # TIFF file paths
    paths = TIFFPaths(input_root=input_root)

    # create extent reference (single band constant value on the wide extent)
    print(f'Creating extent reference: {paths.extent}')
    create_dummy_geotiff(
        paths.extent,
        config=TIFFConfig(
            shape=extent_shape,
            bands=1,
            crs=crs,
            transform=extent_transform,
            dtype=numpy.uint8,
        ),
        data_gen_func=lambda s, b: numpy.ones(s, dtype=numpy.uint8),
    )

    # domain knowledge layers (covering the wide extent)
    print(f'Creating domain knowledge 1: {paths.domain_1}')
    create_dummy_geotiff(
        paths.domain_1,
        config=TIFFConfig(
            shape=extent_shape,
            bands=1,
            crs=crs,
            transform=extent_transform,
            dtype=numpy.uint8,
        ),
        data_gen_func=lambda s, b: numpy.random.randint(
            1, 5, size=s, dtype=numpy.uint8
        ),
    )

    print(f'Creating domain knowledge 2: {paths.domain_2}')
    create_dummy_geotiff(
        paths.domain_2,
        config=TIFFConfig(
            shape=extent_shape,
            bands=1,
            crs=crs,
            transform=extent_transform,
            dtype=numpy.uint8,
        ),
        data_gen_func=lambda s, b: numpy.random.randint(
            1, 10, size=s, dtype=numpy.uint8
        ),
    )

    # raw input rasters in a different CRS  to test reprojection
    utm_crs = 'EPSG:2958' # e.g., EPSG:2958
    # transform target canvas origin (500000, 5000000 in EPSG:3161) to source
    dev_res = rasterio.warp.transform(
        'EPSG:3161', utm_crs, [500000.0], [5000000.0]
    )
    utm_dev_transform = rasterio.transform.from_origin(
        dev_res[0][0], dev_res[1][0], 10.0, 10.0
    )

    test_res = rasterio.warp.transform(
        'EPSG:3161', utm_crs, [500000.0 + (width * 10.0)], [5000000.0]
    )
    utm_test_transform = rasterio.transform.from_origin(
        test_res[0][0], test_res[1][0], 10.0, 10.0
    )

    print(f'Creating raw dev Sentinel-2: {paths.raw_dev_sentinel2}')
    create_dummy_geotiff(
        paths.raw_dev_sentinel2,
        config=TIFFConfig(
            shape=shape,
            bands=10,
            crs=utm_crs,
            transform=utm_dev_transform,
            dtype=numpy.uint16,
        ),
        data_gen_func=lambda s, b: numpy.random.randint(
            100, 3000, size=s, dtype=numpy.uint16
        ),
    )

    print(f'Creating raw dev DEM: {paths.raw_dev_dem}')
    create_dummy_geotiff(
        paths.raw_dev_dem,
        config=TIFFConfig(
            shape=shape,
            bands=1,
            crs=utm_crs,
            transform=utm_dev_transform,
            dtype=numpy.float32,
        ),
        data_gen_func=lambda s, _: _gen_image_data(s, 1),
    )

    print(f'Creating raw dev Landcover Label: {paths.raw_dev_landcover}')
    create_dummy_geotiff(
        paths.raw_dev_landcover,
        config=TIFFConfig(
            shape=shape,
            bands=1,
            crs=utm_crs,
            transform=utm_dev_transform,
            dtype=numpy.uint8,
        ),
        data_gen_func=_gen_label_data,
    )

    print(f'Creating raw dev Leadspc Label: {paths.raw_dev_leadspc}')
    create_dummy_geotiff(
        paths.raw_dev_leadspc,
        config=TIFFConfig(
            shape=shape,
            bands=1,
            crs=utm_crs,
            transform=utm_dev_transform,
            dtype=numpy.uint8,
        ),
        data_gen_func=_gen_label_data,
    )

    print(f'Creating raw test Sentinel-2: {paths.raw_test_sentinel2}')
    create_dummy_geotiff(
        paths.raw_test_sentinel2,
        config=TIFFConfig(
            shape=shape,
            bands=10,
            crs=utm_crs,
            transform=utm_test_transform,
            dtype=numpy.uint16,
        ),
        data_gen_func=lambda s, b: numpy.random.randint(
            100, 3000, size=s, dtype=numpy.uint16
        ),
    )

    print(f'Creating raw test DEM: {paths.raw_test_dem}')
    create_dummy_geotiff(
        paths.raw_test_dem,
        config=TIFFConfig(
            shape=shape,
            bands=1,
            crs=utm_crs,
            transform=utm_test_transform,
            dtype=numpy.float32,
        ),
        data_gen_func=lambda s, _: _gen_image_data(s, 1),
    )

    print(f'Creating raw test Landcover Label: {paths.raw_test_landcover}')
    create_dummy_geotiff(
        paths.raw_test_landcover,
        config=TIFFConfig(
            shape=shape,
            bands=1,
            crs=utm_crs,
            transform=utm_test_transform,
            dtype=numpy.uint8,
        ),
        data_gen_func=_gen_label_data,
    )

    print(f'Creating raw test Leadspc Label: {paths.raw_test_leadspc}')
    create_dummy_geotiff(
        paths.raw_test_leadspc,
        config=TIFFConfig(
            shape=shape,
            bands=1,
            crs=utm_crs,
            transform=utm_test_transform,
            dtype=numpy.uint8,
        ),
        data_gen_func=_gen_label_data,
    )

    # dataset config JSON
    print(f'Creating dataset configuration: {paths.config}')
    data_config = {
        'data_source_dir': f'{paths.input_root}/raw',
        'band_mapping': {
            os.path.basename(paths.raw_dev_sentinel2): {
                1: 'blue',
                2: 'green',
                3: 'red',
                4: 'red_edge1',
                5: 'red_edge2',
                6: 'red_edge3',
                7: 'nir',
                8: 'narrow_nir',
                9: 'swir1',
                10: 'swir2',
            },
            os.path.basename(paths.raw_dev_dem): {
                1: 'dem'
            },
            os.path.basename(paths.raw_dev_landcover): {
                1: 'land_cover'
            },
            os.path.basename(paths.raw_dev_leadspc): {
                1: 'lead_species'
            },
            os.path.basename(paths.raw_test_sentinel2): {
                1: 'blue',
                2: 'green',
                3: 'red',
                4: 'red_edge1',
                5: 'red_edge2',
                6: 'red_edge3',
                7: 'nir',
                8: 'narrow_nir',
                9: 'swir1',
                10: 'swir2',
            },
            os.path.basename(paths.raw_test_dem): {
                1: 'dem'
            },
            os.path.basename(paths.raw_test_landcover): {
                1: 'land_cover'
            },
            os.path.basename(paths.raw_test_leadspc): {
                1: 'lead_species'
            }
        },
        'label_specifications': {
            os.path.basename(paths.raw_dev_landcover): {
                'num_cls': 2,
                'ignore_cls': [255]
            },
            os.path.basename(paths.raw_dev_leadspc): {
                'num_cls': 2,
                'ignore_cls': [255]
            },
            os.path.basename(paths.raw_test_landcover): {
                'num_cls': 2,
                'ignore_cls': [255]
            },
            os.path.basename(paths.raw_test_leadspc): {
                'num_cls': 2,
                'ignore_cls': [255]
            }
        },
    }
    os.makedirs(os.path.dirname(paths.config), exist_ok=True)
    with open(paths.config, 'w', encoding='utf-8') as f:
        json.dump(data_config, f, indent=4)

    print('\nDummy data generation completed successfully!')
    return paths

# -------------------------------Private Functions-----------------------------
def _gen_image_data(shape: tuple[int, int], band_idx: int) -> numpy.ndarray:
    '''Generate dummy image band or terrain elevation data.'''

    # first band as DEM (1-based)
    if band_idx == 1:
        # generates a gradient representing terrain elevation
        x = numpy.linspace(100.0, 300.0, shape[1])
        return numpy.tile(x, (shape[0], 1)).astype(numpy.float32)
    # random mock values for visual image bands
    return numpy.random.randint(
        100, 1000, size=shape, dtype=numpy.uint16
    )

def _gen_label_data(shape: tuple[int, int], _: int) -> numpy.ndarray:
    '''Generate dummy label data with ignore index.'''

    labels = numpy.random.choice([0, 1, 255], size=shape, p=[0.45, 0.45, 0.10])
    return labels.astype(numpy.uint8)
