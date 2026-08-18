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
Multi-raster channel composition and nodata mask unification operations.
'''

# standard imports
import os
import xml.etree.ElementTree
# third-party imports
import numpy
import rasterio
import rasterio.crs
import rasterio.shutil
import rasterio.vrt


# ----- public functions
def stack_canonical_raster(
    source_paths: list[str],
    output_path: str
) -> str:
    '''
    Stack multiple rasters into one composite VRT.

    Here input rasters are assumed to have identical CRS, transform etc.

    Args:
        source_paths:
            Ordered list of input raster file paths.
        output_path:
            Destination path for the composite Virtual Raster (.vrt).

    Returns:
        Absolute path to the created composite Virtual Raster file.
    '''
    if not source_paths:
        raise ValueError('source_paths list cannot be empty.')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    source_paths = [os.path.abspath(p) for p in source_paths]
    _build_stacked_vrt_xml(source_paths, output_path)

    return os.path.abspath(output_path)


def unify_nodata_mask(
    input_path: str,
    output_mask_path: str
) -> str:
    '''
    Create a 1-band boolean valid pixel mask across bands.

    Value mapping: 1=valid, 0=nodata.

    Args:
        input_path:
            Path to the multi-band source raster.
        output_mask_path:
            Destination path for the valid-pixel mask raster.

    Returns:
        Absolute path to the created mask raster.
    '''
    out_dir = os.path.dirname(os.path.abspath(output_mask_path))
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(input_path) as src:
        valid_mask = numpy.ones((src.height, src.width), dtype=numpy.uint8)

        for b in range(1, src.count + 1):
            data = src.read(b)
            if src.nodatavals and src.nodatavals[b - 1] is not None:
                nodata_val = src.nodatavals[b - 1]
                if numpy.isnan(nodata_val):
                    valid_mask[numpy.isnan(data)] = 0
                else:
                    valid_mask[data == nodata_val] = 0

        meta = src.meta.copy()
        meta.update({
            'count': 1,
            'dtype': 'uint8',
            'nodata': 0,
            'compress': 'deflate'
        })
        if output_mask_path.endswith('.vrt'):
            meta['driver'] = 'GTiff'
            raw_tif = output_mask_path[:-4] + '_mask.tif'
            with rasterio.open(raw_tif, 'w', **meta) as dst:
                dst.write(valid_mask, 1)

            with rasterio.open(raw_tif) as mask_src:
                with rasterio.vrt.WarpedVRT(mask_src) as vrt:
                    rasterio.shutil.copy(vrt, output_mask_path, driver='VRT')
        else:
            meta['driver'] = 'GTiff'
            with rasterio.open(output_mask_path, 'w', **meta) as dst:
                dst.write(valid_mask, 1)

    return os.path.abspath(output_mask_path)


def validate_domain_raster_index(
    input_path: str,
    min_allowed: int = 1
) -> None:
    '''
    Validate that a domain raster contains 1-based indices.

    Args:
        input_path: Path to the input domain raster file.
        min_allowed: Minimum allowed index value (default: 1).

    Raises:
        ValueError: If valid pixel values contain any values < min_allowed.
    '''
    with rasterio.open(input_path) as src:
        data = src.read(1)
        nodata = src.nodata
        valid_data = data[data != nodata] if nodata is not None else data
        if valid_data.size > 0 and int(valid_data.min()) < min_allowed:
            raise ValueError(
                f'Domain raster [{input_path}] contains index values '
                f'< {min_allowed} (minimum found: {valid_data.min()}). '
                'Categorical domain rasters must use 1-based indexing.'
            )


def add_band_description_to_vrt(
    vrt_fpath: str,
    band_mapping: dict[int, str]
):
    '''Simple helper to add band description to a `.vrt` raster file.'''
    with rasterio.open(vrt_fpath, 'r+') as vrt:
        if len(band_mapping) != vrt.count:
            raise ValueError(
                f'Expected {vrt.count} band descriptions, '
                f'got {len(band_mapping)}'
            )
        for band, name in band_mapping.items():
            vrt.set_band_description(int(band), name)


def add_tag_to_vrt(vrt_fpath: str, **kwargs):
    '''Simple helper to add metadata to a `.vrt` raster file.'''
    with rasterio.open(vrt_fpath, 'r+') as vrt:
        vrt.update_tags(**kwargs)


# ----- private helpers
def _build_stacked_vrt_xml(
    source_paths: list[str],
    output_path: str,
) -> None:
    '''Build a VRT by stacking bands from multiple source rasters.'''
    if not source_paths:
        raise ValueError('source_paths must not be empty')

    with rasterio.open(source_paths[0]) as src:
        width = src.width
        height = src.height
        crs = src.crs
        crs_wkt = crs.to_wkt()
        transform_txt = (
            f'{src.transform.c}, {src.transform.a}, {src.transform.b}, '
            f'{src.transform.f}, {src.transform.d}, {src.transform.e}'
        )

    root = xml.etree.ElementTree.Element(
        'VRTDataset',
        rasterXSize=str(width),
        rasterYSize=str(height),
    )

    xml.etree.ElementTree.SubElement(root, 'SRS').text = crs_wkt
    xml.etree.ElementTree.SubElement(
        root,
        'GeoTransform',
    ).text = transform_txt

    band_idx = 1

    for path in source_paths:
        with rasterio.open(path) as src:
            if src.width != width or src.height != height:
                raise ValueError(
                    f'Source raster has different dimensions: {path} '
                    f'({src.width}x{src.height} != {width}x{height})'
                )

            if src.crs != crs:
                raise ValueError(
                    f'Source raster has a different CRS: {path}'
                )

            # Dataset-level metadata from the source.
            dataset_tags = src.tags()

            for b in range(1, src.count + 1):
                dtype = _gdal_dtype_name(src.dtypes[b - 1])

                band_node = xml.etree.ElementTree.SubElement(
                    root,
                    'VRTRasterBand',
                    dataType=dtype,
                    band=str(band_idx),
                )

                # -------------------------------------------------------------
                # Description
                # -------------------------------------------------------------
                band_name = src.descriptions[b - 1]

                if band_name and band_name.strip():
                    band_name = band_name.strip()
                else:
                    band_name = f'band_{band_idx}'

                xml.etree.ElementTree.SubElement(
                    band_node,
                    'Description',
                ).text = band_name

                # -------------------------------------------------------------
                # Metadata
                #
                # Start with dataset-level tags, then overlay band-level tags.
                # This preserves the behavior of the original resolver:
                #
                #     src.tags(b) or src.tags()
                #
                # while also preserving both types of metadata.
                # -------------------------------------------------------------
                band_tags = src.tags(b)

                tags = {
                    **dataset_tags,
                    **band_tags,
                }

                if tags:
                    metadata_node = xml.etree.ElementTree.SubElement(
                        band_node,
                        'Metadata',
                    )

                    for key, value in tags.items():
                        xml.etree.ElementTree.SubElement(
                            metadata_node,
                            'MDI',
                            key=str(key),
                        ).text = str(value)

                # -------------------------------------------------------------
                # NoData
                # -------------------------------------------------------------
                nodata_val = (
                    src.nodatavals[b - 1]
                    if src.nodatavals
                    and src.nodatavals[b - 1] is not None
                    else src.nodata
                )

                if nodata_val is not None:
                    xml.etree.ElementTree.SubElement(
                        band_node,
                        'NoDataValue',
                    ).text = str(nodata_val)

                # -------------------------------------------------------------
                # Source
                # -------------------------------------------------------------
                source_node = xml.etree.ElementTree.SubElement(
                    band_node,
                    'SimpleSource',
                )

                xml.etree.ElementTree.SubElement(
                    source_node,
                    'SourceFilename',
                    relativeToVRT='0',
                ).text = path

                xml.etree.ElementTree.SubElement(
                    source_node,
                    'SourceBand',
                ).text = str(b)

                xml.etree.ElementTree.SubElement(
                    source_node,
                    'SourceProperties',
                    RasterXSize=str(src.width),
                    RasterYSize=str(src.height),
                    DataType=dtype,
                )

                band_idx += 1

    xml.etree.ElementTree.indent(root, space='  ', level=0)

    xml.etree.ElementTree.ElementTree(root).write(
        output_path,
        encoding='utf-8',
        xml_declaration=True,
    )


def _gdal_dtype_name(dtype) -> str:
    gdal_dtype_map = {
        'uint8': 'Byte',
        'int8': 'Int8',
        'uint16': 'UInt16',
        'int16': 'Int16',
        'uint32': 'UInt32',
        'int32': 'Int32',
        'float32': 'Float32',
        'float64': 'Float64',
    }
    s = str(dtype).lower()
    return gdal_dtype_map.get(s, s.capitalize())
