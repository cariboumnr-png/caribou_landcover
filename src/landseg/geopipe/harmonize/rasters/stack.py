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
import typing
import xml.etree.ElementTree
# third-party imports
import rasterio


# ----- public functions
def stack_rasters(
    features_fapths: list[str],
    labels_fapths: list[str],
    output_dir: str,
) -> typing.Generator[str, None, dict[str, str]]:
    '''Stack feature and label rasters if applicable.'''

    def _out_path(tag: str) -> str:
        return os.path.join(output_dir, f'harmonized_{tag}_STACKED.vrt')

    stacked: dict[str, str] = {}
    yield 'Stacking rasters if applicable'

    # features
    n = len(features_fapths)
    if n == 0:
        pass
    elif n == 1:
        stacked.update({'features': features_fapths[0]})
    else:
        out_path = _out_path('features')
        _composite_vrt(features_fapths, out_path)
        stacked.update({'features': out_path})
        yield f'Feature rasters stacked to {out_path} (n={n})'

    # labels
    n = len(labels_fapths)
    if n == 0:
        pass
    elif n == 1:
        stacked.update({'labels': labels_fapths[0]})
    else:
        out_path = _out_path('labels')
        _composite_vrt(labels_fapths, out_path)
        stacked.update({'labels': out_path})
        yield f'Label rasters stacked to {out_path} (n={n})'

    return stacked


# ----- private helpers
def _composite_vrt(
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
