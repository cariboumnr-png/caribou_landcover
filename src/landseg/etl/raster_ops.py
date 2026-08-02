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
import rasterio.vrt


# ----- public functions
def stack_canonical_raster(
    source_paths: list[str],
    output_path: str
) -> str:
    '''
    Stack multiple single-band or multi-band rasters into one composite VRT.

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
    Create a 1-band boolean valid pixel mask (1 = valid, 0 = nodata) across bands.

    Args:
        input_path: Path to the multi-band source raster.
        output_mask_path: Destination path for the valid-pixel mask raster.

    Returns:
        Absolute path to the created mask raster.
    '''
    os.makedirs(os.path.dirname(os.path.abspath(output_mask_path)), exist_ok=True)

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
                    vrt_xml = vrt.to_xml()
                    vrt_bytes = (
                        vrt_xml.encode('utf-8')
                        if isinstance(vrt_xml, str)
                        else vrt_xml
                    )
                    with open(output_mask_path, 'wb') as f:
                        f.write(vrt_bytes)
        else:
            meta['driver'] = 'GTiff'
            with rasterio.open(output_mask_path, 'w', **meta) as dst:
                dst.write(valid_mask, 1)

    return os.path.abspath(output_mask_path)


# ----- private functions
def _build_stacked_vrt_xml(source_paths: list[str], output_path: str) -> None:
    '''Fallback manual VRT XML builder for stacking bands.'''
    with rasterio.open(source_paths[0]) as src:
        width, height = src.width, src.height
        crs_wkt = src.crs.to_wkt()
        tr = src.transform
        transform_text = f'{tr.c}, {tr.a}, {tr.b}, {tr.f}, {tr.d}, {tr.e}'

    root = xml.etree.ElementTree.Element(
        'VRTDataset',
        rasterXSize=str(width),
        rasterYSize=str(height)
    )
    xml.etree.ElementTree.SubElement(root, 'SRS').text = crs_wkt
    xml.etree.ElementTree.SubElement(root, 'GeoTransform').text = transform_text

    band_idx = 1
    for path in source_paths:
        with rasterio.open(path) as src:

            for b in range(1, src.count + 1):
                dtype = str(src.dtypes[b - 1]).capitalize()

                band_node = xml.etree.ElementTree.SubElement(
                    root,
                    'VRTRasterBand',
                    dataType=dtype,
                    band=str(band_idx)
                )

                source_node = xml.etree.ElementTree.SubElement(
                    band_node,
                    'SimpleSource'
                )
                xml.etree.ElementTree.SubElement(
                    source_node,
                    'SourceFilename',
                    relativeToVRT='0'
                ).text = path
                xml.etree.ElementTree.SubElement(
                    source_node,
                    'SourceBand'
                ).text = str(b)
                xml.etree.ElementTree.SubElement(
                    source_node,
                    'SourceProperties',
                    RasterXSize=str(width),
                    RasterYSize=str(height),
                    DataType=dtype
                )
                band_idx += 1

    xml.etree.ElementTree.ElementTree(root).write(
        output_path,
        encoding='utf-8',
        xml_declaration=True
    )
