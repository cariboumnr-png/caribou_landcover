'''
Public Class:
    RasterBlocks(): Generates and stores blocks from two input rasters.

Class keys and values:

    # input rasters information
    ['proj']: EPSG:26918
    ['ras_nodata']: 0
    ['res']: 1
    ['transform']: Affine(1.0, 0.0, 500000.0, ...)
    ['extent']: (Left:..., Bottom:..., Right:..., Top:...)

    # processing blocks from the input rasters
    ['blocks']: {'row_0_col_0': [0, 0, 1000, 1000]...}
    ['blocks_both_valid']: {'row_0, col_0': [...] ...}
    ['blocks_ras1_empty']: {'row_0, col_1': [...] ...} *
    ['blocks_ras2_empty']: {'row_1, col_0': [...] ...} *
    ['blocks_both_empty']: {'row_1, col_1': [...] ...} *
    ['blocks_some_empty']: [...] # list of the above three*
'''
# standard imports
import dataclasses
import re
# third-party imports
import numpy
import rasterio
import rasterio.coords
import rasterio.io
import rasterio.transform
import rasterio.windows
# local imports
import utils.funcs
import utils.logger

# initialize logger for the script
log = utils.logger.Logger(name='tiler')

@dataclasses.dataclass
class BlockName:
    '''Simple class to define block name from column and row number.'''
    col: int
    row: int
    colrow: tuple[int, int] = dataclasses.field(init=False)
    name: str = dataclasses.field(init=False)

    def __post_init__(self):
        self.colrow = (self.col, self.row)
        self.name = f'col_{self.col}_row_{self.row}'

class RasterPairedBlocks(dict):
    '''
    A class to check two input rasters and prepare processing blocks.

    Attributes:
        raster (tuple): A tuple of two file paths to the input rasters.
        block_size (int): The block size in meters.
        ras_1 (rasterio.io.DatasetReader): First raster object opened
            with `rasterio.open()`.
        ras_2 (rasterio.io.DatasetReader): Second raster object opened
            with `rasterio.open()`.
    '''

    def __init__(self, rasters: tuple, block_size: int):
        '''
        Creates processing blocks from the input rasters.

        Args:
            rasters (tuple): A tuple of two file paths to the input
                rasters.
            block_size (int): The pixel size of the squared blocks.
        '''

        # assign attributes
        self.rasters = rasters
        self.block_size = block_size
        self.ras_1: rasterio.io.DatasetReader | None = None
        self.ras_2: rasterio.io.DatasetReader | None = None

    # public methods
    def stage(self, **kwargs) -> None:
        '''Stage the class instance for further raster calculations.'''

        # open the input rasters
        with rasterio.open(self.rasters[0]) as ras_1, \
             rasterio.open(self.rasters[1]) as ras_2:
            # ras_1 and ras_2 passed as class attr for followed usage
            self.ras_1 = ras_1
            self.ras_2 = ras_2
            # check rasters validity and then create work blocks
            utils.funcs.timed_run(self._check_input_raster, args=None,
                                     kwargs=kwargs, log=log)
            utils.funcs.timed_run(self._create_work_blocks, args=None,
                                     kwargs=kwargs, log=log)

        # clears raster reader attributes so the class instance can be copied
        self.ras_1 = None
        self.ras_2 = None

    def reset(self) -> None:
        '''Placeholder public method.'''

    # methods for checking raster metadata
    def _check_input_raster(self, **kwargs) -> None:
        '''A glue-up method to check if input rasters are valid.'''

        check_proj = kwargs.get('check_proj', True)
        check_pixels = kwargs.get('check_pixels', True)

        # check if both rasters have the same projection system
        if check_proj:
            self.__check_raster_proj()
        # check if both rasters have the same squared pixels
        if check_pixels:
            self.__check_raster_pixels()

    def __check_raster_proj(self) -> None:
        '''
        Check if the input rasters have the same projection.

        Raises:
            ValueError: If the input rasters do not have the same
                projection system.
        ----------------------------------------------------------------
        Assigns value to `self['proj']`.
        '''

        # type guarding assertion
        assert self.ras_1 is not None
        assert self.ras_2 is not None

        # get projection names, raster.crs might return differently
        try:
            crs_1 = self.ras_1.crs.to_string().split('"')[1]
            crs_2 = self.ras_2.crs.to_string().split('"')[1]
        except IndexError:
            crs_1 = self.ras_1.crs
            crs_2 = self.ras_2.crs

        # check if the projection systems are the same
        if crs_1 != crs_2:
            log.log('ERROR', f' | Projection systems do not match:'
                             f'\nRaster_1: {crs_1}'
                             f'\nRaster_2: {crs_2}')
            raise ValueError(' | Both rasters must have the same projection')

        # if both projections are the same
        self['proj'] = crs_1
        log.log('DEBUG', ' | Checking if the same projection system ... OK')

    def __check_raster_pixels(self) -> None:
        '''
        Check if the input rasters have the same squared pixels.

        This function uses affine transform get pixel resolution in
        both x and y directions and checks:

        * If both rasters have the same pixel size in both directions.
        * If the pixels are squared (pixel size in `x` == that in `y`).

        Raises:
            ValueError: If the pixel sizes are different or the pixels
                are not squared.
        ----------------------------------------------------------------
        Assigns value to `self['res']`.
        '''

        # type guarding assertion
        assert self.ras_1 is not None
        assert self.ras_2 is not None

        # get the transform (Affine matrix) from the metadata
        transform_1 = self.ras_1.transform
        transform_2 = self.ras_2.transform

        # transform[0]: pixel size in the x direction (horizontal).
        # transform[4]: pixel size in the y direction (vertical).
        # Note: y is typically negative as the axis increases downward.
        x1, y1 = transform_1[0], -transform_1[4]
        x2, y2 = transform_2[0], -transform_2[4]

        # check if the pixels are squared (x == y)
        if x1 != y1 or x2 != y2:
            log.log('ERROR', f' | Input rasters do not have squared pixels: '
                             f'Raster1: ({x1}, {y1}), Raster2: ({x2}, {y2})')
            raise ValueError(' | Input rasters must have squared pixels')

        # check if the pixel sizes match
        if (x1, y1) != (x2, y2):
            log.log('ERROR', f' | Input rasters have different pixel sizes: '
                             f'Raster1: ({x1}, {y1}), Raster2: ({x2}, {y2})')
            raise ValueError(' | Input rasters must have the same pixel sizes')

        # assign value and log out
        self['res'] = x1
        log.log('DEBUG', ' | Checking if the same squared pixels ... OK')

    # methods for creating work blocks
    def _create_work_blocks(self, **kwargs) -> None:
        '''A glue-up method to create working raster blocks.'''

        overlap = kwargs.get('overlap', 0)

        # get the overlapping extent from the input rasters
        self.__get_overlap_extent()
        # get the new Affine transform needed for writing output raster
        self.__get_new_transform()
        # tile the raster into blocks
        self.__extent_to_blocks(overlap=overlap)
        # check and mark for blocks where rasters are empty
        self.__check_empty_blocks()

    def __get_overlap_extent(self) -> None:
        '''
        Get the overlapping extent of the input rasters.

        The extent is defined by:
        * max of the left bounds.
        * max of the bottom bounds.
        * min of the right bounds.
        * min of the top bounds.

        Raises:
            ValueError: If input rasters have no overlapping extents.
        ----------------------------------------------------------------
        Assigns value to `self['extent']`.
        '''

        # type guarding assertion
        assert self.ras_1 is not None
        assert self.ras_2 is not None

        # get the bounding boxes
        b1 = self.ras_1.bounds
        b2 = self.ras_2.bounds

        # bounds(0-3) correspond to [left, bottom, right, top] side of the box
        lft = max(b1[0], b2[0]) # max of the left bounds
        btm = max(b1[1], b2[1]) # max of the bottom bounds
        rgt = min(b1[2], b2[2]) # min of the right bounds
        top = min(b1[3], b2[3]) # min of the top bounds

        # if the two do not overlop
        if lft >= rgt or btm >= top:
            log.log('ERROR', ' | The input rasters have no overlapping extent')
            raise ValueError(' | Input rasters must have overlapping extents')

        # get the overlapping extent if no error
        bb = rasterio.coords.BoundingBox(lft, btm, rgt, top)
        self['extent'] = bb

        # log out accordingly
        log.log('DEBUG', ' | Calculating the overlapping extent ... OK')
        if b1 == b2:
            log.log('DEBUG',  ' |  - The input rasters have the same extent:')
            log.log('DEBUG', f' |  - [L{b1[0]}|B{b1[1]}|R{b1[2]}|T{b1[3]}]')
        else:
            log.log('DEBUG',  ' |  - The input rasters have different extents')
            log.log('DEBUG', f' |  - 1: [L{b1[0]}|B{b1[1]}|R{b1[2]}|T{b1[3]}]')
            log.log('DEBUG', f' |  - 2: [L{b2[0]}|B{b2[1]}|R{b2[2]}|T{b2[3]}]')
            log.log('DEBUG',  ' |  - Using overlapping extent:')
            log.log('DEBUG', f' |  - [L{bb[0]}|B{bb[1]}|R{bb[2]}|T{bb[3]}]')

    def __get_new_transform(self) -> None:
        '''
        Create Affine transform from the overlapping extent.

        ----------------------------------------------------------------
        Assigns value to `self['transform']`.
        '''

        # get the boundaries and resolution
        left, _, _, top = self['extent']
        res = self['res']
        # create the new transform
        self['transform'] = rasterio.transform.from_origin(left, top, res, res)
        # log out
        log.log('DEBUG', ' | Generating transform for the new extent ... OK')

    def __extent_to_blocks(self, overlap: int) -> None:
        '''
        Divide the overlapping extent to raster blocks.

        This function takes in a `rasterio.coords.BoundingBox` generated
        by `_get_overlap_extent()` and  calculates the number of pixels
        along the x and y axes based on the provided raster resolution
        and the extent. It divides the extent into square blocks of the
        given `block_size`, from top-left to bottom-right. If the extent
        cannot be evenly divided, right and bottom edge blocks will be
        created. For example:

        * A 100x100 pixel extent divided into 20x20 pixel blocks
        results in exactly 25 blocks.
        * A 110x105 pixel extent divided into 20x20 pixel blocks
        resutls in 25 regular 20x20 blocks, 5 right edge blocks of
        20x5 pixels, 5 bottom edge blocks of 10x20 pixels, and 1
        bottom-right corner block of 10x5 pixels.

        The blocks are indexed by the row and col numbers from top-left
        to bottom-right, e.g., the top-left block will be indexed as
        `[row_0, col_0]`.

        Intended to work on projected rasters with meter unit, such as
        an UTM projected raster.

        Args:
            overlap (int): Number of pixels to overlap between adjacent blocks.

        ----------------------------------------------------------------
        Assigns value to `self['all_blocks']`.
        '''

        # get extent dimensions (pixel count) - round to nearest int
        extent_h = round(
            (self['extent'][3] - self['extent'][1]) / self['res']) # T - B
        extent_w = round(
            (self['extent'][2] - self['extent'][0]) / self['res']) # R - L

        # create a dict to store results
        _blks = {}

        # step size is block_size minus overlap
        step = self.block_size - overlap
        if step <= 0:
            log.log('ERROR', 'Overlap must be smaller than block size.')
            raise ValueError('Overlap must be smaller than block size.')

        # iterate through the blocks by row then col
        for i in range(0, extent_h, step):
            for j in range(0, extent_w, step):
                # create a block index from its position
                index = BlockName(j, i).name
                # dynamically adjust window size to stay within bounds
                block_h = min(self.block_size, extent_h - i) # at the last row
                block_w = min(self.block_size, extent_w - j) # at the last col
                # set up the window and update the result dict
                _blks[index] = rasterio.windows.Window(j, i, block_w, block_h) # type: ignore
        # assign blocks to value
        self['all_blocks'] = _blks

        # calculate block related numbers and log out
        log.log('DEBUG', ' | Creating processing blocks ... OK')
        if overlap:
            log.log('DEBUG', f' | The blocks have {overlap} px in overlap')
        # regular block row and col count as quotients
        reg_rows = extent_h // step
        reg_cols = extent_w // step
        # edge block width and height as remainders
        edge_h = extent_h % step
        edge_w = extent_w % step
        # total block row and col count
        all_rows = reg_rows + 1 if edge_h > 0 else reg_rows
        all_cols = reg_cols + 1 if edge_w > 0 else reg_cols
        # total blocks
        log.log('DEBUG', f' |  - Total: {all_rows}H x {all_cols}W = '
                         f'{all_rows * all_cols} blocks')
        # regular blocks
        log.log('DEBUG', f' |  - with {reg_rows * reg_cols} regular blocks in'
                         f' {self.block_size}H x {self.block_size}W (px^2)')
        # bottom edge blocks
        if edge_h != 0:
            log.log('DEBUG', f' |  - with {reg_cols} bottom row edge blocks in'
                             f' {edge_h}H x {self.block_size}W (px^2)')
        # right edge blocks
        if edge_w != 0:
            log.log('DEBUG', f' |  - with {reg_rows} right col. edge blocks in'
                             f' {self.block_size}H x {edge_w}W (px^2)')
        # bottom right corner block
        if edge_h != 0 and edge_w != 0:
            log.log('DEBUG', f' |  - with 1 bottom right corner block in'
                             f' {edge_w}H x {edge_h}W (px^2)')

    def __check_empty_blocks(self) -> None:
        '''
        Check for blocks where the rasters are of `nodata`.

        This function go through the blocks generated by the function
        `_extent_to_blocks()` and identify:

        * blocks where both rasters are valid.
        * blocks where only raster_1 is empty.
        * blocks where only raster_2 is empty.
        * blocks where both rasters are empty.

        While calculation is needed for the first category, the rest can
        be simply assigned a value of choice, saving overall computing.

        ----------------------------------------------------------------
        Assigns value to `self['blocks_both_valid']`,
        `self['blocks_ras1_empty']`, `self['blocks_ras2_empty']`,
        `self['blocks_both_empty']`, `self['blocks_some_empty']`.
        '''

        # type guarding assertion
        assert self.ras_1 is not None
        assert self.ras_2 is not None

        # create dictionaries
        dict_1 = {} # blocks where both rasters are valid
        dict_2 = {} # blocks where only raster_1 is empty
        dict_3 = {} # blocks where only raster_2 is empty
        dict_4 = {} # blocks where both rasters are empty

        # iterate through blocks
        for idx, block in self['all_blocks'].items():

            # read the rasters at the current block
            raster_data_1 = self.ras_1.read(1, window=block)
            raster_data_2 = self.ras_2.read(1, window=block)

            # both are not empty
            if numpy.any(raster_data_1 != self.ras_1.nodata) and \
                numpy.any(raster_data_2 != self.ras_2.nodata):
                dict_1[idx] = block

            # only raster_1 is empty
            if numpy.all(raster_data_1 == self.ras_1.nodata) and \
                numpy.any(raster_data_2 != self.ras_2.nodata):
                dict_2[idx] = block

            # only raster_2 is empty
            if numpy.any(raster_data_1 != self.ras_1.nodata) and \
                numpy.all(raster_data_2 == self.ras_2.nodata):
                dict_3[idx] = block

            # both are empty
            if numpy.all(raster_data_1 == self.ras_1.nodata) and \
                numpy.all(raster_data_2 == self.ras_2.nodata):
                dict_4[idx] = block

        # assign values
        self['blocks_both_valid'] = dict_1
        self['blocks_ras1_empty'] = dict_2
        self['blocks_ras2_empty'] = dict_3
        self['blocks_both_empty'] = dict_4
        self['blocks_some_empty'] = [dict_2, dict_3, dict_4] # a list of dicts

        # log out
        log.log('DEBUG', ' | Identifying empty blocks for both rasters ... OK')
        # if any kind of empty blocks are found
        if any((dict_2, dict_3, dict_4)):
            log.log('DEBUG', ' |  - Found empty blocks from input rasters')
        else:
            log.log('DEBUG', ' |  - No empty blocks found in input rasters')

def parse_block_name(input_str: str) -> BlockName:
    '''Retrieve block naming info from a given string.'''

    # find pattern from string
    pattern = r'col_(\d+)_row_(\d+)'
    matched = re.search(pattern, input_str)
    # there should be just one match
    if not matched:
        raise ValueError(f'Block naming pattern {pattern} not found')
    # get col and row
    col = int(matched.group(1))
    row = int(matched.group(2))
    # return
    return BlockName(col, row)
