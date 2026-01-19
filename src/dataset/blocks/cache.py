'''Data blocks preperation pipeline.'''

# standard imports
import copy
import os
import pickle
import random
import typing
import warnings
import zipfile
import zlib
# third-party imports
import numpy
import rasterio
import rasterio.io
import rasterio.windows
# local imports
import dataset.blocks
import dataset.blocks.tiler
import utils.funcs
import utils.logger
import utils.multip

log = utils.logger.Logger(name='cache')

def get_block_scheme(
        scheme_fpath: str,
        input_ras: tuple[rasterio.io.DatasetReader, rasterio.io.DatasetReader],
        block_size: int,
        overlap: int,
        rand_test: int=0,
        **kwargs
    ) -> dict[str, rasterio.windows.Window]:
    '''Prepare blocks from the input raster, create new if not exists.'''

    # kwargs
    overwrite = kwargs.get('overwrite', False)

    # read from existing file
    if os.path.exists(scheme_fpath) and not overwrite:
        log.log('INFO', f'Keep existing raster block scheme: {scheme_fpath}')
        with open(scheme_fpath, 'rb') as file:
            blocks_scheme = pickle.load(file)
    # otherwise create if scheme not already exsited
    else:
        log.log('INFO', f'Creating or re-writing scheme: {scheme_fpath}')
        input_blocks = dataset.blocks.tiler.RasterPairedBlocks(
            rasters = input_ras,
            block_size=block_size
        )
        input_blocks.stage(overlap=overlap)
        blocks_scheme = input_blocks['blocks_both_valid']
        log.log('INFO', f'New raster block scheme save at: {scheme_fpath}')
        with open(scheme_fpath, 'wb') as file:
            pickle.dump(blocks_scheme, file)

    # if random test
    if rand_test:
        assert rand_test <= len(blocks_scheme)
        return dict(random.sample(list(blocks_scheme.items()), rand_test))

    # return
    return blocks_scheme

def clean_up_bad_npz(block_dpath: str, to_clean: bool) -> None:
    '''Simple helper to remove bad .npz files at dir.'''

    # if chosen not to clean
    if not to_clean:
        log.log('INFO', 'Skipping checking for bad block .npz files')
        return

    # parallel processing the blocks
    log.log('INFO', f'Checking for bad block .npz files at {block_dpath}')
    fpaths = utils.funcs.get_fpaths_from_dir(block_dpath, '.npz')
    jobs = [(_npz_file_check, (fpath,), {}) for fpath in fpaths]
    results: list[dict] = utils.multip.ParallelExecutor().run(jobs)

    # parse results
    to_remove = []
    for result in results:
        if result.get('remove', None):
            to_remove.append(result['remove'])

    # remove files
    for fpath in to_remove:
        os.remove(fpath)
    log.log('INFO', f'Found and removed {len(to_remove)} bad .npz files')

def _npz_file_check(fpath: str) -> dict[str, str]:
    '''Check if a .npz block file is corrupted.'''

    # pass if the npz file can be loaded properly
    try:
        rb = dataset.blocks.DataBlock()
        rb.load_from_npz(fpath)
        return {'pass': fpath}
    # corrupted/damaged npz file to be removed
    except (zipfile.error, zlib.error):
        return {'remove': fpath}

def create_block_caches(
        scheme_fpath: str,
        blk_dpath: str,
        meta_fpath: str,
        lbl_fpath: str,
        img_fpath: str,
        **kwargs) -> None:
    '''Create cached blocks as npz files.'''

    # kwargs
    overwrite = kwargs.get('overwrite', False)

    #
    scheme = utils.funcs.load_pickle(scheme_fpath)

    # determine blocks to be processed by overwrite flag and file existence
    work_blocks: dict[str, rasterio.windows.Window] = {}
    if overwrite:
        work_blocks = scheme
    else:
        for key in scheme.keys():
            if not os.path.exists(f'{blk_dpath}/block_{key}.npz'):
                work_blocks.update({key: scheme[key]})
    if not work_blocks:
        log.log('INFO', 'No data blocks to be created')
        return
    log.log('INFO', f'{len(work_blocks)} data blocks to be created')

    # read raster and create blocks
    log.log('INFO', 'Generating block cache files from input rasters')
    meta = utils.funcs.load_json(meta_fpath)

    # parallel processing
    jobs = [
        (_block_from_ras, ((lbl_fpath, img_fpath), meta, blk, blk_dpath,), {})
        for blk in work_blocks.items()
    ]
    results = utils.multip.ParallelExecutor().run(jobs)
    custom_warnings = [warning for result in results for warning in result]
    if len(custom_warnings) > 0:
        for warning in custom_warnings:
            log.log('WARNING', f'{warning}')

def _block_from_ras(
        ras_fpaths: tuple[str, str],
        meta: dict[str, typing.Any],
        block: tuple[str, rasterio.windows.Window],
        output_dpath: str
    ) -> list:
    '''Create new a block from input rasters'''

    # deep copy a meata dict to avoid cross-contanimation
    meta = copy.deepcopy(meta)

    # customize warnings context
    warnings.showwarning = _custom_warning_handler
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=RuntimeWarning)
        with rasterio.open(ras_fpaths[0]) as lbl, \
            rasterio.open(ras_fpaths[1]) as img:

            # add entries to meta
            meta['label_nodata'] = lbl.nodata
            meta['image_nodata'] = img.nodata
            meta['block_name'] = block[0]
            meta['block_shape'] = [block[1].width, block[1].height]

            # read original label and image arrays
            lbl_arr = lbl.read(window=block[1]) # (1, 256, 256)
            img_arr = img.read(window=block[1]) # (7, 256, 256)

            # get padded dem array
            dem_band = meta['band_assignment']['dem'] + 1 # rasterio is 1-based
            dem_pad = meta['dem_pad']
            dem_padded = _get_padded_dem(img, block[1], dem_band, dem_pad)

            # init and populate RasterBlock, meta as the kwargs
            raster_block = dataset.blocks.DataBlock()
            raster_block.load_from_rasters(lbl_arr, img_arr, dem_padded, meta)

            # write to target npz file
            raster_block.save_npz(f'{output_dpath}/block_{block[0]}.npz')

        # captures runtime warning if any
        return w

def _custom_warning_handler(
        msg: str,
        fname: str,
        lnum: int
    ) -> None:
    '''custom handling runtime warnings.'''
    log.log('WARNING', f'Warning caught: {msg} (File: {fname}, Line: {lnum}')

def _get_padded_dem(
        ras: rasterio.io.DatasetReader,
        window: rasterio.windows.Window,
        target_band: int,
        pad: int
    ) -> numpy.ndarray:
    '''Get a padded rasterio dataset for topographical channels.'''

    # expand window within the original raster
    nw_x = max(window.col_off - pad, 0)
    nw_y = max(window.row_off - pad, 0)
    se_x = min(window.col_off + window.width + pad, ras.width)
    se_y = min(window.row_off + window.height + pad, ras.height)
    new_window = rasterio.windows.Window(nw_x, nw_y, se_x - nw_x, se_y - nw_y) # type: ignore

    # get expanded array using the new window
    expanded = ras.read(target_band, window=new_window)

    # get required padding on each side
    pad_top = max(0, pad - window.row_off)
    pad_left = max(0, pad - window.col_off)
    pad_bottom = max(0, (window.row_off + window.height + pad) - ras.height)
    pad_right = max(0, (window.col_off + window.width + pad) - ras.width)

    # pad the expanded arr accordingly controlled by pad_width
    expanded_padded = numpy.pad(
        array=expanded,
        pad_width=((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='reflect'
    )

    # return
    return expanded_padded

def get_valid_block_fpaths(
        block_dpath: str,
        valid_fpath: str,
        block_size: int,
        ratio_t: float,
        wat_thres: float,
        **kwargs
    ) -> list[str]:
    '''Get a list of file paths of the valid block with given thres.'''

    # kwargs
    overwrite = kwargs.get('overwrite', False)

    # read and exit if list already pickled to file and not overwrite
    if os.path.exists(valid_fpath) and not overwrite:
        log.log('INFO', f'Gathering valid blocks from: {valid_fpath}')
        v_fpaths = utils.funcs.load_pickle(valid_fpath)
        log.log('INFO', f'Fetched {len(v_fpaths)} valid blocks')
        return v_fpaths

    # otherwise create a new list
    log.log('INFO', f'Validating blocks at {block_dpath}')
    block_fpaths = utils.funcs.get_fpaths_from_dir(block_dpath, '.npz')
    valid_criteria = {
        'block_size': block_size,
        'pixel_threshold': ratio_t,
        'wat_threshold': wat_thres
    }
    jobs = [(_is_valid_block, (f, valid_criteria,), {}) for f in block_fpaths]
    rs: list[dict] = utils.multip.ParallelExecutor().run(jobs)
    v_fpaths = [r.get('valid', 0) for r in rs if r.get('valid', 0)]

    # save and return
    log.log('INFO', f'{len(v_fpaths)} valid blocks file paths saved: {valid_fpath}')
    utils.funcs.write_pickle(valid_fpath, v_fpaths)
    return v_fpaths

def _is_valid_block(
        block_fpath: str,
        criteria: dict[str, int | float]
    ) -> dict:
    '''Helper to flag whether a block is valid for downstream apps.'''

    # get meta from block
    meta = dataset.blocks.DataBlock().load_from_npz(block_fpath).data.meta

    blk_size = criteria.get('block_size', 256)
    px_thres = criteria.get('pixel_threshold', 0.75)
    wat_thres = criteria.get('water_threshold', 0.2)

    # keep only square blocks
    if meta['block_shape'] != [blk_size, blk_size]:
        return {'invalid': block_fpath}
    # valid pixel ratio threshold
    if meta['valid_pixel_ratio']['block'] < px_thres:
        return {'invalid': block_fpath}
    # water pixel ratio threshold - do the calc here for now
    wat_idx = next(
        (int(k) for k, v in meta['label1_reclass_name'].items() if v == 'water')
    )
    wat_ratio = meta['label_count']['layer1'][wat_idx - 1] / \
        sum(meta['label_count']['layer1'])
    if wat_ratio > wat_thres:
        return {'invalid': block_fpath}
    # if all checks passed
    return {'valid': block_fpath}
