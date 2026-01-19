'''Pipeline to prepare data from rasters for training.'''

# standard imports
import os
# third-party imports
import omegaconf
# local imports
import dataset.summary
import dataset.blocks.cache
import dataset.domain
import dataset.split
import dataset.stats

def build_cache_blocks(config: omegaconf.DictConfig) -> None:
    '''Create cache blocks from scratch'''

    # create cache dirs
    os.makedirs(config.paths.cache, exist_ok=True)
    os.makedirs(config.paths.blksdpath, exist_ok=True)

    # gather blocks from raw inputs
    dataset.blocks.cache.get_block_scheme(
        scheme_fpath=config.paths.blkscheme,
        input_ras=(config.inputs.image, config.inputs.label),
        block_size=config.blocks.size,
        overlap=config.blocks.overlap,
        rand_test=config.blocks.randtest,
        overwrite=config.overwrite.scheme
    )

    # clean-up routine to remove corrupted .npz files (to be recreated next)
    dataset.blocks.cache.clean_up_bad_npz(
        block_dpath=config.paths.blksdpath,
        to_clean=config.cleanup.clean_npz
    )

    # create block caches - first step, no image normalization
    dataset.blocks.cache.create_block_caches(
        scheme_fpath=config.paths.blkscheme,
        blk_dpath=config.paths.blksdpath,
        meta_fpath=config.inputs.meta,
        lbl_fpath=config.inputs.label,
        img_fpath=config.inputs.image,
        overwrite=config.overwrite.cache
    )

    # filter valid blocks and create a list of npz files
    dataset.blocks.cache.get_valid_block_fpaths(
        block_dpath=config.paths.blksdpath,
        valid_fpath=config.paths.blkvalid,
        block_size=config.blocks.size,
        ratio_t=config.filters.pxthres,
        wat_thres=config.filters.watthres,
        overwrite=config.overwrite.valid
    )

def parse_domain(
        config: omegaconf.DictConfig,
        domain_config: list[dict] | None
    ) -> None:
    '''Parse domain knowledge if provided.'''

    dataset.domain.parse(
        valid_fpath=config.paths.blkvalid,
        scheme_fpath=config.paths.blkscheme,
        domain_config=domain_config,
        domain_fpath=config.paths.domain,
        overwrite=config.overwrite.domain
    )

def split_datasets(
        config: omegaconf.DictConfig,
        score_params: dict,
        valselect_param: dict
    ) -> None:
    '''Split blocks into train/validation/test sets.'''

    # count classes from all blocks
    dataset.stats.count_label_classes(
        blkslist_fpath=config.paths.blkvalid,
        count_fpath=config.paths.lblcountg, # ..g for gloabl
        overwrite=config.overwrite.count
    )

    # score all blocks by class distribution from selected layer
    dataset.stats.score_blocks(
        blkscore_fpath=config.paths.blkscore,
        blkvalid_fpath=config.paths.blkvalid,
        score_param=score_params,
        global_count_fpath=config.paths.lblcountg,
        overwrite=config.overwrite.score
    )

    dataset.split.run(
        scores_fpath=config.paths.blkscore,
        v_fpath=config.paths.dataval,
        t_fpath=config.paths.datatrain,
        valselect_param=valselect_param,
        overwrite=config.overwrite.split
    )

def normalize_datasets(config: omegaconf.DictConfig) -> None:
    '''Aggragate block stats'''

    # count classes from training blocks
    dataset.stats.count_label_classes(
        blkslist_fpath=config.paths.datatrain,
        count_fpath=config.paths.lblcountt, # ..t for training
        overwrite=config.overwrite.count
    )

    # aggregate stats from training blocks
    dataset.stats.get_image_stats(
        blkslist_fpath=config.paths.datatrain,
        stats_fpath=config.paths.imgstats,
        overwrite=config.overwrite.stats
    )

    # normalize all valid blocks using the aggregated stats from training data
    dataset.stats.normalize_blocks(
        blkslist_fpath=config.paths.blkvalid,
        stats_fpath=config.paths.imgstats,
        update_norm=config.normalize.update_norm,
        overwrite=config.overwrite.norm
    )

def run(config: omegaconf.DictConfig) -> dataset.summary.DataSummary:
    '''Data preparation pipeline.'''

    dom_cfg, score_params, valselect = _validate_config(config)

    # if to run the whole process
    if not config.skip_dataprep:

        # get all valid blocks
        build_cache_blocks(config)

        # parse domain knowledge if provided
        parse_domain(config, dom_cfg)

        # get training blocks
        split_datasets(config, score_params, valselect)

        # use stats from training blocks to normalize all valid blocks
        normalize_datasets(config)

    # return blocks metadata
    return dataset.summary.generate(
        validblks_fpath=config.paths.blkvalid,
        train_lblstats_fpath=config.paths.lblcountt,
        train_datablks_fpaths=config.paths.datatrain,
        val_datablks_fpaths=config.paths.dataval,
        domain_fpath=config.paths.domain
    )

def _validate_config(config: omegaconf.DictConfig):
    '''doc'''

    dom = omegaconf.OmegaConf.to_container(config.inputs.domain, resolve=True)
    if dom is not None:
        assert isinstance(dom, list)
        assert all(isinstance(x, dict) for x in dom)

    score = omegaconf.OmegaConf.to_container(config.scoring, resolve=True)
    assert isinstance(score, dict)

    val = omegaconf.OmegaConf.to_container(config.valselect, resolve=True)
    assert isinstance(val, dict)

    return dom, score, val
