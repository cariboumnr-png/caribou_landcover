'''Get dataloaders'''

# standard imports
import dataclasses
import random
# third-party imports
import torch
import torch.utils.data
# local imports
import training.common
import training.dataloading
import utils.logger

log = utils.logger.Logger(name='dldrs')

@dataclasses.dataclass
class DataLoaders:
    '''doc'''
    train: torch.utils.data.DataLoader
    val: torch.utils.data.DataLoader

@dataclasses.dataclass(frozen=True)
class _LoaderConfig:
    '''Static config for `dataloader.py`.'''
    block_size: int
    patch_size: int
    batch_size: int
    stream_cache: bool
    rand_sample_n: int | None = None
    rand_seed: int | None = None

def parse_loader_config(
        block_size: int,
        patch_size: int,
        batch_size: int,
        stream_cache: bool,
        **kwargs
    ) -> _LoaderConfig:
    '''Generate loader config for dataloaders.'''

    rand_sample_n = kwargs.get('rand_sample_n', None)
    rand_seed = kwargs.get('rand_seed', None)
    return _LoaderConfig(
        block_size=block_size,
        patch_size=patch_size,
        batch_size=batch_size,
        stream_cache=stream_cache,
        rand_sample_n=rand_sample_n,
        rand_seed=rand_seed,
    )

def get_dataloaders(
        data_summary: training.common.DataSummaryLoader,
        loader_config: _LoaderConfig
    ) -> DataLoaders:
    '''Entry to the module, returns two dataloaders for training.'''

    #
    t_fpaths = list(data_summary.data.train)
    v_fpaths = list(data_summary.data.val)
    domain_dict = data_summary.dom.data

    # parse args from config
    block_size = loader_config.block_size
    patch_size = loader_config.patch_size
    batch_size = loader_config.batch_size
    stream_blk = loader_config.stream_cache
    rand_sample_n = loader_config.rand_sample_n
    rand_seed = loader_config.rand_seed

    # if this is for a test sample randomly pick files from each
    if rand_sample_n:
        random.seed(rand_seed)
        t_fpaths = random.sample(t_fpaths, rand_sample_n)
        v_fpaths = random.sample(v_fpaths, max(1, rand_sample_n // 5))

    # get training data loader
    d = training.dataloading.MultiBlockDataset(
            fpaths=t_fpaths,
            blk_cfg=training.dataloading.BlockConfig(
                augment_flip=True,      # flip for training data
                block_size=block_size,
                patch_size=patch_size,
                domain_dict=domain_dict
            ),
            preload=False,              # no preload for larger training data
            blk_cache_num=stream_blk  # max stream size
        )
    t_loader = torch.utils.data.DataLoader(
        dataset=d,
        batch_size=batch_size,
        shuffle=True,                    # shuffle for training data
        collate_fn=_collate_multi_block
    )
    log.log('INFO', f'Training dataset length: {len(t_loader)}')

    # get validation data loader
    d = training.dataloading.MultiBlockDataset(
            fpaths=v_fpaths,
            blk_cfg=training.dataloading.BlockConfig(
                augment_flip=False,     # no flip for validation data
                block_size=block_size,
                patch_size=patch_size,
                domain_dict=domain_dict
            ),
            preload=bool(rand_sample_n == 0),# preload if not a test
            blk_cache_num=stream_blk  # max stream size
        )
    v_loader = torch.utils.data.DataLoader(
        dataset=d,
        batch_size=batch_size,
        shuffle=False,                   # no shuffle for validation data
        collate_fn=_collate_multi_block
    )
    log.log('INFO', f'Validation dataset length: {len(v_loader)}')

    # return
    return DataLoaders(t_loader, v_loader)

def _collate_multi_block(batch):
    '''Customized collate function to properly stack a batch.'''

    # unpack batch as a list
    xs, ys, doms = zip(*batch)

    # x -> [B, C, H, W]
    xs = torch.stack(xs, dim=0)

    # y_dict -> [B, C, H, W]
    ys = torch.stack(ys, dim=0)

    # domain -> dict[str, [B, V]] or dict[str, [B]]
    dom_out = {}
    first_dom = doms[0]
    for key in first_dom.keys():
        dom_out[key] = torch.stack([d[key] for d in doms], dim=0)


    # return
    return xs, ys, dom_out
