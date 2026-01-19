'''Optimizer factory.'''

# standard imports
import dataclasses
# third-party imports
import torch
# local imports
import training.optim

@dataclasses.dataclass
class Optimization:
    '''Wrapper for optimization components'''
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None

@dataclasses.dataclass
class _Config:
    '''doc'''
    # optimizer config
    opt_cls: str = 'AdamW' # default
    lr: float = 1e-4
    weight_decay: float = 1e-3
    # optional scheduler config
    sched_cls: str | None = 'CosAnneal' # default
    sched_args: dict = dataclasses.field(default_factory=dict)

# add more if needed
_OPTIMIZERS: dict[str, training.optim.OptimizerFactory] = {
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
}
_SCHEDULERS: dict[str, training.optim.SchedulerFactory] = {
    "CosAnneal": torch.optim.lr_scheduler.CosineAnnealingLR,
    "OneCycle": torch.optim.lr_scheduler.OneCycleLR,
}

# public functions
def build_optimizer(
        model: training.optim.ModelWithParams,
        cfg: _Config
    ) -> torch.optim.Optimizer:
    '''
    Docstring for build_optimizer

    :param model: Description
    :param cfg: Description
    '''

    opt_cls = _OPTIMIZERS.get(cfg.opt_cls)
    if opt_cls is None:
        raise ValueError(f"Unknown optimizer: {cfg.opt_cls}")
    return opt_cls(
        params=model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

def build_scheduler(
        optimizer: torch.optim.Optimizer,
        cfg: _Config
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
    '''
    Docstring for build_scheduler

    :param optimizer: Description
    :param cfg: Description
    '''
    if cfg.sched_cls is None:
        return None
    sched_cls = _SCHEDULERS.get(cfg.sched_cls)
    if sched_cls is None:
        raise ValueError(f"Unknown scheduler: {cfg.sched_cls}")
    return sched_cls(optimizer=optimizer, **cfg.sched_args)

def build_optim_config(
    opt_cls: str,
    lr: float,
    weight_decay: float,
    sched_cls: str,
    sched_args: dict
) -> _Config:
    '''Build optimization config'''
    return _Config(opt_cls, lr, weight_decay, sched_cls, sched_args)

def build_optimization(
        model: training.optim.ModelWithParams,
        config: _Config
    ) -> Optimization:
    '''Wrapper'''

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    return Optimization(optimizer=optimizer, scheduler=scheduler)
