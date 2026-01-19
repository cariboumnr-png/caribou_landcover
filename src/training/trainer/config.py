'''Module internal: trainer runtime state.'''

# standard imports
import dataclasses

# ---------------------------trainer runtime config---------------------------
@dataclasses.dataclass
class _DataConfig:
    dom_ids_name: str | None = None
    dom_vec_name: str | None = None

@dataclasses.dataclass
class _Schedule:
    '''Progression and scheduling'''
    max_epoch: int = -1                    # total epochs target
    max_step: int | None = None            # optional hard cap on steps (global)
    logging_interval: int = -1             # log every N steps (or batches)
    eval_interval: int | None = None       # evaluate every N steps (optional)
    checkpoint_interval: int | None = None # checkpoint every N steps (optional)
    patience_epochs: int | None = None     # for early stopping
    min_delta: float | None = None         # minimum improvement threshold

@dataclasses.dataclass
class _Monitor:
    enabled: tuple[str, ...] = tuple('iou') # e.g., ('iou', ...)
    metric: str = 'iou'                     # e.g., 'iou'
    head: str = ''                          # e.g., 'layer1' (as the parent layer)
    mode: str = 'max'                       # e.g., 'max'

@dataclasses.dataclass
class _Precision:
    '''Compute precision.'''
    use_amp: bool = True

@dataclasses.dataclass
class _OptimConfig:
    '''Optimization related.'''
    grad_clip_norm: float | None = 1.0

@dataclasses.dataclass
class RuntimeConfig:
    '''Minimal runtime config'''
    data: _DataConfig = dataclasses.field(default_factory=_DataConfig)
    schedule: _Schedule = dataclasses.field(default_factory=_Schedule)
    monitor: _Monitor = dataclasses.field(default_factory=_Monitor)
    precision: _Precision = dataclasses.field(default_factory=_Precision)
    optim: _OptimConfig = dataclasses.field(default_factory=_OptimConfig)
