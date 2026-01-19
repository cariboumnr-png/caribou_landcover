'''Common protocols'''

from .data import DataSummaryFull
from .data import DataSummaryHeads
from .data import DataSummaryLoader

from .trainer import TrainerLike
from .trainer_comps import (
    TrainerComponentsLike,
    MultiHeadTrainable,
    DataLoadersLike,
    HeadSpecsLike,
    SpecLike,
    HeadLossesLike,
    CompositeLossLike,
    HeadMetricsLike,
    MetricLike,
    OptimizationLike,
    CallBacksLike,
    ProgressCallbackLike,
    TrainCallbackLike,
    ValCallbackLike,
    LoggingCallbackLike
)
from .trainer_config import (
    RuntimeConfigLike,
    DataConfigLike,
    ScheduleLike,
    MonitorLike,
    PrecisionLike,
    OptimConfigLike
)
from .trainer_state import RuntimeStateLike

__all__ = [
    #
    'TrainerLike',
    #
    'TrainerComponentsLike',
    'MultiHeadTrainable',
    'DataLoadersLike',
    'HeadSpecsLike',
    'SpecLike',
    'HeadLossesLike',
    'CompositeLossLike',
    'HeadMetricsLike',
    'MetricLike',
    'OptimizationLike',
    #
    'RuntimeConfigLike',
    'DataConfigLike',
    'ScheduleLike',
    'MonitorLike',
    'PrecisionLike',
    'OptimConfigLike',
    #
    'RuntimeStateLike',
    #
    'DataSummaryFull',
    'DataSummaryHeads',
    'DataSummaryLoader',
    #
    'CallBacksLike',
    'ProgressCallbackLike',
    'TrainCallbackLike',
    'ValCallbackLike',
    'LoggingCallbackLike'
]
