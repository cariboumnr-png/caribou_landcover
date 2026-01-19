'''Collection of external protocols for trainer components.'''

# standard imports
import dataclasses
# local imports
import training.common

# ---------------------------trainer runtime config---------------------------
@dataclasses.dataclass
class TrainerComponents:
    '''Trainer components protocol.'''
    model: training.common.MultiHeadTrainable
    dataloaders: training.common.DataLoadersLike
    headspecs: training.common.HeadSpecsLike
    headlosses: training.common.HeadLossesLike
    headmetrics: training.common.HeadMetricsLike
    optimization: training.common.OptimizationLike
    callbacks: training.common.CallBacksLike
