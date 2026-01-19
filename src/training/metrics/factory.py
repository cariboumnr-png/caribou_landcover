'''Validation metrics calculation functions for trainer.'''

# local imports
import training.common
import training.metrics

class HeadMetrics:
    '''
    Typed wrapper around a mapping of heads to `ConfusionMatrix` objects.

    This class provides:
    - key-based access to individual `ConfusionMatrix` instances.
    - a stable, strongly-typed container for passing head specs through
    the codebase.

    It is *not* a full `dict` replacement. To work with the underlying
    mapping directly, use method: `as_dict()`.
    '''

    def __init__(self, specs: dict[str, training.metrics.ConfusionMatrix]):
        self._specs = specs

    def __getitem__(self, key: str) -> training.common.MetricLike:
        return self._specs[key]

    def __len__(self) -> int:
        return len(self._specs)

    def as_dict(self) -> dict[str, training.common.MetricLike]:
        '''Return a shallow copy of the mapping as `dict[str, Spec]`.'''
        return dict(self._specs)

def build_headmetrics(
        headspecs: training.common.HeadSpecsLike,
        ignore_index: int
    ) -> HeadMetrics:
    '''Generate concreate head metric classes indexed by head name.'''

    out: dict[str, training.metrics.ConfusionMatrix] = {}
    # iterate through configs
    for hname, hspec in headspecs.as_dict().items():
        # attach to output list
        config = {
            'num_classes': hspec.num_classes,
            'ignore_index': ignore_index,
            'parent_class_1b': hspec.parent_cls,
            'exclude_class_1b': hspec.exclude_cls
        }
        out[hname] = training.metrics.ConfusionMatrix(config)
    return HeadMetrics(out)
