'''Metrics config validation.'''

# standard imports
import typing
# local imports
import training.metrics

def is_cm_config(
        cfg: dict[str, int | None]
    ) -> typing.TypeGuard[training.metrics.ConfusionMatrixConfig]:
    '''Validate confusion matrix calc config dict.'''

    def _is_parent_class(v):
        return isinstance(v, int) or v is None

    def _is_exlcude_class(v):
        return (
            isinstance(v, tuple) and len(v) == 0 or
            (
                isinstance(v, tuple) and len(v) > 0 and
                all(isinstance(x, int) for x in v)
            )
        )

    return(
        isinstance(cfg, dict) and
        isinstance(cfg.get('num_classes'), int) and
        isinstance(cfg.get('ignore_index'), int) and
        _is_parent_class(cfg.get('parent_class_1b')) and
        _is_exlcude_class(cfg.get('exclude_class_1b'))
    )
